import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels, self.out_channels, self.modes = (
            in_channels,
            out_channels,
            int(modes),
        )
        scale = 1.0 / (in_channels * out_channels)
        self.weights1_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights1_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights2_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights2_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    @staticmethod
    def complex_mul2d_safe(x_ft, w_real, w_imag):
        xr, xi = x_ft.real, x_ft.imag
        return torch.complex(
            torch.einsum("bixy,ioxy->boxy", xr, w_real)
            - torch.einsum("bixy,ioxy->boxy", xi, w_imag),
            torch.einsum("bixy,ioxy->boxy", xr, w_imag)
            + torch.einsum("bixy,ioxy->boxy", xi, w_real),
        )

    def forward(self, x):
        bsz, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            bsz, self.out_channels, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device
        )
        m = min(self.modes, h // 2, w // 2 + 1)
        if m > 0:
            out_ft[:, :, :m, :m] = self.complex_mul2d_safe(
                x_ft[:, :, :m, :m],
                self.weights1_real[:, :, :m, :m],
                self.weights1_imag[:, :, :m, :m],
            )
            out_ft[:, :, -m:, :m] = self.complex_mul2d_safe(
                x_ft[:, :, -m:, :m],
                self.weights2_real[:, :, :m, :m],
                self.weights2_imag[:, :, :m, :m],
            )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, channels, modes, dropout=0.0, residual_scale=0.5):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.residual_scale * self.dropout(
            F.gelu(self.norm(self.spectral(x) + self.bypass(x)))
        )


class DANN_HybridFNONet(nn.Module):
    def __init__(
        self, num_classes=11, num_domains=3, channels=64, modes=12, dropout=0.3
    ):
        super().__init__()
        self.lifting = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fno1 = FNOBlock(channels, modes)
        self.fno2 = FNOBlock(channels, modes)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.feature_expand = nn.Sequential(
            nn.Conv2d(
                channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_domains),
        )

    def forward(self, x, alpha=1.0):
        b, _, h, w = x.shape
        grid_h = (
            torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(x.device)
        )
        grid_w = (
            torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(x.device)
        )
        x = torch.cat([x, grid_h, grid_w], dim=1)
        feat = self.feature_expand(
            self.spatial_refine(self.fno2(self.fno1(self.lifting(x))))
        ).mean(dim=(2, 3))
        class_output = self.classifier(feat)
        reverse_feat = GradientReversalLayer.apply(feat, alpha)
        domain_output = self.domain_classifier(reverse_feat)
        return class_output, domain_output


def build_model(num_classes=11, num_domains=3, channels=64, modes=12):
    return DANN_HybridFNONet(
        num_classes=num_classes, num_domains=num_domains, channels=channels, modes=modes
    )
