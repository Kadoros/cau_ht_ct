import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ══════════════════════════════════════════════════════════
# DANN Modules (Integrated)
# ══════════════════════════════════════════════════════════


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),  # 3 domains: Axial, Coronal, Sagittal
        )

    def forward(self, x):
        return self.discriminator(x)


# ══════════════════════════════════════════════════════════
# Original Model Architecture
# ══════════════════════════════════════════════════════════


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = int(modes)
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
    def complex_mul2d_safe(
        x_ft: torch.Tensor, w_real: torch.Tensor, w_imag: torch.Tensor
    ) -> torch.Tensor:
        x_real, x_imag = x_ft.real, x_ft.imag
        out_real = torch.einsum("bixy,ioxy->boxy", x_real, w_real) - torch.einsum(
            "bixy,ioxy->boxy", x_imag, w_imag
        )
        out_imag = torch.einsum("bixy,ioxy->boxy", x_real, w_imag) + torch.einsum(
            "bixy,ioxy->boxy", x_imag, w_real
        )
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        device = x.device
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=device,
        )
        m = min(self.modes, height // 2, width // 2 + 1)
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
        x_out = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x_out


class FNOBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        modes: int,
        dropout: float = 0.0,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.bypass(x)
        y = F.gelu(self.norm(y))
        y = self.dropout(y)
        return x + self.residual_scale * y


class HybridFNONet(nn.Module):
    def __init__(
        self,
        num_classes: int = 11,
        channels: int = 64,
        modes: int = 12,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.modes = modes
        self.lifting = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fno1 = FNOBlock(channels, modes, residual_scale=0.5)
        self.fno2 = FNOBlock(channels, modes, residual_scale=0.5)
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
            nn.Dropout(classifier_dropout),
            nn.Linear(channels * 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, h, w = x.size()
        grid_h = (
            torch.linspace(-1, 1, h)
            .view(1, 1, h, 1)
            .expand(batch_size, 1, h, w)
            .to(x.device)
        )
        grid_w = (
            torch.linspace(-1, 1, w)
            .view(1, 1, 1, w)
            .expand(batch_size, 1, h, w)
            .to(x.device)
        )
        x = torch.cat([x, grid_h, grid_w], dim=1)
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = self.feature_expand(x)
        features = x.mean(dim=(2, 3))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model(
    num_classes: int = 11, channels: int = 64, modes: int = 12
) -> HybridFNONet:
    return HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
