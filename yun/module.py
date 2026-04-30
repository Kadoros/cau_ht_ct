import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [팀원 파트] FNO 블록 (수정 금지) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
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
    def complex_mul2d_safe(x_ft, w_real, w_imag):
        xr, xi = x_ft.real, x_ft.imag
        or_ = torch.einsum("bixy,ioxy->boxy", xr, w_real) - torch.einsum(
            "bixy,ioxy->boxy", xi, w_imag
        )
        oi = torch.einsum("bixy,ioxy->boxy", xr, w_imag) + torch.einsum(
            "bixy,ioxy->boxy", xi, w_real
        )
        return torch.complex(or_, oi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            bsz,
            self.weights1_real.size(1),
            h,
            w // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
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


# --- [네 파트] 공간 인코더 고도화 모듈 ---
class CBAM(nn.Module):
    """채널과 공간 어텐션의 결합 (Woo et al., 2018)"""

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        res = torch.cat(
            [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1
        )
        return x * self.sa(res)


class MultiScaleDilatedBlock(nn.Module):
    """다양한 장기 크기 대응 (Yu & Koltun, 2016)"""

    def __init__(self, channels):
        super().__init__()
        c = channels // 4
        self.d1 = nn.Conv2d(channels, c, 3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(channels, c, 3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(channels, c, 3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(
            channels, channels - 3 * c, 3, padding=8, dilation=8, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], 1)
        return F.gelu(self.bn(out) + x)


class HybridFNONet(nn.Module):
    def __init__(self, num_classes: int = 11, channels: int = 64, modes: int = 12):
        super().__init__()
        self.lifting = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fno1 = FNOBlock(channels, modes)
        self.fno2 = FNOBlock(channels, modes)

        # 네 핵심 역할: 공간 특징 정제 및 어텐션
        self.spatial_refine = MultiScaleDilatedBlock(channels)
        self.feature_expand = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
        )
        self.cbam = CBAM(channels * 2)

        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fno2(self.fno1(self.lifting(x)))
        x = self.cbam(self.feature_expand(self.spatial_refine(x)))
        return self.classifier(x.mean(dim=(2, 3)))


def build_model(
    num_classes: int = 11, channels: int = 64, modes: int = 12
) -> HybridFNONet:
    return HybridFNONet(num_classes, channels, modes)
