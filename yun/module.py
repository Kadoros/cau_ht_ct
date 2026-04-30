import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ─────────────────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)  # 어떤 특징(What)이 중요한가?
        x = self.sa(x)  # 어느 위치(Where)가 중요한가?
        return x


# ─────────────────────────────────────────────────────────
# Dilated Convolution 기반 다중 스케일 정제
# ─────────────────────────────────────────────────────────
class MultiScaleDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 다양한 크기의 장기를 잡기 위해 수용 영역(Receptive Field)을 3단계로 분할
        c_part = channels // 3
        self.conv_d1 = nn.Conv2d(
            channels, c_part, kernel_size=3, padding=1, dilation=1, bias=False
        )
        self.conv_d2 = nn.Conv2d(
            channels, c_part, kernel_size=3, padding=2, dilation=2, bias=False
        )
        self.conv_d3 = nn.Conv2d(
            channels,
            channels - 2 * c_part,
            kernel_size=3,
            padding=3,
            dilation=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out1 = self.conv_d1(x)  # 작은 특징 (비장, 혈관 등)
        out2 = self.conv_d2(x)  # 중간 특징 (신장 등)
        out3 = self.conv_d3(x)  # 거대 특징 (간, 폐 등)
        out = torch.cat([out1, out2, out3], dim=1)
        return F.gelu(self.bn(out) + x)  # Residual Connection 포함


# ─────────────────────────────────────────────────────────
# 메인 네트워크 조립
# ─────────────────────────────────────────────────────────
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

        # 1채널 흑백 입력을 바로 받음 (CoordConv 폐기)
        self.lifting = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.fno1 = FNOBlock(channels, modes, residual_scale=0.5)
        self.fno2 = FNOBlock(channels, modes, residual_scale=0.5)

        # 단일 Conv 대신 다중 스케일(Dilated) 정제 블록 사용
        self.spatial_refine = MultiScaleDilatedBlock(channels)

        self.feature_expand = nn.Sequential(
            nn.Conv2d(
                channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
        )

        # 차원 확장 직후 핵심 특징과 픽셀 위치를 찾는 CBAM 어텐션 적용
        self.attention = CBAMBlock(channels * 2)

        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)

        # 공간 인코딩 파트
        x = self.spatial_refine(x)  # 1. 다중 스케일 문맥 스캔
        x = self.feature_expand(x)  # 2. 14x14로 압축하며 특징 128채널로 뻥튀기
        x = self.attention(x)  # 3. CBAM으로 핵심 장기 위치와 특징 맵에 어텐션 집중

        x = x.mean(dim=(2, 3))  # 4. Global Average Pooling
        return self.classifier(x)


def build_model(
    num_classes: int = 11, channels: int = 64, modes: int = 12
) -> HybridFNONet:
    return HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
