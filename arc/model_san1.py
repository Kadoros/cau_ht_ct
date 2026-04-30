import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_num_groups(channels: int, preferred: int = 8) -> int:
    preferred = min(preferred, channels)
    for g in range(preferred, 0, -1):
        if channels % g == 0:
            return g
    return 1


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_valid_num_groups(channels, preferred_groups), num_channels=channels
    )


class SpectralConv2d(nn.Module):
    """MPS 및 모든 디바이스에서 안전하게 동작하는 실수-허수 분리형 FNO 레이어"""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = int(modes)

        # 주파수 도메인 초기화 최적화 (Kaiming Normal 유사)
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
        """
        MPS 복소수 einsum 에러를 방지하기 위한 실수/허수 분리 계산
        (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        """
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
        self.norm = _group_norm(channels)
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
        channels: int = 32,
        modes: int = 8,
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
            _group_norm(channels),
            nn.GELU(),
        )

        # [수정] 차원 병목 해결: 채널 수 2배 확장 및 공간 크기 축소 (14x14)
        self.feature_expand = nn.Sequential(
            nn.Conv2d(
                channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
            _group_norm(channels * 2),
            nn.GELU(),
        )

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
        x = self.spatial_refine(x)
        x = self.feature_expand(x)  # (B, 64, 14, 14)
        x = x.mean(dim=(2, 3))  # Global Average Pooling
        return self.classifier(x)


def build_model(
    num_classes: int = 11, channels: int = 32, modes: int = 8
) -> HybridFNONet:
    return HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
