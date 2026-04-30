import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# DANN을 위한 그래디언트 반전 계층: 뷰(View) 편향을 제거합니다.[span_5](start_span)[span_5](end_span)
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class SpectralConv2d(nn.Module):
    """실수-허수 분리형 FNO 레이어: 전역적 위상 구조를 학습합니다.[span_6](start_span)[span_6](end_span)"""

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
    def __init__(self, channels: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.bypass(x)
        return x + 0.5 * self.dropout(F.gelu(self.norm(y)))


class HybridFNONet(nn.Module):
    def __init__(self, num_classes=11, channels=64, modes=12):
        super().__init__()
        self.num_classes = num_classes
        # CoordConv: X, Y 좌표 정보를 추가하여 좌우 장기 구분 능력을 강화합니다.[span_7](start_span)[span_7](end_span)
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
        self.embedding = nn.Sequential(
            nn.Linear(channels * 2, channels * 2), nn.GELU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(channels * 2, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2), nn.GELU(), nn.Linear(channels * 2, 3)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(
                m.weight
            )  # 직교 초기화로 특징 공간 분리 최적화[span_8](start_span)[span_8](end_span)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.size()
        gh = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(x.device)
        gw = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(x.device)
        x = torch.cat(
            [x, gh, gw], dim=1
        )  # CoordConv 적용[span_9](start_span)[span_9](end_span)
        x = self.feature_expand(
            self.spatial_refine(self.fno2(self.fno1(self.lifting(x))))
        )
        return self.embedding(x.mean(dim=(2, 3)))

    def forward(self, x):
        return self.classifier(self.get_feature(x))

    def forward_domain(self, x, alpha):
        feat = self.get_feature(x)
        return self.domain_classifier(grad_reverse(feat, alpha))


def build_model(num_classes=11, channels=64, modes=12):
    return HybridFNONet(num_classes, channels, modes)
