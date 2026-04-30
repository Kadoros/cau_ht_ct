"""
model.py
Final Hybrid Spectral-Spatial Encoder (RTX 3090 Optimized) with DANN & CoordConv.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Gradient Reversal Layer (for DANN)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Spectral Convolution & FNO Block
# ──────────────────────────────────────────────
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

        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


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


# ──────────────────────────────────────────────
# Main Model Structure
# ──────────────────────────────────────────────
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

        self.embedding = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )

        self.classifier = nn.Linear(channels * 2, num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2), nn.GELU(), nn.Linear(channels * 2, 3)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
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

        x = x.mean(dim=(2, 3))
        return self.embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_feature(x))

    def forward_domain(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        feat = self.get_feature(x)
        feat_rev = grad_reverse(feat, alpha)
        return self.domain_classifier(feat_rev)


def build_model(
    num_classes: int = 11, channels: int = 64, modes: int = 12
) -> HybridFNONet:
    return HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
