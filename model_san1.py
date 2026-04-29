"""
model.py
Hybrid Spectral-Spatial Encoder for 28x28 grayscale CT images.

Key fixes compared with the initial version:
- Uses both positive and negative height Fourier modes in SpectralConv2d.
- Clamps modes safely to the actual input resolution.
- Uses residual FNO blocks for more stable training / fine-tuning.
- Uses GroupNorm with an automatic valid group count.
- Keeps the public build_model() API used by train.py and inference.py.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def _valid_num_groups(channels: int, preferred: int = 8) -> int:
    """Return a GroupNorm group count that divides channels."""
    preferred = min(preferred, channels)
    for g in range(preferred, 0, -1):
        if channels % g == 0:
            return g
    return 1


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_valid_num_groups(channels, preferred_groups), num_channels=channels)


# ──────────────────────────────────────────────
# Spectral Convolution (Fourier Layer)
# ──────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer.

    rFFT keeps the non-negative frequency half in the width dimension. For the
    height dimension, both low positive modes (:m) and low negative modes (-m:)
    are useful. The initial code only used :m, which unnecessarily reduced the
    spectral path's expressiveness.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = int(modes)

        # Complex weights for height positive modes and height negative modes.
        # Small initialization keeps FFT path stable at the beginning of training.
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights1_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weights1_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weights2_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weights2_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))

    @staticmethod
    def complex_mul2d(x_ft: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor) -> torch.Tensor:
        """
        x_ft:   (B, C_in, Hm, Wm), complex
        weights:(C_in, C_out, Hm, Wm), real/imag tensors
        returns:(B, C_out, Hm, Wm), complex
        """
        weights = torch.complex(weights_real, weights_imag)
        return torch.einsum("bixy,ioxy->boxy", x_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape

        # Keep the MPS fallback for local Apple Silicon experiments. The official
        # evaluation server is CUDA, so this branch is not used there.
        orig_device = x.device
        is_mps = str(orig_device).startswith("mps")
        if is_mps:
            x = x.cpu()
            fft_device = torch.device("cpu")
            w1r, w1i = self.weights1_real.cpu(), self.weights1_imag.cpu()
            w2r, w2i = self.weights2_real.cpu(), self.weights2_imag.cpu()
        else:
            fft_device = orig_device
            w1r, w1i = self.weights1_real, self.weights1_imag
            w2r, w2i = self.weights2_real, self.weights2_imag

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=fft_device,
        )

        # Avoid mode overlap in height and avoid indexing beyond rFFT width.
        m = min(self.modes, height // 2, width // 2 + 1)
        if m > 0:
            out_ft[:, :, :m, :m] = self.complex_mul2d(
                x_ft[:, :, :m, :m], w1r[:, :, :m, :m], w1i[:, :, :m, :m]
            )
            out_ft[:, :, -m:, :m] = self.complex_mul2d(
                x_ft[:, :, -m:, :m], w2r[:, :, :m, :m], w2i[:, :, :m, :m]
            )

        x_out = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x_out.to(orig_device)


# ──────────────────────────────────────────────
# FNO Block
# ──────────────────────────────────────────────
class FNOBlock(nn.Module):
    """
    Residual FNO block:
      spectral path + 1x1 spatial bypass -> GroupNorm -> GELU -> residual add
    """

    def __init__(self, channels: int, modes: int, dropout: float = 0.0, residual_scale: float = 0.5):
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


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────
class HybridFNONet(nn.Module):
    """
    Architecture for 28x28 single-channel CT slice classification.

    Default values are intentionally small because the target domain is few-shot:
    channels=32, modes=8. You can still override these from train.py.
    """

    def __init__(
        self,
        num_classes: int = 11,
        channels: int = 32,
        modes: int = 8,
        classifier_dropout: float = 0.25,
        fno_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.modes = modes

        self.lifting = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )

        self.fno1 = FNOBlock(channels, modes, dropout=fno_dropout, residual_scale=0.5)
        self.fno2 = FNOBlock(channels, modes, dropout=fno_dropout, residual_scale=0.5)

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(channels, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding before classifier."""
        return self.forward_features(x)


def build_model(num_classes: int = 11, channels: int = 32, modes: int = 8) -> HybridFNONet:
    return HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
