"""
model.py
Hybrid Spectral-Spatial Encoder for 28x28 grayscale CT images.
- FNO (Fourier Neural Operator) blocks for geometry/topology learning
- GroupNorm instead of BatchNorm (safe for few-shot fine-tuning)
- 11-class organ classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Spectral Convolution (Fourier Layer)
# ──────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT -> linear transform in freq domain -> IFFT
    modes: number of Fourier modes to keep (low-freq = geometric info)
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # how many freq modes to keep

        # Complex weights for 4 quadrant combinations
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def complex_mul2d(self, x_ft, weights_real, weights_imag):
        # x_ft: (B, C_in, H, W//2+1) complex
        # weights: (C_in, C_out, modes, modes)
        w = torch.complex(weights_real, weights_imag)  # (Cin, Cout, m, m)
        # einsum: batch × in_ch × h × w -> batch × out_ch × h × w
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x):
        B, C, H, W = x.shape

        # MPS (Apple Silicon)는 complex tensor / FFT 미지원
        # → FFT 연산만 CPU로 내렸다가 결과를 원래 device로 복귀
        orig_device = x.device
        is_mps = str(orig_device).startswith("mps")
        if is_mps:
            x = x.cpu()
            wr = self.weights_real.cpu()
            wi = self.weights_imag.cpu()
            fft_device = torch.device("cpu")
        else:
            wr, wi = self.weights_real, self.weights_imag
            fft_device = orig_device

        # FFT
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Allocate output in freq domain
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=fft_device
        )

        # Apply learned weights to low-freq modes only
        m = self.modes
        out_ft[:, :, :m, :m] = self.complex_mul2d(x_ft[:, :, :m, :m], wr, wi)

        # IFFT back to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        # 원래 device로 복귀
        return x_out.to(orig_device)


# ──────────────────────────────────────────────
# FNO Block
# ──────────────────────────────────────────────
class FNOBlock(nn.Module):
    """
    Single FNO block:
      Spectral path  (Fourier conv)
    + Spatial path   (1x1 conv bypass)
    -> GroupNorm -> GELU
    """

    def __init__(self, channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────
class HybridFNONet(nn.Module):
    """
    Architecture:
      Lifting Conv (1 -> 32 channels)
      FNO Block x2
      Spatial refinement (3x3 conv)
      Global Average Pooling
      Classifier (32 -> 11)

    Design choices:
      - 32 channels (not 64) to reduce overfitting on 50-shot data
      - GroupNorm everywhere (safe for small batch / fine-tuning)
      - modes=8 (fast enough, still captures geometry at 28x28)
    """

    def __init__(self, num_classes=11, channels=64, modes=12):
        super().__init__()
        self.channels = channels

        # ── Lifting: 1 -> channels ──
        self.lifting = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
        )

        # ── FNO Blocks ──
        self.fno1 = FNOBlock(channels, modes)
        self.fno2 = FNOBlock(channels, modes)

        # ── Spatial refinement ──
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.lifting(x)  # (B, C, 28, 28)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling -> (B, C)
        x = self.classifier(x)  # (B, 11)
        return x

    def get_feature(self, x):
        """Return embedding before classifier (for analysis)."""
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        return x.mean(dim=[2, 3])


def build_model(num_classes=11, channels=64, modes=12):
    model = HybridFNONet(num_classes=num_classes, channels=channels, modes=modes)
    return model
