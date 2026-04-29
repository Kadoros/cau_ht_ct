import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)

        # ⚠️ MPS 에러 방지: 복소수 파라미터 대신 실수부/허수부 파라미터로 분리하여 생성
        self.w1_re = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.w1_im = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.w2_re = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.w2_im = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def complex_mul2d(self, x, w_re, w_im):
        # (B, C, H, W) 복소수 연산을 실수 연산으로 분해해서 수행
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        res_re = torch.einsum("bixy,ioxy->boxy", x.real, w_re) - torch.einsum(
            "bixy,ioxy->boxy", x.imag, w_im
        )
        res_im = torch.einsum("bixy,ioxy->boxy", x.real, w_im) + torch.einsum(
            "bixy,ioxy->boxy", x.imag, w_re
        )
        return torch.complex(res_re, res_im)

    def forward(self, x):
        B, C, H, W = x.shape
        orig_device = x.device

        # ── FFT 연산 ──
        # MPS는 Complex 타입을 지원하지 않으므로 무조건 CPU로 옮겨서 연산
        x_cpu = x.cpu()
        x_ft = torch.fft.rfft2(x_cpu, norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat)

        m = self.modes
        # 가중치를 복소수로 결합 (CPU에서 연산)
        w1 = torch.complex(self.w1_re.cpu(), self.w1_im.cpu())
        w2 = torch.complex(self.w2_re.cpu(), self.w2_im.cpu())

        out_ft[:, :, :m, :m] = self.complex_mul2d(x_ft[:, :, :m, :m], w1.real, w1.imag)
        out_ft[:, :, -m:, :m] = self.complex_mul2d(
            x_ft[:, :, -m:, :m], w2.real, w2.imag
        )

        # ── IFFT 연산 ──
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        # 결과를 다시 원래 장치(MPS)로 복귀
        return x_out.to(orig_device)


class FNOBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class HybridFNONet(nn.Module):
    def __init__(self, num_classes=11, channels=64, modes=12):
        super().__init__()
        self.lifting = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
        )
        self.fno1 = FNOBlock(channels, modes)
        self.fno2 = FNOBlock(channels, modes)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def build_model(num_classes=11, channels=64, modes=12):
    return HybridFNONet(num_classes, channels, modes)
