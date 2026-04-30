import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ==========================================
# [참가자 작성 구간] 모델 정의
# ==========================================

import math
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """MPS 및 모든 디바이스에서 안전하게 동작하는 실수-허수 분리형 FNO 레이어"""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.classifier(x)


# ==========================================
# 데이터셋 정의 (수정 불필요)
# ==========================================
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return img_name, image


# [필수 1] 학습 시와 동일한 CT 이미지 스케일링 전처리 클래스 추가
class CTMinMaxTransform:
    def __call__(self, img):
        if img.size != (28, 28):
            img = img.resize((28, 28), Image.BILINEAR)
        tensor = transforms.functional.to_tensor(img)

        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        else:
            tensor = torch.zeros_like(tensor)

        return (tensor - 0.5) / 0.5


# ==========================================
# 메인 실행 함수
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Hackathon Inference Script")
    parser.add_argument(
        "--input_dir", type=str, default="./images", help="Path to test images"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="./model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./submission.csv",
        help="Path to save results",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. [참가자 작성 구간] 모델 인스턴스 생성
    model = HybridFNONet(num_classes=11, channels=64, modes=12)

    # 2. 가중치 로드
    try:
        checkpoint = torch.load(args.weight_path, map_location=device)

        # [필수 2] 우리가 저장한 "model_state_dict" 키 대응 추가
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "net" in checkpoint:
            model.load_state_dict(checkpoint["net"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Successfully loaded weights from {args.weight_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device).eval()

    # 3. 전처리 정의 (CT용 Min-Max 스케일링으로 변경)
    transform = CTMinMaxTransform()

    # 4. 데이터 로더
    if not os.path.exists(args.input_dir):
        print(f"Directory not found: {args.input_dir}")
        return

    dataset = TestDataset(args.input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    # 클래스 이름 및 순서 정의 (변경 금지)
    class_names = [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen",
    ]

    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for filenames, images in loader:
            images = images.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            for i in range(len(filenames)):
                img_id = os.path.splitext(filenames[i])[0]

                res = {"index": img_id}

                pred_idx = predicted[i].item()
                for idx, name in enumerate(class_names):
                    res[name] = 1 if idx == pred_idx else 0

                results.append(res)

    df = pd.DataFrame(results)
    cols = ["index"] + class_names
    df = df[cols]
    df.to_csv(args.output_csv, index=False)
    print(f"Inference complete! Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
