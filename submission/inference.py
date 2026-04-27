import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ==========================================
# [참가자 작성 구간] 모델 정의
# ==========================================


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
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
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=fft_device
        )
        m = self.modes
        w = torch.complex(wr, wi)
        out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m, :m], w)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(orig_device)


class FNOBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class MyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, channels=32, modes=8):
        super().__init__()
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
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
            nn.Dropout(0.3),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = x.mean(dim=[2, 3])
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
    print(f"Device: {device}")

    # 1. [참가자 작성 구간] 모델 인스턴스 생성
    model = MyModel(in_channels=1, num_classes=11, channels=32, modes=8)

    # 2. 가중치 로드
    try:
        checkpoint = torch.load(args.weight_path, map_location=device)

        if isinstance(checkpoint, dict) and "net" in checkpoint:
            model.load_state_dict(checkpoint["net"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 우리 저장 방식
            cfg = checkpoint.get("config", {})
            model = MyModel(
                in_channels=1,
                num_classes=cfg.get("num_classes", 11),
                channels=cfg.get("channels", 32),
                modes=cfg.get("modes", 8),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Successfully loaded weights from {args.weight_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device).eval()

    # 3. 전처리 (학습 시와 동일)
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # 4. 데이터 로더
    if not os.path.exists(args.input_dir):
        print(f"Directory not found: {args.input_dir}")
        return

    dataset = TestDataset(args.input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
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

            # TTA: 원본 + H-flip 앙상블
            out1 = torch.softmax(model(images), dim=1)
            out2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            outputs = (out1 + out2) / 2

            _, predicted = torch.max(outputs, 1)

            for i in range(len(filenames)):
                img_id = os.path.splitext(filenames[i])[0]
                pred_idx = predicted[i].item()
                res = {"index": img_id}
                for idx, name in enumerate(class_names):
                    res[name] = 1 if idx == pred_idx else 0
                results.append(res)

    df = pd.DataFrame(results)
    cols = ["index"] + class_names
    df = df[cols]
    df.to_csv(args.output_csv, index=False)
    print(f"Inference complete! Results saved to {args.output_csv}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()
