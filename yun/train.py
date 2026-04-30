"""
train.py
Fixed full pipeline with Focal Loss, Cutout, and 8:2 Early Stopping.
"""

import os, math, copy, argparse, random
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from model import build_model


# --- [Focal Loss & Cutout] ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class Cutout(object):
    def __init__(self, length=6):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[-2:]
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(
            y + self.length // 2, 0, h
        )
        x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(
            x + self.length // 2, 0, w
        )
        img[..., y1:y2, x1:x2] = -1.0
        return img


# --- 데이터 로딩 및 Augmentation ---
class MedDataset(Dataset):
    CLASS_NAMES = [
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

    def __init__(self, image_dir: str, label_csv: str, augment: bool = False):
        self.augment = augment
        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        labels = raw[self.CLASS_NAMES].astype(int).values.argmax(axis=1)
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(raw.index.tolist(), labels)}

        all_pngs = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        )
        self.image_paths, self.labels_list = [], []
        for fn in all_pngs:
            key = os.path.splitext(fn)[0]
            if key in fname_to_label:
                self.image_paths.append(os.path.join(image_dir, fn))
                self.labels_list.append(fname_to_label[key])

        imgs = []
        for path in self.image_paths:
            img = read_image(path, mode=ImageReadMode.GRAY).float()
            img = F.interpolate(img.unsqueeze(0), (28, 28), mode="bilinear").squeeze(0)
            img = (img / (img.max() + 1e-6) - 0.5) / 0.5
            imgs.append(img)

        self.imgs = torch.stack(imgs).contiguous()
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)
        self.cutout = Cutout()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img, lbl = self.imgs[idx].clone(), self.labels[idx]
        if self.augment:
            if torch.rand(()) < 0.5:
                img = _affine_tensor(img, angle_deg=(torch.rand(()).item() - 0.5) * 20)
            if torch.rand(()) < 0.5:
                img = self.cutout(img)
        return img.clamp(-1.0, 1.0), lbl


class MedSubset(Dataset):
    """데이터셋을 8:2로 나눌 때 사용하는 서브셋 래퍼"""

    def __init__(self, base: MedDataset, indices: Sequence[int], augment: bool = False):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.augment = augment
        self.labels = base.labels[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx].item())
        img = self.base.imgs[base_idx].clone()
        lbl = self.base.labels[base_idx]
        if self.augment:
            if torch.rand(()) < 0.5:
                img = _affine_tensor(img, angle_deg=(torch.rand(()).item() - 0.5) * 20)
            if torch.rand(()) < 0.5:
                img = self.base.cutout(img)
        return img.clamp(-1.0, 1.0), lbl


def _affine_tensor(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle = math.radians(angle_deg)
    theta = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0]],
        dtype=img.dtype,
        device=img.device,
    )
    grid = F.affine_grid(
        theta.unsqueeze(0), img.unsqueeze(0).shape, align_corners=False
    )
    return F.grid_sample(
        img.unsqueeze(0),
        grid,
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    ).squeeze(0)


def stratified_split_indices(
    labels: torch.Tensor, val_per_class: int = 10, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """클래스별로 균등하게 Train/Validation 셋 분할"""
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_idx, val_idx = [], []
    for cls in range(11):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        if len(idx) == 0:
            continue
        n_val = min(val_per_class, max(1, len(idx) // 5))
        if len(idx) - n_val < 1:
            n_val = max(0, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# --- 평가 유틸리티 ---
def evaluate(model, dataset, device):
    from sklearn.metrics import f1_score

    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())
    return float(f1_score(trues, preds, average="macro", zero_division=0))


# --- 20% 검증 & 학습 로직 ---
def train_model_with_eval(
    model, train_ds, val_ds, device, epochs, lr, is_pretrain=False
):
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss()

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = epochs

    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

        # 검증셋이 있을 경우 최고 기록 저장
        if val_ds is not None:
            f1 = evaluate(model, val_ds, device)
            if f1 > best_metric:
                best_metric = f1
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f" Epoch [{epoch+1:3d}/{epochs}] S-Val F1: {f1:.4f} | Best: {best_metric:.4f}"
                )
        else:
            if (epoch + 1) % 10 == 0:
                print(
                    f" Epoch [{epoch+1:3d}/{epochs}] Training Loss: {loss.item():.4f}"
                )

    if val_ds is not None:
        model.load_state_dict(best_state)
        print(f" 👉 Best Validation F1: {best_metric:.4f} at Epoch {best_epoch}")
        return model, best_epoch
    else:
        return model, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./model.pth")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = args.train_dir
    print("Loading datasets...")
    A_train = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), augment=True
    )
    C_train = MedDataset(
        os.path.join(t, "coronal"), os.path.join(t, "label/coronal.csv"), augment=True
    )
    S_full = MedDataset(
        os.path.join(t, "sagittal"), os.path.join(t, "label/sagittal.csv"), augment=True
    )
    S_eval = MedDataset(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label/sagittal.csv"),
        augment=False,
    )

    model = build_model().to(device)

    # 1. 소스 도메인 학습 (Axial + Coronal)
    print("\n[Step 1] Pre-training on Axial + Coronal...")
    # 사전학습 시에는 Early Stopping 없이 전체 학습 후 Sagittal 평가 확인
    model, _ = train_model_with_eval(
        model,
        torch.utils.data.ConcatDataset([A_train, C_train]),
        None,
        device,
        100,
        1e-3,
        is_pretrain=True,
    )
    pretrained_state = copy.deepcopy(model.state_dict())

    # 2. 내부 20% 검증을 위한 데이터 분리 (모의고사)
    print("\n[Step 2] Internal Validation to find best fine-tuning Epoch...")
    train_idx, val_idx = stratified_split_indices(
        S_eval.labels, val_per_class=10, seed=42
    )
    S_train_split = MedSubset(S_eval, train_idx, augment=True)
    S_val_split = MedSubset(S_eval, val_idx, augment=False)

    # 3. 20% 평가를 통해 최적의 에포크(Early Stopping) 찾기
    model, best_epoch = train_model_with_eval(
        model, S_train_split, S_val_split, device, 50, 5e-5
    )

    # 4. 100% Sagittal 데이터로 최종 병기 제작
    print(
        f"\n[Step 3] Final Fine-Tuning on ALL Sagittal Data for {best_epoch} Epochs..."
    )
    model.load_state_dict(pretrained_state)  # 파인튜닝 전 백지 상태로 롤백
    train_model_with_eval(
        model, S_full, None, device, best_epoch, 5e-5
    )  # 100% 학습 시엔 평가 생략

    # 5. 모델 저장
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"num_classes": 11, "channels": 64, "modes": 12},
        },
        args.save_path,
    )
    print(f"\n✅ Training Complete. Model Saved to {args.save_path}")


if __name__ == "__main__":
    main()
