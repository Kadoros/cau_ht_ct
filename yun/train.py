"""
train_spatial_optimized.py
Pure Spatial & Feature Extractor Optimized Pipeline.
(FOMAML, L2-SP removed. Cutout & Focal Loss integrated.)
"""

import os
import math
import copy
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from model import build_model


# ─────────────────────────────────────────────────────────
# Cutout: 공간 증강 기법 (CBAM 어텐션 강화용)
# ─────────────────────────────────────────────────────────
class Cutout(object):
    def __init__(self, length=8):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        img[..., y1:y2, x1:x2] = -1.0  # 정규화된 이미지의 배경(0)에 맞춰 구멍 뚫기
        return img


# ─────────────────────────────────────────────────────────
# 데이터셋 및 로더
# ─────────────────────────────────────────────────────────
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
            if img.shape[-2:] != (28, 28):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            img = (img / (img.max() + 1e-6) - 0.5) / 0.5
            imgs.append(img)

        self.imgs = torch.stack(imgs).contiguous()
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)

        # 공간 기하학 증강 + Cutout
        self.cutout = Cutout(length=6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self._augment(img)
        return img, lbl

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            angle = (torch.rand(()).item() - 0.5) * 20.0
            tx, ty = int(torch.randint(-1, 2, ()).item()), int(
                torch.randint(-1, 2, ()).item()
            )
            img = _affine_tensor(img, angle_deg=angle, translate_px=(tx, ty), scale=1.0)

        # 50% 확률로 Cutout 적용하여 CBAM 훈련 유도
        if torch.rand(()) < 0.5:
            img = self.cutout(img)

        return img.clamp(-1.0, 1.0)


def _affine_tensor(
    img: torch.Tensor,
    angle_deg: float,
    translate_px: Tuple[int, int] = (0, 0),
    scale: float = 1.0,
) -> torch.Tensor:
    _, h, w = img.shape
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle) / scale, math.sin(angle) / scale
    tx, ty = 2.0 * translate_px[0] / max(w, 1), 2.0 * translate_px[1] / max(h, 1)
    theta = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=img.dtype, device=img.device
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


# ─────────────────────────────────────────────────────────
# [네 파트 2] Focal Loss: 작고 어려운 장기에 집중
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────
# 학습 로직
# ─────────────────────────────────────────────────────────
def pretrain(model, A_ds, C_ds, device, epochs, lr):
    print("\n[Phase 1] Robust Pre-training on Source Domains (Focal Loss + Cutout)...")
    loader = DataLoader(
        torch.utils.data.ConcatDataset([A_ds, C_ds]),
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs
    )

    # 일반 CrossEntropy 대신 Focal Loss 사용
    criterion = FocalLoss(gamma=2.0)

    model.to(device)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += len(lbls)

        avg_loss = total_loss / total
        if avg_loss < best_loss:
            best_loss, best_state = avg_loss, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}] Focal Loss: {avg_loss:.4f} | Acc: {correct/total*100:.1f}%"
            )

    model.load_state_dict(best_state)
    return model


def finetune(model, S_train, S_val, device, epochs, lr):
    print("\n[Phase 2] Direct Fine-Tuning on Target Domain...")
    loader = DataLoader(S_train, batch_size=32, shuffle=True, num_workers=4)
    # 인코더가 깨지지 않게 학습률을 낮춰서 미세조정
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(gamma=2.0)

    best_f1, best_state = 0.0, copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            f1 = evaluate(model, S_val, device)
            if f1 > best_f1:
                best_f1, best_state = f1, copy.deepcopy(model.state_dict())
            print(
                f"  FT Epoch [{epoch+1:3d}/{epochs}] S-Val F1: {f1:.4f} (Best: {best_f1:.4f})"
            )

    model.load_state_dict(best_state)
    return model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./model.pth")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = args.train_dir

    A_train = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), augment=True
    )
    C_train = MedDataset(
        os.path.join(t, "coronal"), os.path.join(t, "label/coronal.csv"), augment=True
    )
    S_train = MedDataset(
        os.path.join(t, "sagittal"), os.path.join(t, "label/sagittal.csv"), augment=True
    )
    A_eval = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), augment=False
    )
    C_eval = MedDataset(
        os.path.join(t, "coronal"), os.path.join(t, "label/coronal.csv"), augment=False
    )
    S_eval = MedDataset(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label/sagittal.csv"),
        augment=False,
    )

    model = build_model().to(device)

    # 1. Pretrain (Cutout + Focal Loss)
    model = pretrain(model, A_train, C_train, device, epochs=150, lr=1e-3)

    # 2. Finetune on Target
    model = finetune(model, S_train, S_eval, device, epochs=80, lr=5e-5)

    # 3. Final Eval
    f1_a, f1_c, f1_s = (
        evaluate(model, A_eval, device),
        evaluate(model, C_eval, device),
        evaluate(model, S_eval, device),
    )
    final = 0.7 * f1_s + 0.3 * (f1_a + f1_c) / 2.0
    print(f"\n🏆 FINAL SCORE: {final:.4f} (A:{f1_a:.4f}, C:{f1_c:.4f}, S:{f1_s:.4f})")

    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
