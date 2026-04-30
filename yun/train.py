"""
train.py
Fixed full pipeline with Focal Loss & Cutout.
"""

import os, math, copy, argparse, random
from typing import Dict, List, Tuple  # [Fixed] Tuple is now defined!

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from model import build_model


# --- [네 파트] Focal Loss & Cutout ---
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


# --- 데이터 로딩 로직 ---
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


# --- 학습 루프 ---
def train_model(model, train_ds, val_ds, device, epochs, lr):
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            f1 = evaluate(model, val_ds, device)
            print(f" Epoch [{epoch+1:3d}/{epochs}] S-Val F1: {f1:.4f}")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = args.train_dir
    A_train = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), True
    )
    C_train = MedDataset(
        os.path.join(t, "coronal"), os.path.join(t, "label/coronal.csv"), True
    )
    S_train = MedDataset(
        os.path.join(t, "sagittal"), os.path.join(t, "label/sagittal.csv"), True
    )
    S_eval = MedDataset(
        os.path.join(t, "sagittal"), os.path.join(t, "label/sagittal.csv"), False
    )

    model = build_model().to(device)
    # 1. 소스 도메인 학습 (Axial + Coronal)
    train_model(
        model,
        torch.utils.data.ConcatDataset([A_train, C_train]),
        S_eval,
        device,
        100,
        1e-3,
    )
    # 2. 타겟 도메인 파인튜닝 (Sagittal)
    train_model(model, S_train, S_eval, device, 50, 5e-5)

    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Training Complete. Model Saved to {args.save_path}")


if __name__ == "__main__":
    main()
