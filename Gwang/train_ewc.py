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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.io import read_image, ImageReadMode
from model_ewc import build_model


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

    def __init__(self, image_dir, label_csv, augment=False):
        self.image_dir, self.augment = image_dir, augment
        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        labels = raw[self.CLASS_NAMES].astype(int).values.argmax(axis=1)
        filenames = raw.index.tolist()
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(filenames, labels)}
        self.imgs, self.labels = [], []
        for fn in sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        ):
            key = os.path.splitext(fn)[0]
            if key in fname_to_label:
                img = read_image(
                    os.path.join(image_dir, fn), mode=ImageReadMode.GRAY
                ).float()
                if img.shape[-2:] != (28, 28):
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(28, 28),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                self.imgs.append((img - 0.5) / 0.5)
                self.labels.append(fname_to_label[key])
        self.imgs = torch.stack(self.imgs).contiguous()
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].clone()
        if self.augment:
            img = self._augment(img)
        return img, self.labels[idx]

    def _augment(self, img):
        if torch.rand(()) < 0.6:
            angle = (torch.rand(()).item() - 0.5) * 60.0
            theta = torch.tensor(
                [
                    [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                    [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                ],
                dtype=img.dtype,
            ).unsqueeze(0)
            grid = F.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
            img = F.grid_sample(
                img.unsqueeze(0),
                grid,
                align_corners=False,
                mode="bilinear",
                padding_mode="border",
            ).squeeze(0)
        return img.clamp(-1.0, 1.0)


class EWC:
    def __init__(self, model, loader, device):
        self.model, self.device = model, device
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = self._diag_fisher(loader)

    def _diag_fisher(self, loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        for img, lbl in loader:
            self.model.zero_grad()
            F.cross_entropy(
                self.model(img.to(self.device)), lbl.to(self.device)
            ).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data**2 / len(loader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


def train_ewc(train_dir, device, pre_epochs=100, ft_epochs=100, lam=1000):
    model = build_model().to(device)
    axial = MedDataset(
        os.path.join(train_dir, "axial"),
        os.path.join(train_dir, "label", "axial.csv"),
        augment=True,
    )
    coronal = MedDataset(
        os.path.join(train_dir, "coronal"),
        os.path.join(train_dir, "label", "coronal.csv"),
        augment=True,
    )
    sagittal = MedDataset(
        os.path.join(train_dir, "sagittal"),
        os.path.join(train_dir, "label", "sagittal.csv"),
        augment=True,
    )
    loader = DataLoader(ConcatDataset([axial, coronal]), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(pre_epochs):
        model.train()
        for img, lbl in loader:
            opt.zero_grad()
            F.cross_entropy(model(img.to(device)), lbl.to(device)).backward()
            opt.step()
    ewc = EWC(model, DataLoader(ConcatDataset([axial, coronal]), batch_size=32), device)
    loader = DataLoader(sagittal, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(ft_epochs):
        model.train()
        for img, lbl in loader:
            opt.zero_grad()
            (
                F.cross_entropy(model(img.to(device)), lbl.to(device))
                + lam * ewc.penalty(model)
            ).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"FT Epoch {epoch+1} done.")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    args = parser.parse_args()
    train_ewc(
        args.train_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
