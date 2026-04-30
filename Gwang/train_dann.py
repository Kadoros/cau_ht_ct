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
from model_dann import build_model


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

    def __init__(self, image_dir, label_csv, domain_idx, augment=False):
        self.image_dir, self.domain_idx, self.augment = image_dir, domain_idx, augment
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
        return img, self.labels[idx], self.domain_idx

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


def train_dann(train_dir, device, epochs=100, lr=1e-3):
    model = build_model(num_classes=11, num_domains=3).to(device)
    axial = MedDataset(
        os.path.join(train_dir, "axial"),
        os.path.join(train_dir, "label", "axial.csv"),
        0,
        augment=True,
    )
    coronal = MedDataset(
        os.path.join(train_dir, "coronal"),
        os.path.join(train_dir, "label", "coronal.csv"),
        1,
        augment=True,
    )
    sagittal = MedDataset(
        os.path.join(train_dir, "sagittal"),
        os.path.join(train_dir, "label", "sagittal.csv"),
        2,
        augment=True,
    )
    src_loader = DataLoader(
        ConcatDataset([axial, coronal]), batch_size=64, shuffle=True, drop_last=True
    )
    tgt_loader = DataLoader(sagittal, batch_size=32, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_cls, criterion_dom = (
        nn.CrossEntropyLoss(label_smoothing=0.1),
        nn.CrossEntropyLoss(),
    )
    for epoch in range(epochs):
        model.train()
        alpha = 2.0 / (1.0 + np.exp(-10 * (epoch / epochs))) - 1
        for (s_img, s_lbl, s_dom), (t_img, _, t_dom) in zip(src_loader, tgt_loader):
            optimizer.zero_grad()
            s_out_cls, s_out_dom = model(s_img.to(device), alpha)
            loss_s_cls = criterion_cls(s_out_cls, s_lbl.to(device))
            loss_s_dom = criterion_dom(s_out_dom, s_dom.to(device))
            _, t_out_dom = model(t_img.to(device), alpha)
            loss_t_dom = criterion_dom(t_out_dom, t_dom.to(device))
            (loss_s_cls + loss_s_dom + loss_t_dom).backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} Loss Cls: {loss_s_cls.item():.4f}")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    args = parser.parse_args()
    train_dann(
        args.train_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
