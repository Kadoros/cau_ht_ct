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
from model_dann import DANN_HybridFNO


# ══════════════════════════════════════════════════════════
# 0. Dataset with Explicit Rotation Augmentation
# ══════════════════════════════════════════════════════════
class DANNMedDataset(Dataset):
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
        self.image_dir = image_dir
        self.domain_idx = domain_idx
        self.augment = augment

        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        one_hot = raw[self.CLASS_NAMES].astype(int).values
        labels = one_hot.argmax(axis=1)
        filenames = raw.index.tolist()
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(filenames, labels)}

        all_pngs = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        )
        self.imgs = []
        self.labels = []
        for fn in all_pngs:
            key = os.path.splitext(fn)[0]
            if key in fname_to_label:
                path = os.path.join(image_dir, fn)
                img = read_image(path, mode=ImageReadMode.GRAY).float()
                if img.shape[-2:] != (28, 28):
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(28, 28),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                img_min, img_max = img.min(), img.max()
                img = (img - img_min) / (img_max - img_min + 1e-8)
                img = (img - 0.5) / 0.5
                self.imgs.append(img)
                self.labels.append(fname_to_label[key])

        self.imgs = torch.stack(self.imgs).contiguous()
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self._apply_rotation_augment(img)
        return img, lbl, self.domain_idx

    def _apply_rotation_augment(self, img):
        # Explicit Rotation Augmentation
        if torch.rand(()) < 0.8:  # High probability for rotation
            angle_deg = (torch.rand(()).item() - 0.5) * 60.0  # -30 to 30 degrees
            img = self._rotate_tensor(img, angle_deg)

        # Other mild augmentations
        if torch.rand(()) < 0.5:
            img = (
                img * (0.8 + torch.rand(()).item() * 0.4)
                + (torch.rand(()).item() - 0.5) * 0.1
            )
        return img.clamp(-1.0, 1.0)

    def _rotate_tensor(self, img, angle_deg):
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=img.dtype
        ).unsqueeze(0)
        grid = F.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
        return F.grid_sample(
            img.unsqueeze(0),
            grid,
            align_corners=False,
            mode="bilinear",
            padding_mode="border",
        ).squeeze(0)


# ══════════════════════════════════════════════════════════
# 1. DANN Training Loop
# ══════════════════════════════════════════════════════════
def train_dann(model, source_loader, target_loader, device, epochs=100, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.CrossEntropyLoss()

    best_model_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        len_dataloader = min(len(source_loader), len(target_loader))
        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)

        total_class_loss = 0
        total_domain_loss = 0

        # Dynamic Alpha for GRL
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        for i in range(len_dataloader):
            # 1. Train with Source Domain
            s_img, s_cls, s_dom = next(data_source_iter)
            s_img, s_cls, s_dom = s_img.to(device), s_cls.to(device), s_dom.to(device)

            optimizer.zero_grad()
            class_output, domain_output = model(s_img, alpha=alpha)
            err_s_label = class_criterion(class_output, s_cls)
            err_s_domain = domain_criterion(domain_output, s_dom)

            # 2. Train with Target Domain
            t_img, t_cls, t_dom = next(data_target_iter)
            t_img, t_dom = t_img.to(device), t_dom.to(device)
            _, domain_output = model(t_img, alpha=alpha)
            err_t_domain = domain_criterion(domain_output, t_dom)

            loss = err_s_label + err_s_domain + err_t_domain
            loss.backward()
            optimizer.step()

            total_class_loss += err_s_label.item()
            total_domain_loss += err_s_domain.item() + err_t_domain.item()

        avg_class_loss = total_class_loss / len_dataloader
        if avg_class_loss < best_loss:
            best_loss = avg_class_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] Class Loss: {avg_class_loss:.4f}, Domain Loss: {total_domain_loss/len_dataloader:.4f}, Alpha: {alpha:.4f}"
            )

    model.load_state_dict(best_model_state)
    return model


# ══════════════════════════════════════════════════════════
# 2. Execution
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Domains: 0: Axial, 1: Coronal, 2: Sagittal
    print("Loading datasets for DANN...")
    axial = DANNMedDataset(
        os.path.join(args.train_dir, "axial"),
        os.path.join(args.train_dir, "label", "axial.csv"),
        domain_idx=0,
        augment=True,
    )
    coronal = DANNMedDataset(
        os.path.join(args.train_dir, "coronal"),
        os.path.join(args.train_dir, "label", "coronal.csv"),
        domain_idx=1,
        augment=True,
    )
    sagittal = DANNMedDataset(
        os.path.join(args.train_dir, "sagittal"),
        os.path.join(args.train_dir, "label", "sagittal.csv"),
        domain_idx=2,
        augment=True,
    )

    source_dataset = ConcatDataset([axial, coronal])
    source_loader = DataLoader(
        source_dataset, batch_size=64, shuffle=True, drop_last=True
    )
    target_loader = DataLoader(sagittal, batch_size=32, shuffle=True, drop_last=True)

    model = DANN_HybridFNO(num_classes=11, num_domains=3)

    print("Starting DANN Training with Rotation Augmentation...")
    model = train_dann(
        model, source_loader, target_loader, device, epochs=args.epochs, lr=args.lr
    )

    torch.save(model.state_dict(), "dann_fno_final.pth")
    print("Training Complete. Model saved as dann_fno_final.pth")
