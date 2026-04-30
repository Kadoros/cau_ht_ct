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
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from model_protonet import ProtoNet_FNO, euclidean_dist


# ══════════════════════════════════════════════════════════
# 0. Dataset with Rotation Augmentation
# ══════════════════════════════════════════════════════════
class ProtoDataset(Dataset):
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
        self.image_dir = image_dir
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

        # Group by class for task sampling
        self.class_indices = [
            torch.where(self.labels == i)[0] for i in range(len(self.CLASS_NAMES))
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self._apply_rotation_augment(img)
        return img, lbl

    def _apply_rotation_augment(self, img):
        if torch.rand(()) < 0.8:
            angle_deg = (torch.rand(()).item() - 0.5) * 60.0
            img = self._rotate_tensor(img, angle_deg)
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
# 1. ProtoNet Training Step (Episodic Training)
# ══════════════════════════════════════════════════════════
def train_protonet_epoch(model, dataset, optimizer, n_way, k_shot, q_query, device):
    model.train()

    # Sample Classes
    num_classes = len(dataset.CLASS_NAMES)
    classes = torch.randperm(num_classes)[:n_way]

    support_imgs = []
    query_imgs = []

    for cls in classes:
        indices = dataset.class_indices[cls]
        perm = torch.randperm(len(indices))

        # Select Support & Query
        s_idx = indices[perm[:k_shot]]
        q_idx = indices[perm[k_shot : k_shot + q_query]]

        for idx in s_idx:
            img, _ = dataset[idx]
            support_imgs.append(img)
        for idx in q_idx:
            img, _ = dataset[idx]
            query_imgs.append(img)

    support_imgs = torch.stack(support_imgs).to(device)
    query_imgs = torch.stack(query_imgs).to(device)

    optimizer.zero_grad()

    # Compute embeddings
    input_imgs = torch.cat([support_imgs, query_imgs], dim=0)
    embeddings = model(input_imgs)

    support_embeddings = embeddings[: n_way * k_shot].view(n_way, k_shot, -1)
    query_embeddings = embeddings[n_way * k_shot :]

    # Compute Prototypes (Mean of support embeddings)
    prototypes = support_embeddings.mean(1)  # [n_way, D]

    # Compute Distances & Loss
    dists = euclidean_dist(query_embeddings, prototypes)  # [n_way * q_query, n_way]

    query_labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

    loss = F.cross_entropy(-dists, query_labels)
    loss.backward()
    optimizer.step()

    acc = ((-dists).argmax(1) == query_labels).float().mean().item()
    return loss.item(), acc


# ══════════════════════════════════════════════════════════
# 2. Execution
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=11)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets for ProtoNet...")
    axial = ProtoDataset(
        os.path.join(args.train_dir, "axial"),
        os.path.join(args.train_dir, "label", "axial.csv"),
        augment=True,
    )

    model = ProtoNet_FNO(channels=64, modes=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting ProtoNet Training ({args.episodes} episodes)...")
    for ep in range(args.episodes):
        loss, acc = train_protonet_epoch(
            model, axial, optimizer, args.n_way, args.k_shot, args.q_query, device
        )

        if (ep + 1) % 100 == 0:
            print(f"Episode [{ep+1}/{args.episodes}] Loss: {loss:.4f}, Acc: {acc:.4f}")

    torch.save(model.state_dict(), "protonet_fno_final.pth")
    print("Training Complete. Model saved as protonet_fno_final.pth")
