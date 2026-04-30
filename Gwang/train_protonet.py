import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from model_protonet import build_model, euclidean_dist


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
        self.class_indices = [
            torch.where(self.labels == i)[0] for i in range(len(self.CLASS_NAMES))
        ]

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


def train_protonet(train_dir, device, episodes=2000, n_way=11, k_shot=5, q_query=5):
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    axial = ProtoDataset(
        os.path.join(train_dir, "axial"),
        os.path.join(train_dir, "label", "axial.csv"),
        augment=True,
    )
    coronal = ProtoDataset(
        os.path.join(train_dir, "coronal"),
        os.path.join(train_dir, "label", "coronal.csv"),
        augment=True,
    )
    for ep in range(episodes):
        model.train()
        optimizer.zero_grad()
        ds = axial if random.random() < 0.5 else coronal
        classes = torch.randperm(len(ds.CLASS_NAMES))[:n_way]
        s_imgs, q_imgs, q_lbls = [], [], []
        for i, cls in enumerate(classes):
            idx = ds.class_indices[cls]
            perm = torch.randperm(len(idx))
            for j in perm[:k_shot]:
                s_imgs.append(ds[idx[j]][0])
            for j in perm[k_shot : k_shot + q_query]:
                q_imgs.append(ds[idx[j]][0])
                q_lbls.append(i)
        embeds = model(torch.cat([torch.stack(s_imgs), torch.stack(q_imgs)]).to(device))
        s_embeds = embeds[: n_way * k_shot].view(n_way, k_shot, -1).mean(1)
        q_embeds = embeds[n_way * k_shot :]
        dists = euclidean_dist(q_embeds, s_embeds)
        loss = F.cross_entropy(-dists, torch.tensor(q_lbls).to(device))
        loss.backward()
        optimizer.step()
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1} Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    args = parser.parse_args()
    train_protonet(
        args.train_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
