"""
train_proto.py
Prototypical Networks integrated training pipeline for the CAU medical-AI hackathon.
"""

from __future__ import annotations

import os
import math
import copy
import argparse
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from model_proto import build_model

# ══════════════════════════════════════════════════════════
# Prototypical Networks Logic
# ══════════════════════════════════════════════════════════

def compute_prototypes(features: torch.Tensor, labels: torch.Tensor, n_way: int) -> torch.Tensor:
    """Compute prototypes as the mean of features for each class."""
    prototypes = []
    for cls in range(n_way):
        mask = (labels == cls)
        if mask.any():
            prototypes.append(features[mask].mean(0))
        else:
            # Fallback for empty classes (should not happen in balanced tasks)
            prototypes.append(torch.zeros_like(features[0]))
    return torch.stack(prototypes)

def compute_distances(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distances between features and prototypes."""
    # (n_query, feature_dim) and (n_way, feature_dim)
    # distance = ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    n_query = features.size(0)
    n_way = prototypes.size(0)
    
    sq_features = torch.sum(features**2, dim=1).view(n_query, 1)
    sq_prototypes = torch.sum(prototypes**2, dim=1).view(1, n_way)
    dot_product = torch.matmul(features, prototypes.t())
    
    distances = sq_features + sq_prototypes - 2 * dot_product
    return distances

def train_proto(
    model: nn.Module,
    axial_dataset: Dataset,
    coronal_dataset: Dataset,
    device: torch.device,
    epochs: int = 50,
    n_way: int = 11,
    k_shot: int = 5,
    q_query: int = 10,
    lr: float = 1e-4,
) -> nn.Module:
    print(f"\n[Step 2] Prototypical Networks Meta-training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Combined source dataset for episodic sampling
    source_dataset = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])
    # To sample episodes easily, we need a way to group indices by class
    all_labels = []
    # ConcatDataset doesn't have .labels, so we might need to extract them
    # For efficiency, we assume MedDataset has .labels
    if isinstance(axial_dataset, MedDataset) and isinstance(coronal_dataset, MedDataset):
        all_labels = torch.cat([axial_dataset.labels, coronal_dataset.labels])
    else:
        # Fallback (slower)
        all_labels = torch.tensor([lbl for _, lbl in source_dataset])
        
    class_indices = {cls: torch.where(all_labels == cls)[0] for cls in range(n_way)}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_episodes = 20 # Number of episodes per epoch
        
        for _ in range(n_episodes):
            optimizer.zero_grad()
            
            support_imgs, support_lbls = [], []
            query_imgs, query_lbls = [], []
            
            for cls in range(n_way):
                indices = class_indices[cls]
                perm = torch.randperm(len(indices))
                
                s_idx = indices[perm[:k_shot]]
                q_idx = indices[perm[k_shot:k_shot+q_query]]
                
                for idx in s_idx:
                    img, _ = source_dataset[int(idx)]
                    support_imgs.append(img)
                    support_lbls.append(cls)
                for idx in q_idx:
                    img, _ = source_dataset[int(idx)]
                    query_imgs.append(img)
                    query_lbls.append(cls)
            
            s_imgs = torch.stack(support_imgs).to(device)
            s_lbls = torch.tensor(support_lbls, device=device)
            q_imgs = torch.stack(query_imgs).to(device)
            q_lbls = torch.tensor(query_lbls, device=device)
            
            # Extract features
            _, s_features = model(s_imgs, return_features=True)
            _, q_features = model(q_imgs, return_features=True)
            
            # ProtoNet logic
            prototypes = compute_prototypes(s_features, s_lbls, n_way)
            distances = compute_distances(q_features, prototypes)
            
            # Loss: minimize distance to correct prototype
            # distances are like negative logits
            loss = criterion(-distances, q_lbls)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Proto Epoch [{epoch+1:3d}/{epochs}] Loss: {epoch_loss/n_episodes:.4f}")
            
    return model

# ══════════════════════════════════════════════════════════
# Original Dataset & Augmentation (Minimal version)
# ══════════════════════════════════════════════════════════

class MedDataset(Dataset):
    CLASS_NAMES = ["bladder", "femur-left", "femur-right", "heart", "kidney-left", "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
    def __init__(self, image_dir, label_csv, augment=False):
        self.image_dir, self.augment = image_dir, augment
        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        labels = raw[self.CLASS_NAMES].astype(int).values.argmax(axis=1)
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(raw.index, labels)}
        self.image_paths, self.labels_list = [], []
        for fn in sorted(os.listdir(image_dir)):
            if (key := os.path.splitext(fn)[0]) in fname_to_label:
                self.image_paths.append(os.path.join(image_dir, fn))
                self.labels_list.append(fname_to_label[key])
        self.imgs = torch.stack([self._load(p) for p in self.image_paths])
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)
    def _load(self, path):
        img = read_image(path, mode=ImageReadMode.GRAY).float()
        img = F.interpolate(img.unsqueeze(0), size=(28,28)).squeeze(0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return (img - 0.5) / 0.5
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.imgs[idx].clone()
        if self.augment:
            if torch.rand(()) < 0.5: img = img + torch.randn_like(img)*0.02
        return img, self.labels[idx]

def pretrain(model, axial_dataset, coronal_dataset, device, epochs=100, lr=1e-3):
    print("\n[Step 1] Pre-training on Axial + Coronal...")
    loader = DataLoader(torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset]), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_proto", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("Loading datasets...")
    # Using axial and coronal for source, sagittal for target
    ds_a = MedDataset(os.path.join(args.train_dir, "axial"), os.path.join(args.train_dir, "label", "axial.csv"), augment=True)
    ds_c = MedDataset(os.path.join(args.train_dir, "coronal"), os.path.join(args.train_dir, "label", "coronal.csv"), augment=True)
    # Sagittal will be used for final fine-tuning if needed, but ProtoNet can also adapt to it
    ds_s = MedDataset(os.path.join(args.train_dir, "sagittal"), os.path.join(args.train_dir, "label", "sagittal.csv"), augment=True)

    model = build_model().to(device)
    
    # Standard pre-training
    model = pretrain(model, ds_a, ds_c, device)
    
    # ProtoNet meta-training
    if args.use_proto:
        model = train_proto(model, ds_a, ds_c, device)
    
    # Final step: standard fine-tuning on Sagittal shots to ensure high F1_S
    print("\n[Step 3] Fine-tuning on Sagittal...")
    s_loader = DataLoader(ds_s, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(50):
        model.train()
        for imgs, lbls in s_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            optimizer.step()
            
    torch.save({"model_state_dict": model.state_dict(), "config": {"num_classes": 11, "channels": 64, "modes": 12}}, args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
