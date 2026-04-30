"""
train_proto_opt.py
Prototypical Networks integrated training pipeline with 8:2 Validation & Early Stopping
for the CAU medical-AI hackathon.

(PyTorch Hook을 사용하여 model_san1.py 수정 없이 작동하도록 최적화됨)
"""

from __future__ import annotations

import os
import copy
import argparse
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 모델 수정 없이 그대로 임포트
from model_san1 import build_model
from train_san2 import (
    load_domain,
    evaluate_all,
    stratified_split_indices,
    MedSubset,
    pretrain,
    l2sp_penalty,
)


# ══════════════════════════════════════════════════════════
# Prototypical Networks Logic
# ══════════════════════════════════════════════════════════
def compute_prototypes(
    features: torch.Tensor, labels: torch.Tensor, n_way: int
) -> torch.Tensor:
    """Compute prototypes as the mean of features for each class."""
    prototypes = []
    for cls in range(n_way):
        mask = labels == cls
        if mask.any():
            prototypes.append(features[mask].mean(0))
        else:
            prototypes.append(torch.zeros_like(features[0]))
    return torch.stack(prototypes)


def compute_distances(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distances between features and prototypes."""
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
    epochs: int = 40,
    n_way: int = 11,
    k_shot: int = 5,
    q_query: int = 10,
    lr: float = 1e-4,
) -> nn.Module:
    print(f"\n[Step 2] Prototypical Networks Meta-training (A+C)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    source_dataset = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])

    # Extract labels for episodic sampling
    all_labels = torch.cat([axial_dataset.labels, coronal_dataset.labels])
    class_indices = {cls: torch.where(all_labels == cls)[0] for cls in range(n_way)}

    # -------------------------------------------------------------
    # [핵심] model_san1.py 수정 없이 중간 feature를 빼오기 위한 Hook 설정
    # -------------------------------------------------------------
    extracted_features = {}

    def pre_classifier_hook(module, input):
        # classifier에 들어가기 직전의 텐서 (GAP 통과 후)
        extracted_features["feat"] = input[0]

    # 모델의 classifier 레이어 앞에 훅(hook)을 걸어둠
    hook_handle = model.classifier.register_forward_pre_hook(pre_classifier_hook)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_episodes = 20

        for _ in range(n_episodes):
            optimizer.zero_grad(set_to_none=True)

            support_imgs, support_lbls = [], []
            query_imgs, query_lbls = [], []

            for cls in range(n_way):
                indices = class_indices[cls]
                perm = torch.randperm(len(indices))

                s_idx = indices[perm[:k_shot]]
                q_idx = indices[perm[k_shot : k_shot + q_query]]

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

            # 훅(Hook)을 통해 원본 모델 호출만으로 feature 획득
            _ = model(s_imgs)
            s_features = extracted_features["feat"]  # Support Feature

            _ = model(q_imgs)
            q_features = extracted_features["feat"]  # Query Feature

            prototypes = compute_prototypes(s_features, s_lbls, n_way)
            distances = compute_distances(q_features, prototypes)

            # ProtoNet loss
            loss = criterion(-distances, q_lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Proto Epoch [{epoch+1:2d}/{epochs}] Loss: {epoch_loss/n_episodes:.4f}"
            )

    # 학습 완료 후 훅 제거
    hook_handle.remove()
    return model


# ══════════════════════════════════════════════════════════
# Fine-Tuning with Validation (Early Stopping)
# ══════════════════════════════════════════════════════════
def finetune_with_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_datasets: tuple,
    device: torch.device,
    epochs: int = 50,
    lr: float = 5e-5,
    lam: float = 0.005,  # L2-SP 규제 유지
):
    print(f"\n[Step 3] Fine-tuning on Sagittal (LR={lr})...")

    # ProtoNet 메타학습 직후의 가중치를 닻(Anchor)으로 삼아 L2-SP 적용
    theta_meta = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            ce_loss = criterion(logits, lbls)

            # L2-SP Penalty 추가 (망각 방지)
            reg = l2sp_penalty(model, theta_meta, device)
            loss = ce_loss + lam * reg

            loss.backward()
            optimizer.step()

            bs = len(lbls)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == lbls).sum().item()
            total += bs

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1) * 100.0

        if val_datasets is not None:
            scores = evaluate_all(model, *val_datasets, device=device, verbose=False)
            metric = scores["Final"]
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  FT Epoch [{epoch+1:2d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}% best_val={best_metric:.4f}"
                )
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  FT Epoch [{epoch+1:2d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}%"
                )

    if val_datasets is not None:
        model.load_state_dict(best_state)
        print(
            f"  Fine-tuning done. Best validation metric: {best_metric:.4f} at Epoch {best_epoch}"
        )
        return model, best_epoch
    else:
        return model, epochs


# ══════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./proto_fno_final.pth")
    parser.add_argument("--pretrain_epochs", type=int, default=150)
    parser.add_argument("--proto_epochs", type=int, default=40)
    parser.add_argument("--ft_epochs", type=int, default=80)
    parser.add_argument("--use_proto", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    axial = load_domain(
        os.path.join(args.train_dir, "axial"),
        os.path.join(args.train_dir, "label", "axial.csv"),
        augment=True,
    )
    coronal = load_domain(
        os.path.join(args.train_dir, "coronal"),
        os.path.join(args.train_dir, "label", "coronal.csv"),
        augment=True,
    )
    sagittal = load_domain(
        os.path.join(args.train_dir, "sagittal"),
        os.path.join(args.train_dir, "label", "sagittal.csv"),
        augment=True,
    )

    axial_eval = load_domain(
        os.path.join(args.train_dir, "axial"),
        os.path.join(args.train_dir, "label", "axial.csv"),
        augment=False,
    )
    coronal_eval = load_domain(
        os.path.join(args.train_dir, "coronal"),
        os.path.join(args.train_dir, "label", "coronal.csv"),
        augment=False,
    )
    sagittal_eval = load_domain(
        os.path.join(args.train_dir, "sagittal"),
        os.path.join(args.train_dir, "label", "sagittal.csv"),
        augment=False,
    )

    # 2. Build Model (using the 64-channel 12-mode optmized model)
    model = build_model(num_classes=11, channels=64, modes=12).to(device)

    # 3. Pre-training on A + C
    model = pretrain(
        model, axial, coronal, device, epochs=args.pretrain_epochs, lr=1e-3
    )

    # 4. ProtoNet Meta-training
    if args.use_proto:
        model = train_proto(model, axial, coronal, device, epochs=args.proto_epochs)

    proto_state = copy.deepcopy(model.state_dict())

    # 5. [핵심] 8:2 데이터 분할을 통한 Validation (최적 Epoch 찾기)
    print("\n[Internal Validation] Find best Epoch for Sagittal Fine-tuning...")
    train_idx, val_idx = stratified_split_indices(
        sagittal_eval.labels, val_per_class=10, seed=42
    )
    sagittal_train_split = MedSubset(sagittal_eval, train_idx, augment=True)
    sagittal_val_split = MedSubset(sagittal_eval, val_idx, augment=False)

    val_loader = DataLoader(
        sagittal_train_split, batch_size=32, shuffle=True, num_workers=4
    )

    _, best_epoch = finetune_with_eval(
        model,
        val_loader,
        (axial_eval, coronal_eval, sagittal_val_split),
        device=device,
        epochs=args.ft_epochs,
        lr=5e-5,
        lam=0.005,
    )

    # 6. [핵심] 최적 Epoch로 100% Sagittal 최종 학습
    print(f"\n[Final Training] Fine-tune on ALL Sagittal shots for {best_epoch} Epochs")
    model.load_state_dict(proto_state)  # ProtoNet 메타학습 직후로 롤백

    full_loader = DataLoader(sagittal, batch_size=32, shuffle=True, num_workers=4)
    model, _ = finetune_with_eval(
        model, full_loader, None, device=device, epochs=best_epoch, lr=5e-5, lam=0.005
    )

    # 7. Final Evaluation
    print("\n[Final Evaluation (F1 Score)]")
    evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"num_classes": 11, "channels": 64, "modes": 12},
        },
        args.save_path,
    )
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
