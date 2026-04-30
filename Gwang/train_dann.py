"""
train_dann_opt.py
DANN integrated training pipeline with 8:2 Validation & Early Stopping
for the CAU medical-AI hackathon.
"""

import os
import math
import copy
import argparse
import random
from typing import Tuple, Sequence, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 앞서 사용하던 모델 아키텍처와 DANN 컴포넌트 임포트
from model_dann import build_model, GradientReversalLayer, DomainDiscriminator
from train_san2 import (
    MedDataset,
    MedSubset,
    _affine_tensor,
    load_domain,
    stratified_split_indices,
    evaluate_all,
    pretrain,
)


# ══════════════════════════════════════════════════════════
# DANN Training Logic with Internal Validation
# ══════════════════════════════════════════════════════════
def train_dann_with_eval(
    model: nn.Module,
    discriminator: nn.Module,
    a_loader: DataLoader,
    c_loader: DataLoader,
    s_loader: DataLoader,
    val_datasets: tuple,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    dann_weight: float = 0.1,
):
    print(f"\n[DANN Training] weight={dann_weight}, lr={lr}")

    grl = GradientReversalLayer()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1

    for epoch in range(epochs):
        model.train()
        discriminator.train()

        # S 도메인 데이터가 적으므로, S 데이터 길이에 맞춰 1 Epoch를 정의 (과적합 방지)
        num_iters = len(s_loader)
        loaders = [iter(a_loader), iter(c_loader), iter(s_loader)]

        total_cls_loss, total_dom_loss = 0.0, 0.0

        for i in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            p = float(epoch * num_iters + i) / (epochs * num_iters)
            grl.alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            for d_idx, loader_iter in enumerate(loaders):
                try:
                    imgs, lbls = next(loader_iter)
                except StopIteration:
                    # 데이터가 모자라면 로더를 다시 초기화
                    if d_idx == 0:
                        loaders[0] = iter(a_loader)
                    elif d_idx == 1:
                        loaders[1] = iter(c_loader)
                    else:
                        loaders[2] = iter(s_loader)
                    imgs, lbls = next(loaders[d_idx])

                imgs, lbls = imgs.to(device), lbls.to(device)

                # 1. Classification (Source 도메인(A,C)과 Target 도메인(S) 모두 사용)
                # 단, 원본 논문에서는 S 도메인 라벨을 안 쓰지만, 해커톤은 퓨샷 파인튜닝이므로 S 라벨도 사용
                logits, features = model(imgs, return_features=True)
                cls_loss = criterion_cls(logits, lbls)

                # 2. Domain Classification (0: Axial, 1: Coronal, 2: Sagittal)
                domain_logits = discriminator(grl.apply(features, grl.alpha))
                domain_lbls = torch.full(
                    (imgs.size(0),), d_idx, dtype=torch.long, device=device
                )
                domain_loss = criterion_domain(domain_logits, domain_lbls)

                # Loss 역전파 (배치마다 누적)
                loss = cls_loss + dann_weight * domain_loss
                loss.backward()

                total_cls_loss += cls_loss.item()
                total_dom_loss += domain_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 매 에포크마다 20% 분리된 검증셋으로 성능 체크
        if val_datasets is not None:
            scores = evaluate_all(model, *val_datasets, device=device, verbose=False)
            metric = scores["Final"]
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch [{epoch+1:2d}/{epochs}] Cls: {total_cls_loss/num_iters:.3f}, Dom: {total_dom_loss/num_iters:.3f}, Alpha: {grl.alpha:.3f} | best_val={best_metric:.4f}"
                )
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch [{epoch+1:2d}/{epochs}] Cls: {total_cls_loss/num_iters:.3f}, Dom: {total_dom_loss/num_iters:.3f}, Alpha: {grl.alpha:.3f}"
                )

    if val_datasets is not None:
        model.load_state_dict(best_state)
        print(
            f"  DANN Training done. Best validation metric: {best_metric:.4f} at Epoch {best_epoch}"
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
    parser.add_argument("--pretrain_epochs", type=int, default=150)
    parser.add_argument("--ft_epochs", type=int, default=80)
    parser.add_argument("--dann_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    # 2. Build Model
    model = build_model(num_classes=11, channels=64, modes=12).to(device)
    feature_dim = 64 * 2  # feature_expand layer output
    discriminator = DomainDiscriminator(feature_dim).to(device)

    # 3. Pre-training on A + C
    model = pretrain(
        model, axial, coronal, device, epochs=args.pretrain_epochs, lr=1e-3
    )
    pretrained_state = copy.deepcopy(model.state_dict())

    # 4. [핵심] Sagittal 80(Train) : 20(Val) 분할
    print("\n[Step 2] Internal Validation to find best DANN Epoch...")
    train_idx, val_idx = stratified_split_indices(
        sagittal_eval.labels, val_per_class=10, seed=42
    )
    sagittal_train_split = MedSubset(sagittal_eval, train_idx, augment=True)
    sagittal_val_split = MedSubset(sagittal_eval, val_idx, augment=False)

    # DataLoader 준비
    a_loader = DataLoader(
        axial, batch_size=64, shuffle=True, drop_last=True, num_workers=4
    )
    c_loader = DataLoader(
        coronal, batch_size=64, shuffle=True, drop_last=True, num_workers=4
    )
    s_val_loader = DataLoader(
        sagittal_train_split, batch_size=32, shuffle=True, drop_last=True, num_workers=4
    )

    # 20% 평가 셋으로 최고 성능 지점 찾기
    _, best_epoch = train_dann_with_eval(
        model,
        discriminator,
        a_loader,
        c_loader,
        s_val_loader,
        val_datasets=(axial_eval, coronal_eval, sagittal_val_split),
        device=device,
        epochs=args.ft_epochs,
        lr=args.lr,
        dann_weight=args.dann_weight,
    )

    # 5. [핵심] 찾은 최적 Epoch로 100% Sagittal 데이터 최종 DANN 학습
    print(
        f"\n[Final Training] DANN Fine-tune on ALL Sagittal shots for {best_epoch} Epochs"
    )
    model.load_state_dict(pretrained_state)  # 파인튜닝 전 롤백

    s_full_loader = DataLoader(
        sagittal, batch_size=32, shuffle=True, drop_last=True, num_workers=4
    )
    discriminator = DomainDiscriminator(feature_dim).to(
        device
    )  # Discriminator도 초기화

    model, _ = train_dann_with_eval(
        model,
        discriminator,
        a_loader,
        c_loader,
        s_full_loader,
        val_datasets=None,
        device=device,
        epochs=best_epoch,
        lr=args.lr,
        dann_weight=args.dann_weight,
    )

    # 6. 최종 평가 및 저장
    print("\n[Step 4] Final Evaluation (F1 Score)...")
    evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"num_classes": 11, "channels": 64, "modes": 12},
        },
        "./dann_fno_final.pth",
    )
    print("\nModel saved to: ./dann_fno_final.pth")


if __name__ == "__main__":
    main()
