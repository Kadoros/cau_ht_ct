"""
train.py
Ultimate DANN + ProtoNet + EWC Pipeline for the CAU medical-AI hackathon.
(RTX 3090 Optimized Version)
"""

from __future__ import annotations

import os
import math
import copy
import argparse
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from model import build_model


# ══════════════════════════════════════════════════════════
# 0. Dataset & Geometry Augmentation
# ══════════════════════════════════════════════════════════
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

    def __init__(
        self, image_dir: str, label_csv: str, domain_lbl: int, augment: bool = False
    ):
        self.domain_lbl = domain_lbl
        self.augment = augment

        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]

        one_hot = raw[self.CLASS_NAMES].astype(int).values
        labels = one_hot.argmax(axis=1)
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

            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = torch.zeros_like(img)
            img = (img - 0.5) / 0.5
            imgs.append(img)

        self.imgs = torch.stack(imgs).contiguous()
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)

        self.geo_aug = T.RandomApply(
            [T.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10)], p=0.6
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self.geo_aug(img)
            if torch.rand(()) < 0.65:
                img = (
                    img * (0.85 + torch.rand(()).item() * 0.3)
                    + (torch.rand(()).item() - 0.5) * 0.16
                )
        return img.clamp(-1.0, 1.0), lbl, self.domain_lbl


class MedSubset(Dataset):
    def __init__(self, base: MedDataset, indices: Sequence[int], augment: bool = False):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.augment = augment
        self.labels = base.labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        base_idx = int(self.indices[idx].item())
        img = self.base.imgs[base_idx].clone()
        lbl = self.base.labels[base_idx]
        if self.augment:
            img = self.base.geo_aug(img)
            if torch.rand(()) < 0.65:
                img = (
                    img * (0.85 + torch.rand(()).item() * 0.3)
                    + (torch.rand(()).item() - 0.5) * 0.16
                )
        return img.clamp(-1.0, 1.0), lbl, self.base.domain_lbl


def load_domain(
    image_dir: str, label_csv: str, domain_lbl: int, augment: bool = False
) -> MedDataset:
    return MedDataset(image_dir, label_csv, domain_lbl, augment=augment)


def stratified_split_indices(
    labels: torch.Tensor, val_per_class: int = 10, seed: int = 42
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_idx, val_idx = [], []
    for cls in range(11):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        if len(idx) == 0:
            continue
        n_val = min(val_per_class, max(1, len(idx) // 5))
        if len(idx) - n_val < 1:
            n_val = max(0, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ══════════════════════════════════════════════════════════
# 1. Step 1: DANN Pre-training
# ══════════════════════════════════════════════════════════
def pretrain_dann(
    model: nn.Module,
    axial_ds: Dataset,
    coronal_ds: Dataset,
    device: torch.device,
    epochs: int,
    lr: float,
):
    print("\n[Step 1] DANN Pre-training on Axial + Coronal (OneCycleLR)...")
    combined = torch.utils.data.ConcatDataset([axial_ds, coronal_ds])
    loader = DataLoader(
        combined,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs
    )

    criterion_organ = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_domain = nn.CrossEntropyLoss()

    model.to(device)
    total_steps = epochs * len(loader)
    step = 0
    best_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct_org = 0.0, 0

        for imgs, organ_lbls, domain_lbls in loader:
            imgs, organ_lbls, domain_lbls = (
                imgs.to(device),
                organ_lbls.to(device),
                domain_lbls.to(device),
            )
            optimizer.zero_grad(set_to_none=True)

            p = float(step) / total_steps
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            features = model.get_feature(imgs)
            logits_organ = model.classifier(features)
            logits_domain = model.domain_classifier(grad_reverse(features, alpha))

            loss_org = criterion_organ(logits_organ, organ_lbls)
            loss_dom = criterion_domain(logits_domain, domain_lbls)
            loss = loss_org + loss_dom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(imgs)
            correct_org += (logits_organ.argmax(1) == organ_lbls).sum().item()
            step += 1

        acc = correct_org / len(combined) * 100.0
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}] Loss: {total_loss/len(combined):.4f} | Organ Acc: {acc:.1f}%"
            )

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════
# 2. Step 2: Importance Check (Fisher Matrix)
# ══════════════════════════════════════════════════════════
def compute_fisher(
    model: nn.Module, dataset: Dataset, device: torch.device
) -> Dict[str, torch.Tensor]:
    print("\n[Step 2] Computing Fisher Information Matrix for EWC...")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    fisher = {
        n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
    }
    criterion = nn.CrossEntropyLoss()

    for imgs, lbls, _ in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2) / len(loader)

    return {n: f.detach() for n, f in fisher.items()}


# ══════════════════════════════════════════════════════════
# 3. Step 3: Few-shot Adapt (Proto-Init + EWC)
# ══════════════════════════════════════════════════════════
def exact_proto_init(model: nn.Module, support_ds: Dataset, device: torch.device):
    print("\n[Step 3-1] Executing Proto-Init...")
    loader = DataLoader(support_ds, batch_size=32, shuffle=False)
    model.eval()

    feats_list, lbls_list = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            feats_list.append(model.get_feature(imgs.to(device)))
            lbls_list.append(lbls.to(device))

    features, labels = torch.cat(feats_list), torch.cat(lbls_list)
    prototypes = torch.zeros(model.num_classes, features.shape[-1], device=device)

    for c in range(model.num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = features[mask].mean(dim=0)

    model.classifier.weight.data = 2.0 * prototypes
    model.classifier.bias.data = -(prototypes**2).sum(dim=1)
    return model


def finetune_ewc(
    model: nn.Module,
    target_ds: Dataset,
    fisher: Dict,
    theta_meta: Dict,
    device: torch.device,
    epochs: int,
    lr: float,
    lam_ewc: float,
    val_datasets=None,
    eval_every=5,
):
    print(f"\n[Step 3-2] Fine-tuning with EWC (λ={lam_ewc:.4f})...")
    loader = DataLoader(target_ds, batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf") if val_datasets is not None else float("inf")
    best_epoch = 1

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls, _ in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            ce_loss = criterion(logits, lbls)

            ewc_loss = torch.zeros((), device=device)
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    ewc_loss += (fisher[n] * (p - theta_meta[n]).pow(2)).sum()

            loss = ce_loss + (lam_ewc / 2.0) * ewc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(lbls)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == lbls).sum().item()
            total += bs

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1) * 100.0

        if val_datasets is not None and (
            (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        ):
            scores = evaluate_all(model, *val_datasets, device=device, verbose=False)
            metric = scores["Final"]
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
        elif val_datasets is None:
            if avg_loss < best_metric:
                best_metric = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_info = (
                f" best_val={best_metric:.4f}" if val_datasets is not None else ""
            )
            print(
                f"  FT Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}%{val_info}"
            )

    model.load_state_dict(best_state)
    print(f"  Fine-tuning done. Best metric: {best_metric:.4f} at Epoch {best_epoch}")
    return model, best_epoch


# ══════════════════════════════════════════════════════════
# Evaluation & Search
# ══════════════════════════════════════════════════════════
def evaluate_all(model, axial_ds, coronal_ds, sagittal_ds, device, verbose=True):
    from sklearn.metrics import f1_score

    def eval_domain(ds):
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
                trues.extend(lbls.numpy())
        return float(
            f1_score(
                trues, preds, labels=list(range(11)), average="macro", zero_division=0
            )
        )

    f1_a, f1_c, f1_s = (
        eval_domain(axial_ds),
        eval_domain(coronal_ds),
        eval_domain(sagittal_ds),
    )
    final = 0.7 * f1_s + 0.3 * (f1_a + f1_c) / 2.0
    scores = {"F1_A": f1_a, "F1_C": f1_c, "F1_S": f1_s, "Final": final}
    if verbose:
        print(
            f"\n[Evaluation]\n  F1 Axial(A): {f1_a:.4f} | Coronal(C): {f1_c:.4f} | Sagittal(S): {f1_s:.4f}"
        )
        print(f"  Final Score: {final:.4f}  (0.7×S + 0.3×(A+C)/2)")
    return scores


def search_ewc_lambda(
    model_init_state,
    fisher,
    theta_meta,
    axial_eval,
    coronal_eval,
    sagittal_train,
    sagittal_val,
    device,
    channels,
    modes,
    epochs,
    lr,
    lambdas,
):
    print("\n[Lambda Search for EWC on stratified Sagittal validation]")
    best_lam, best_score, best_epoch_to_use = lambdas[0], -float("inf"), 40

    for lam in lambdas:
        model = build_model(num_classes=11, channels=channels, modes=modes)
        model.load_state_dict(copy.deepcopy(model_init_state))

        model, best_epoch = finetune_ewc(
            model,
            sagittal_train,
            fisher,
            theta_meta,
            device,
            epochs,
            lr,
            lam,
            val_datasets=(axial_eval, coronal_eval, sagittal_val),
            eval_every=5,
        )
        scores = evaluate_all(
            model, axial_eval, coronal_eval, sagittal_val, device, verbose=True
        )

        if scores["Final"] > best_score:
            best_score = scores["Final"]
            best_lam = lam
            best_epoch_to_use = best_epoch

    print(
        f"  Best EWC λ = {best_lam:.2f} at Epoch {best_epoch_to_use} (Val Final={best_score:.4f})"
    )
    return best_lam, best_epoch_to_use


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./model.pth")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_epochs", type=int, default=160)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--lambda_ewc", type=float, default=200.0)
    parser.add_argument("--search_lambda", action="store_true")
    parser.add_argument("--lambda_candidates", default="50.0,100.0,200.0,500.0")
    parser.add_argument("--val_per_class", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    t = args.train_dir

    A_train = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), 0, augment=True
    )
    C_train = load_domain(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        1,
        augment=True,
    )
    S_full = load_domain(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        2,
        augment=True,
    )

    A_eval = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), 0
    )
    C_eval = load_domain(
        os.path.join(t, "coronal"), os.path.join(t, "label", "coronal.csv"), 1
    )
    S_eval = load_domain(
        os.path.join(t, "sagittal"), os.path.join(t, "label", "sagittal.csv"), 2
    )

    model = build_model(channels=args.channels, modes=args.modes)

    # Step 1: DANN Pre-train
    model = pretrain_dann(
        model,
        A_train,
        C_train,
        device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
    )

    # Step 2: Fisher Info Calculation
    combined_source = torch.utils.data.ConcatDataset([A_train, C_train])
    fisher = compute_fisher(model, combined_source, device)

    # Step 3: Proto-Init
    model = exact_proto_init(model, S_full, device)
    theta_meta = {n: p.detach().clone() for n, p in model.named_parameters()}
    proto_state = copy.deepcopy(model.state_dict())

    # Search or Set Lambda
    lam, optimal_epoch = args.lambda_ewc, args.finetune_epochs
    if args.search_lambda:
        train_idx, val_idx = stratified_split_indices(
            S_eval.labels, val_per_class=args.val_per_class, seed=args.seed
        )
        lam, optimal_epoch = search_ewc_lambda(
            proto_state,
            fisher,
            theta_meta,
            A_eval,
            C_eval,
            MedSubset(S_eval, train_idx, True),
            MedSubset(S_eval, val_idx, False),
            device,
            args.channels,
            args.modes,
            max(30, args.finetune_epochs // 2),
            args.finetune_lr,
            [float(x) for x in args.lambda_candidates.split(",")],
        )

    print(
        f"\n[Final Training] Fine-tune on all Sagittal shots for {optimal_epoch} Epochs"
    )
    model.load_state_dict(proto_state)
    model, _ = finetune_ewc(
        model,
        S_full,
        fisher,
        theta_meta,
        device,
        epochs=optimal_epoch,
        lr=args.finetune_lr,
        lam_ewc=lam,
        val_datasets=None,
    )

    scores = evaluate_all(model, A_eval, C_eval, S_eval, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_classes": 11,
                "channels": args.channels,
                "modes": args.modes,
            },
            "scores_on_train_domains": scores,
            "selected_lambda": lam,
            "early_stop_epoch": optimal_epoch,
            "seed": args.seed,
        },
        args.save_path,
    )
    print(f"\n✅ Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
