"""
train.py
Few-shot domain generalization pipeline for the CAU medical-AI hackathon.

Main fixes compared with the initial version:
- Removed horizontal flip augmentation because the labels contain left/right organs.
- Replaced the unsafe param.data based FOMAML implementation with functional_call.
- Fixed episodic labels: tasks now keep the original 0~10 organ labels.
- Added cross-view meta-tasks: A->C and C->A, which better matches view generalization.
- Added stratified Sagittal validation for lambda search, then final training on all Sagittal shots.
- Fixed lambda search model config mismatch.
- Added layer-wise mean L2-SP regularization for safer few-shot adaptation.
- Uses only PyTorch/torchvision/numpy/pandas/sklearn and Python standard library.

Usage:
  python train.py --train_dir ./train --save_path ./model.pth

Recommended first run:
  python train.py --train_dir ./train --save_path ./model.pth --search_lambda
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

try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0 fallback; official env is PyTorch 2.1.x
    from torch.nn.utils.stateless import functional_call

from model_san1 import build_model


# ══════════════════════════════════════════════════════════
# 0. Dataset & Augmentation
# ══════════════════════════════════════════════════════════
class MedDataset(Dataset):
    """
    Loads PNG images + one-hot CSV label file.

    CSV format:
      Index,bladder,femur-left,femur-right,heart,kidney-left,kidney-right,liver,lung-left,lung-right,pancreas,spleen
      image_00000,0,0,0,0,0,0,1,0,0,0,0
    """

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
        self.image_dir = image_dir
        self.augment = augment

        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]  # robust to duplicated header rows

        missing = [c for c in self.CLASS_NAMES if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing class columns in CSV: {missing}")

        one_hot = raw[self.CLASS_NAMES].astype(int).values
        labels = one_hot.argmax(axis=1)
        filenames = raw.index.tolist()
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(filenames, labels)}

        all_pngs = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        )
        self.image_paths: List[str] = []
        self.labels_list: List[int] = []
        for fn in all_pngs:
            key = os.path.splitext(fn)[0]
            if key in fname_to_label:
                self.image_paths.append(os.path.join(image_dir, fn))
                self.labels_list.append(fname_to_label[key])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images matched between {image_dir} and {label_csv}")

        imgs = []
        for path in self.image_paths:
            img = read_image(path, mode=ImageReadMode.GRAY).float()  # (1,H,W)
            if img.shape[-2:] != (28, 28):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Most challenge PNGs are 8-bit. This branch is defensive for 16-bit PNGs.
            max_val = float(img.max().item()) if img.numel() > 0 else 255.0
            denom = 65535.0 if max_val > 255.0 else 255.0
            img = img / denom
            img = (img - 0.5) / 0.5  # [-1, 1]
            imgs.append(img)

        self.imgs = torch.stack(imgs).contiguous()
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)

        print(f"    Loaded {len(self.labels)} images from {image_dir}")
        print(
            f"    Label range: {self.labels.min().item()} ~ {self.labels.max().item()}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self._augment(img)
        return img, lbl

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Medical-image-safe augmentation for this label set.

        Do NOT use horizontal flip here: femur/kidney/lung have left/right labels.
        """
        # Mild geometry. 28x28 images are tiny, so keep transforms conservative.
        if torch.rand(()) < 0.55:
            angle = (torch.rand(()).item() - 0.5) * 24.0  # ±12 degrees
            tx = int(torch.randint(-1, 2, ()).item())
            ty = int(torch.randint(-1, 2, ()).item())
            scale = 0.95 + torch.rand(()).item() * 0.10  # 0.95~1.05
            img = _affine_tensor(
                img, angle_deg=angle, translate_px=(tx, ty), scale=scale
            )

        # Intensity robustness across patients/scanners.
        if torch.rand(()) < 0.65:
            alpha = 0.85 + torch.rand(()).item() * 0.30  # 0.85~1.15
            beta = (torch.rand(()).item() - 0.5) * 0.16  # -0.08~0.08
            img = img * alpha + beta

        if torch.rand(()) < 0.30:
            img = img + torch.randn_like(img) * 0.025

        return img.clamp(-1.0, 1.0)


class MedSubset(Dataset):
    """Subset wrapper that can turn augmentation on/off while preserving .labels."""

    def __init__(self, base: MedDataset, indices: Sequence[int], augment: bool = False):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.augment = augment
        self.labels = base.labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = int(self.indices[idx].item())
        img = self.base.imgs[base_idx].clone()
        lbl = self.base.labels[base_idx]
        if self.augment:
            img = self.base._augment(img)
        return img, lbl


def _affine_tensor(
    img: torch.Tensor,
    angle_deg: float,
    translate_px: Tuple[int, int] = (0, 0),
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a mild affine transform to a (1,H,W) tensor."""
    _, h, w = img.shape
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle) / scale
    sin_a = math.sin(angle) / scale
    tx = 2.0 * translate_px[0] / max(w, 1)
    ty = 2.0 * translate_px[1] / max(h, 1)

    theta = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
        dtype=img.dtype,
        device=img.device,
    )
    grid = F.affine_grid(
        theta.unsqueeze(0), img.unsqueeze(0).shape, align_corners=False
    )
    out = F.grid_sample(
        img.unsqueeze(0),
        grid,
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    )
    return out.squeeze(0)


def load_domain(image_dir: str, label_csv: str, augment: bool = False) -> MedDataset:
    return MedDataset(image_dir, label_csv, augment=augment)


def stratified_split_indices(
    labels: torch.Tensor, val_per_class: int = 10, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Return train/val indices with an equal validation count per class when possible."""
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_idx: List[int] = []
    val_idx: List[int] = []
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
# 1. Pre-training on Source Domains (A + C)
# ══════════════════════════════════════════════════════════
def pretrain(
    model: nn.Module,
    axial_dataset: Dataset,
    coronal_dataset: Dataset,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> nn.Module:
    print("\n[Step 1] Pre-training on Axial + Coronal...")

    combined = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])
    loader = DataLoader(
        combined, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.to(device)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(lbls)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == lbls).sum().item()
            total += bs

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1) * 100.0

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}%")

    model.load_state_dict(best_state)
    print(f"  Pre-training done. Best loss: {best_loss:.4f}")
    return model


# ══════════════════════════════════════════════════════════
# 2. FOMAML Meta-Learning on Source Domains
# ══════════════════════════════════════════════════════════
class TaskSampler:
    """Samples N-way K-shot tasks from one dataset while keeping original labels."""

    def __init__(
        self, dataset: Dataset, n_way: int = 11, k_shot: int = 5, q_query: int = 10
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.class_indices = _build_class_indices(dataset.labels)
        self.classes = list(self.class_indices.keys())

    def sample_task(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(self.n_way, len(self.classes))
        chosen_classes = np.random.choice(self.classes, size=n, replace=False)
        support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

        for cls in chosen_classes:
            idx = self.class_indices[int(cls)]
            needed = self.k_shot + self.q_query
            replace = len(idx) < needed
            chosen = np.random.choice(idx, size=needed, replace=replace)
            for i in chosen[: self.k_shot]:
                img, _ = self.dataset[int(i)]
                support_imgs.append(img)
                support_lbls.append(int(cls))
            for i in chosen[self.k_shot :]:
                img, _ = self.dataset[int(i)]
                query_imgs.append(img)
                query_lbls.append(int(cls))

        return _stack_task(support_imgs, support_lbls, query_imgs, query_lbls, device)


class CrossDomainTaskSampler:
    """
    Samples support from one view and query from another view.

    This better matches the challenge: learn from one plane, adapt/evaluate on a
    different plane while preserving the same organ label semantics.
    """

    def __init__(
        self,
        support_dataset: Dataset,
        query_dataset: Dataset,
        n_way: int = 11,
        k_shot: int = 5,
        q_query: int = 10,
    ):
        self.support_dataset = support_dataset
        self.query_dataset = query_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.support_indices = _build_class_indices(support_dataset.labels)
        self.query_indices = _build_class_indices(query_dataset.labels)
        self.classes = sorted(
            set(self.support_indices.keys()) & set(self.query_indices.keys())
        )

    def sample_task(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(self.n_way, len(self.classes))
        chosen_classes = np.random.choice(self.classes, size=n, replace=False)
        support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

        for cls in chosen_classes:
            s_idx = self.support_indices[int(cls)]
            q_idx = self.query_indices[int(cls)]
            s_chosen = np.random.choice(
                s_idx, size=self.k_shot, replace=len(s_idx) < self.k_shot
            )
            q_chosen = np.random.choice(
                q_idx, size=self.q_query, replace=len(q_idx) < self.q_query
            )

            for i in s_chosen:
                img, _ = self.support_dataset[int(i)]
                support_imgs.append(img)
                support_lbls.append(int(cls))
            for i in q_chosen:
                img, _ = self.query_dataset[int(i)]
                query_imgs.append(img)
                query_lbls.append(int(cls))

        return _stack_task(support_imgs, support_lbls, query_imgs, query_lbls, device)


def _build_class_indices(labels: torch.Tensor) -> Dict[int, np.ndarray]:
    labels_np = labels.cpu().numpy()
    out: Dict[int, np.ndarray] = {}
    for cls in range(11):
        idx = np.where(labels_np == cls)[0]
        if len(idx) > 0:
            out[cls] = idx
    return out


def _stack_task(
    support_imgs: List[torch.Tensor],
    support_lbls: List[int],
    query_imgs: List[torch.Tensor],
    query_lbls: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_imgs = torch.stack(support_imgs).to(device)
    s_lbls = torch.tensor(support_lbls, dtype=torch.long, device=device)
    q_imgs = torch.stack(query_imgs).to(device)
    q_lbls = torch.tensor(query_lbls, dtype=torch.long, device=device)
    return s_imgs, s_lbls, q_imgs, q_lbls


def _forward_with_weights(
    model: nn.Module, x: torch.Tensor, weights: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Functional forward pass with custom trainable parameters and original buffers."""
    buffers = dict(model.named_buffers())
    state = {**buffers, **weights}
    return functional_call(model, state, (x,))


def fomaml_step(
    model: nn.Module,
    task_sampler,
    device: torch.device,
    inner_lr: float = 0.02,
    inner_steps: int = 2,
    n_tasks: int = 4,
) -> float:
    """
    First-Order MAML update.

    We compute query gradients with respect to adapted fast weights and copy those
    first-order gradients to the original model parameters. No param.data hack is
    used, so autograd remains valid.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    meta_grads: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    total_q_loss = 0.0

    for _ in range(n_tasks):
        s_imgs, s_lbls, q_imgs, q_lbls = task_sampler.sample_task(device)

        fast_weights: Dict[str, torch.Tensor] = {
            name: param.detach().clone().requires_grad_(True)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for _step in range(inner_steps):
            logits = _forward_with_weights(model, s_imgs, fast_weights)
            loss = criterion(logits, s_lbls)
            grads = torch.autograd.grad(
                loss,
                tuple(fast_weights.values()),
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            updated: Dict[str, torch.Tensor] = {}
            for (name, weight), grad in zip(fast_weights.items(), grads):
                if grad is None:
                    updated[name] = weight.detach().clone().requires_grad_(True)
                else:
                    updated[name] = (
                        (weight - inner_lr * grad).detach().clone().requires_grad_(True)
                    )
            fast_weights = updated

        q_logits = _forward_with_weights(model, q_imgs, fast_weights)
        q_loss = criterion(q_logits, q_lbls)
        total_q_loss += q_loss.item()

        q_grads = torch.autograd.grad(
            q_loss,
            tuple(fast_weights.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for (name, _), grad in zip(fast_weights.items(), q_grads):
            if grad is not None:
                meta_grads[name] += grad.detach()

    for name, param in model.named_parameters():
        if param.requires_grad and name in meta_grads:
            param.grad = meta_grads[name] / float(max(n_tasks, 1))

    return total_q_loss / float(max(n_tasks, 1))


def meta_train(
    model: nn.Module,
    axial_dataset: Dataset,
    coronal_dataset: Dataset,
    device: torch.device,
    meta_epochs: int = 40,
    meta_lr: float = 5e-4,
    inner_lr: float = 0.02,
    inner_steps: int = 2,
    n_tasks: int = 4,
    k_shot: int = 5,
    q_query: int = 10,
) -> nn.Module:
    print("\n[Step 2] FOMAML meta-training on Axial + Coronal...")

    samplers = [
        CrossDomainTaskSampler(
            axial_dataset, coronal_dataset, n_way=11, k_shot=k_shot, q_query=q_query
        ),
        CrossDomainTaskSampler(
            coronal_dataset, axial_dataset, n_way=11, k_shot=k_shot, q_query=q_query
        ),
        TaskSampler(axial_dataset, n_way=11, k_shot=k_shot, q_query=q_query),
        TaskSampler(coronal_dataset, n_way=11, k_shot=k_shot, q_query=q_query),
    ]

    meta_optimizer = torch.optim.AdamW(
        model.parameters(), lr=meta_lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=max(1, meta_epochs)
    )
    model.to(device)

    for epoch in range(meta_epochs):
        model.train()
        meta_optimizer.zero_grad(set_to_none=True)
        sampler = samplers[epoch % len(samplers)]
        q_loss = fomaml_step(
            model,
            sampler,
            device,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            n_tasks=n_tasks,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        meta_optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Meta-epoch [{epoch+1:3d}/{meta_epochs}] query_loss={q_loss:.4f}")

    print("  Meta-training done.")
    return model


# ══════════════════════════════════════════════════════════
# 3. Few-shot Fine-tuning on Sagittal with L2-SP
# ══════════════════════════════════════════════════════════
def finetune_l2sp(
    model: nn.Module,
    sagittal_dataset: Dataset,
    device: torch.device,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 2e-4,
    lam: float = 0.05,
    val_datasets: Optional[Tuple[Dataset, Dataset, Dataset]] = None,
    eval_every: int = 5,
) -> nn.Module:
    """Fine-tune on Sagittal with layer-wise mean L2-SP regularization."""
    print(f"\n[Step 3] Fine-tuning on Sagittal with L2-SP (λ={lam:.4f})...")

    theta_meta = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    train_loader = DataLoader(
        sagittal_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf") if val_datasets is not None else float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            ce_loss = criterion(logits, lbls)
            reg = l2sp_penalty(model, theta_meta, device)
            loss = ce_loss + lam * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        elif val_datasets is None:
            # Final full-shot training has no validation split; use train loss only.
            if avg_loss < best_metric:
                best_metric = avg_loss
                best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_datasets is not None:
                print(
                    f"  FT Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}% best_val={best_metric:.4f}"
                )
            else:
                print(
                    f"  FT Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f} acc={acc:.1f}%"
                )

    model.load_state_dict(best_state)
    print(f"  Fine-tuning done. Best metric: {best_metric:.4f}")
    return model


def _l2sp_layer_weight(name: str) -> float:
    """Stronger anchor on early/shared encoder, weaker anchor on classifier."""
    if name.startswith("classifier"):
        return 0.10
    if name.startswith("spatial_refine"):
        return 0.35
    if name.startswith("fno2"):
        return 0.60
    return 1.00  # lifting + fno1


def l2sp_penalty(
    model: nn.Module, theta_meta: Dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Layer-wise mean L2-SP penalty. Mean reduction keeps λ numerically stable."""
    penalty = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if name not in theta_meta:
            continue
        anchor = theta_meta[name].to(device)
        penalty = penalty + _l2sp_layer_weight(name) * (param - anchor).pow(2).mean()
    return penalty


# ══════════════════════════════════════════════════════════
# 4. Evaluation Helper
# ══════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 128
) -> float:
    from sklearn.metrics import f1_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_lbls: List[int] = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_lbls.extend(lbls.cpu().numpy().tolist())

    return float(
        f1_score(
            all_lbls,
            all_preds,
            labels=list(range(11)),
            average="macro",
            zero_division=0,
        )
    )


def evaluate_all(
    model: nn.Module,
    axial_dataset: Dataset,
    coronal_dataset: Dataset,
    sagittal_dataset: Dataset,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    f1_a = evaluate(model, axial_dataset, device)
    f1_c = evaluate(model, coronal_dataset, device)
    f1_s = evaluate(model, sagittal_dataset, device)
    final = 0.7 * f1_s + 0.3 * (f1_a + f1_c) / 2.0
    scores = {"F1_A": f1_a, "F1_C": f1_c, "F1_S": f1_s, "Final": final}
    if verbose:
        print("\n[Evaluation]")
        print(f"  F1 Axial    (A): {f1_a:.4f}")
        print(f"  F1 Coronal  (C): {f1_c:.4f}")
        print(f"  F1 Sagittal (S): {f1_s:.4f}")
        print(f"  Final Score    : {final:.4f}  (0.7×S + 0.3×(A+C)/2)")
    return scores


# ══════════════════════════════════════════════════════════
# 5. Lambda Search
# ══════════════════════════════════════════════════════════
def parse_lambda_candidates(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def search_lambda(
    model_init_state: Dict[str, torch.Tensor],
    axial_eval: Dataset,
    coronal_eval: Dataset,
    sagittal_train: Dataset,
    sagittal_val: Dataset,
    device: torch.device,
    channels: int,
    modes: int,
    finetune_epochs: int,
    finetune_lr: float,
    lambdas: Sequence[float] = (0.01, 0.03, 0.05, 0.1, 0.2),
) -> float:
    print("\n[Lambda Search on stratified Sagittal validation]")
    best_lam = float(lambdas[0])
    best_score = -float("inf")

    for lam in lambdas:
        model = build_model(num_classes=11, channels=channels, modes=modes)
        model.load_state_dict(copy.deepcopy(model_init_state))
        model = finetune_l2sp(
            model,
            sagittal_train,
            device,
            epochs=finetune_epochs,
            lr=finetune_lr,
            lam=float(lam),
            val_datasets=(axial_eval, coronal_eval, sagittal_val),
            eval_every=5,
        )
        scores = evaluate_all(
            model, axial_eval, coronal_eval, sagittal_val, device, verbose=True
        )
        print(f"  λ={float(lam):.4f}  Val Final={scores['Final']:.4f}")
        if scores["Final"] > best_score:
            best_score = scores["Final"]
            best_lam = float(lam)

    print(f"  Best λ = {best_lam:.4f}  (Val Final={best_score:.4f})")
    return best_lam


# ══════════════════════════════════════════════════════════
# 6. Main Entry Point
# ══════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-Shot Domain Generalization Training"
    )
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Root train dir containing axial/, coronal/, sagittal/, label/",
    )
    parser.add_argument("--save_path", default="./model.pth", help="Output model path")

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", default=default_device)

    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--meta_epochs", type=int, default=50)
    parser.add_argument("--meta_lr", type=float, default=5e-4)
    parser.add_argument("--inner_lr", type=float, default=0.02)
    parser.add_argument("--inner_steps", type=int, default=2)
    parser.add_argument("--meta_tasks", type=int, default=4)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument(
        "--skip_meta",
        action="store_true",
        help="Skip FOMAML if you need a faster ablation",
    )

    parser.add_argument("--finetune_epochs", type=int, default=160)
    parser.add_argument("--finetune_lr", type=float, default=2e-4)
    parser.add_argument(
        "--lambda_val", type=float, default=0.05, help="L2-SP regularization weight"
    )
    parser.add_argument(
        "--search_lambda",
        action="store_true",
        help="Run lambda grid search using stratified Sagittal validation",
    )
    parser.add_argument("--lambda_candidates", default="0.01,0.03,0.05,0.1,0.2")
    parser.add_argument("--val_per_class", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Config: channels={args.channels}, modes={args.modes}, λ={args.lambda_val}")

    t = args.train_dir
    print("\nLoading datasets...")
    axial_train = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), augment=True
    )
    coronal_train = load_domain(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        augment=True,
    )
    sagittal_full_train = load_domain(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        augment=True,
    )

    axial_eval = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), augment=False
    )
    coronal_eval = load_domain(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        augment=False,
    )
    sagittal_eval = load_domain(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        augment=False,
    )

    print(f"  Axial   : {len(axial_train)} samples")
    print(f"  Coronal : {len(coronal_train)} samples")
    print(f"  Sagittal: {len(sagittal_full_train)} samples")

    model = build_model(num_classes=11, channels=args.channels, modes=args.modes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params:,}")

    model = pretrain(
        model,
        axial_train,
        coronal_train,
        device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
    )

    if not args.skip_meta:
        model = meta_train(
            model,
            axial_train,
            coronal_train,
            device,
            meta_epochs=args.meta_epochs,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            n_tasks=args.meta_tasks,
            k_shot=args.k_shot,
            q_query=args.q_query,
        )
    else:
        print("\n[Step 2] Skipped meta-training by --skip_meta")

    meta_state = copy.deepcopy(model.state_dict())

    lam = args.lambda_val
    if args.search_lambda:
        train_idx, val_idx = stratified_split_indices(
            sagittal_eval.labels, val_per_class=args.val_per_class, seed=args.seed
        )
        sagittal_train_split = MedSubset(sagittal_eval, train_idx, augment=True)
        sagittal_val_split = MedSubset(sagittal_eval, val_idx, augment=False)
        print(
            f"\nSagittal split for lambda search: train={len(sagittal_train_split)}, val={len(sagittal_val_split)}"
        )
        lam = search_lambda(
            meta_state,
            axial_eval,
            coronal_eval,
            sagittal_train_split,
            sagittal_val_split,
            device,
            channels=args.channels,
            modes=args.modes,
            finetune_epochs=max(30, args.finetune_epochs // 2),
            finetune_lr=args.finetune_lr,
            lambdas=parse_lambda_candidates(args.lambda_candidates),
        )

    print("\n[Final Training] Fine-tune on all Sagittal shots with selected λ")
    model = build_model(num_classes=11, channels=args.channels, modes=args.modes)
    model.load_state_dict(meta_state)
    model = finetune_l2sp(
        model,
        sagittal_full_train,
        device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        lam=lam,
        val_datasets=None,
    )

    scores = evaluate_all(
        model, axial_eval, coronal_eval, sagittal_eval, device, verbose=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_classes": 11,
            "channels": args.channels,
            "modes": args.modes,
        },
        "scores_on_train_domains": scores,
        "selected_lambda": lam,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.save_path)
    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
