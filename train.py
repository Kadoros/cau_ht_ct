"""
train.py
Full training pipeline:
  Step 0: Data loading & augmentation
  Step 1: Pre-training on A+C (source domains)
  Step 2: FOMAML meta-learning on A+C
  Step 3: Few-shot fine-tuning on S (Sagittal) with L2-SP

Usage:
  python train.py \
      --train_dir   ./train \
      --save_path   ./best_model.pth

Data directory format expected:
  train/
    axial/          image_00000.png, image_00001.png, ...
    coronal/        image_00000.png, ...
    sagittal/       image_00000.png, ...
    label/
      axial.csv
      coronal.csv
      sagittal.csv
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from model import build_model


# ══════════════════════════════════════════════════════════
# 0. Dataset & Augmentation
# ══════════════════════════════════════════════════════════


class MedDataset(Dataset):
    """
    Loads PNG images + one-hot CSV label file.

    CSV format (actual competition format):
      Index,bladder,femur-left,femur-right,heart,kidney-left,kidney-right,liver,lung-left,lung-right,pancreas,spleen
      image_00000,0,0,0,0,0,0,1,0,0,0,0

    Class index mapping (argmax of one-hot row):
      bladder=0, femur-left=1, femur-right=2, heart=3,
      kidney-left=4, kidney-right=5, liver=6,
      lung-left=7, lung-right=8, pancreas=9, spleen=10
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

    def __init__(self, image_dir, label_csv, augment=False):
        import pandas as pd
        from PIL import Image as PILImage

        self.image_dir = image_dir
        self.augment = augment

        # ── Load one-hot CSV ──
        # Some CSV files have duplicate header rows — handle robustly
        raw = pd.read_csv(label_csv, index_col=0, dtype=str)

        # Drop rows where the index value literally equals "Index" (dup headers)
        raw = raw[raw.index != "Index"]

        # Verify all class columns are present
        missing = [c for c in self.CLASS_NAMES if c not in raw.columns]
        assert not missing, f"Missing class columns in CSV: {missing}"

        # Convert one-hot → class index
        one_hot = raw[self.CLASS_NAMES].astype(int).values  # (N, 11)
        labels = one_hot.argmax(axis=1)  # (N,) int

        # filename index (e.g. "image_00000")
        filenames = raw.index.tolist()

        # ── Build image list matched by filename ──
        fname_to_label = {fn: int(lbl) for fn, lbl in zip(filenames, labels)}

        all_pngs = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.image_paths = []
        self.labels_list = []
        for fn in all_pngs:
            key = fn.replace(".png", "")
            if key in fname_to_label:
                self.image_paths.append(os.path.join(image_dir, fn))
                self.labels_list.append(fname_to_label[key])

        assert (
            len(self.image_paths) > 0
        ), f"No images matched between {image_dir} and {label_csv}"

        # ── Pre-load all images into memory (28x28 = tiny) ──
        imgs = []
        for path in self.image_paths:
            img = PILImage.open(path).convert("L")  # Grayscale
            img = img.resize((28, 28), PILImage.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
            arr = (arr - 0.5) / 0.5  # Normalize(0.5, 0.5)
            imgs.append(arr)

        self.imgs = torch.from_numpy(np.stack(imgs))[:, None, :, :]  # (N,1,28,28)
        self.labels = torch.tensor(self.labels_list, dtype=torch.long)

        print(f"    Loaded {len(self.labels)} images from {image_dir}")
        print(
            f"    Label range: {self.labels.min().item()} ~ {self.labels.max().item()}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].clone()
        lbl = self.labels[idx]
        if self.augment:
            img = self._augment(img)
        return img, lbl

    def _augment(self, img):
        """Lightweight augmentation safe for medical images."""
        if torch.rand(1) < 0.3:
            img = torch.flip(img, dims=[2])

        if torch.rand(1) < 0.5:
            angle = (torch.rand(1).item() - 0.5) * 20
            img = _rotate_tensor(img, angle)

        if torch.rand(1) < 0.4:
            img = img + torch.randn_like(img) * 0.02
            img = img.clamp(0, 1)

        if torch.rand(1) < 0.4:
            alpha = 0.8 + torch.rand(1).item() * 0.4
            beta = (torch.rand(1).item() - 0.5) * 0.1
            img = (img * alpha + beta).clamp(0, 1)

        return img


def _rotate_tensor(img, angle_deg):
    """Rotate a (1, H, W) tensor by angle_deg degrees."""
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
    grid = F.affine_grid(
        theta.unsqueeze(0), img.unsqueeze(0).shape, align_corners=False
    )
    rotated = F.grid_sample(
        img.unsqueeze(0), grid, align_corners=False, mode="bilinear"
    )
    return rotated.squeeze(0)


def load_domain(image_dir, label_csv, augment=False):
    """
    Args:
      image_dir : e.g. ./train/axial
      label_csv : e.g. ./train/label/axial.csv
    """
    return MedDataset(image_dir, label_csv, augment=augment)


# ══════════════════════════════════════════════════════════
# 1. Pre-training on Source Domains (A + C)
# ══════════════════════════════════════════════════════════


def pretrain(
    model, axial_dataset, coronal_dataset, device, epochs=40, batch_size=64, lr=1e-3
):
    """Standard cross-entropy training on combined A+C data."""
    print("\n[Step 1] Pre-training on Axial + Coronal...")

    combined = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])
    loader = DataLoader(
        combined, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    model.to(device)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += len(lbls)

        scheduler.step()
        avg_loss = total_loss / total
        acc = correct / total * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f}  acc={acc:.1f}%"
            )

    model.load_state_dict(best_state)
    print(f"  Pre-training done. Best loss: {best_loss:.4f}")
    return model


# ══════════════════════════════════════════════════════════
# 2. FOMAML Meta-Learning on Source Domains
# ══════════════════════════════════════════════════════════


class TaskSampler:
    """
    Samples N-way K-shot tasks from a dataset.
    Ensures support and query sets come from distinct indices.
    """

    def __init__(self, dataset, n_way=11, k_shot=5, q_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Build class -> indices mapping
        labels = dataset.labels.numpy()
        self.class_indices = {}
        for cls in range(11):
            idx = np.where(labels == cls)[0]
            if len(idx) > 0:
                self.class_indices[cls] = idx
        self.classes = list(self.class_indices.keys())

    def sample_task(self, device):
        # Pick n_way classes (use all 11 if available)
        n = min(self.n_way, len(self.classes))
        chosen_classes = np.random.choice(self.classes, size=n, replace=False)

        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for new_lbl, cls in enumerate(chosen_classes):
            idx = self.class_indices[cls]
            needed = self.k_shot + self.q_query
            if len(idx) < needed:
                chosen = np.random.choice(idx, size=needed, replace=True)
            else:
                chosen = np.random.choice(idx, size=needed, replace=False)

            s_idx = chosen[: self.k_shot]
            q_idx = chosen[self.k_shot :]

            for i in s_idx:
                img, _ = self.dataset[i]
                support_imgs.append(img)
                support_lbls.append(new_lbl)
            for i in q_idx:
                img, _ = self.dataset[i]
                query_imgs.append(img)
                query_lbls.append(new_lbl)

        s_imgs = torch.stack(support_imgs).to(device)
        s_lbls = torch.tensor(support_lbls, dtype=torch.long).to(device)
        q_imgs = torch.stack(query_imgs).to(device)
        q_lbls = torch.tensor(query_lbls, dtype=torch.long).to(device)

        return s_imgs, s_lbls, q_imgs, q_lbls


def fomaml_step(model, task_sampler, device, inner_lr=0.01, inner_steps=3, n_tasks=4):
    """
    First-Order MAML (FOMAML) update step.
    Returns total query loss for logging.
    """
    criterion = nn.CrossEntropyLoss()
    meta_grads = None  # accumulated gradients

    total_q_loss = 0.0

    for _ in range(n_tasks):
        s_imgs, s_lbls, q_imgs, q_lbls = task_sampler.sample_task(device)

        # ── Inner loop: clone params, do gradient steps ──
        fast_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for _step in range(inner_steps):
            # Forward with fast_weights
            logits = _forward_with_weights(model, s_imgs, fast_weights)
            loss = criterion(logits, s_lbls)
            grads = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=False, allow_unused=True
            )
            fast_weights = {
                name: param
                - inner_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        # ── Outer loop: compute query loss with fast_weights ──
        q_logits = _forward_with_weights(model, q_imgs, fast_weights)
        q_loss = criterion(q_logits, q_lbls)
        total_q_loss += q_loss.item()

        # Accumulate gradients w.r.t. original model params
        task_grads = torch.autograd.grad(
            q_loss, model.parameters(), retain_graph=False, allow_unused=True
        )
        if meta_grads is None:
            meta_grads = [g.clone() if g is not None else None for g in task_grads]
        else:
            for i, g in enumerate(task_grads):
                if g is not None:
                    if meta_grads[i] is None:
                        meta_grads[i] = g.clone()
                    else:
                        meta_grads[i] += g.clone()

    # Apply averaged meta-gradients
    with torch.no_grad():
        for param, grad in zip(model.parameters(), meta_grads):
            if grad is not None:
                param.grad = grad / n_tasks

    return total_q_loss / n_tasks


def _forward_with_weights(model, x, weights):
    """
    Run a functional forward pass using custom weight dict.
    We temporarily replace model params (no in-place modification).
    Uses torch.nn.utils.parametrize trick via manual assignment.
    """
    # Save originals
    orig = {name: param.data for name, param in model.named_parameters()}

    # Assign fast weights
    for name, param in model.named_parameters():
        if name in weights:
            param.data = weights[name]

    out = model(x)

    # Restore originals
    for name, param in model.named_parameters():
        param.data = orig[name]

    return out


def meta_train(
    model,
    axial_dataset,
    coronal_dataset,
    device,
    meta_epochs=30,
    meta_lr=1e-3,
    inner_lr=0.05,
    inner_steps=3,
    n_tasks=4,
    k_shot=5,
):
    print("\n[Step 2] FOMAML meta-training on Axial + Coronal...")

    # Combine both domains for task sampling
    combined = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])
    # Wrap to expose .labels
    combined_labels = torch.cat([axial_dataset.labels, coronal_dataset.labels])
    combined.labels = combined_labels
    combined.dataset = combined  # make __getitem__ work via TaskSampler

    # Build task sampler from each domain separately, then pick randomly
    axial_sampler = TaskSampler(axial_dataset, n_way=11, k_shot=k_shot, q_query=15)
    coronal_sampler = TaskSampler(coronal_dataset, n_way=11, k_shot=k_shot, q_query=15)
    samplers = [axial_sampler, coronal_sampler]

    meta_optimizer = torch.optim.AdamW(
        model.parameters(), lr=meta_lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=meta_epochs
    )

    model.to(device)

    for epoch in range(meta_epochs):
        model.train()
        meta_optimizer.zero_grad()

        # Alternate between axial and coronal task samplers
        sampler = samplers[epoch % 2]
        q_loss = fomaml_step(
            model,
            sampler,
            device,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            n_tasks=n_tasks,
        )

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        meta_optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Meta-epoch [{epoch+1:3d}/{meta_epochs}] query_loss={q_loss:.4f}")

    print("  Meta-training done.")
    return model


# ══════════════════════════════════════════════════════════
# 3. Few-shot Fine-tuning on Sagittal with L2-SP
# ══════════════════════════════════════════════════════════


def finetune_l2sp(
    model,
    sagittal_dataset,
    device,
    epochs=50,
    batch_size=32,
    lr=2e-4,
    lam=0.1,
    val_ratio=0.1,
):
    """
    Fine-tune on Sagittal 50-shot data with L2-SP regularization.
    L2-SP: loss = CE + λ * ||θ - θ_meta||²

    BN safety: We use GroupNorm (no running stats issue).
    """
    print("\n[Step 3] Fine-tuning on Sagittal with L2-SP (λ={:.3f})...".format(lam))

    # Save meta weights as anchor (θ_meta)
    theta_meta = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # 전체 440개로 학습 (val split 없음 - 샘플이 너무 적어서 val이 오히려 노이즈)
    train_loader = DataLoader(
        sagittal_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.to(device)

    best_train_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)

            ce_loss = criterion(logits, lbls)
            l2sp_reg = l2sp_penalty(model, theta_meta, device)
            loss = ce_loss + lam * l2sp_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (logits.argmax(1) == lbls).sum().item()
            total += len(lbls)

        scheduler.step()
        avg_loss = total_loss / total
        acc = correct / total * 100

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"  FT Epoch [{epoch+1:3d}/{epochs}] loss={avg_loss:.4f}  acc={acc:.1f}%"
            )

    model.load_state_dict(best_state)
    print(f"  Fine-tuning done. Best train loss: {best_train_loss:.4f}")
    return model


def l2sp_penalty(model, theta_meta, device):
    """Compute L2-SP penalty: sum of squared difference from meta weights."""
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in theta_meta:
            penalty = penalty + ((param - theta_meta[name].to(device)) ** 2).sum()
    return penalty


# ══════════════════════════════════════════════════════════
# 4. Evaluation Helper
# ══════════════════════════════════════════════════════════


def evaluate(model, dataset, device, batch_size=128):
    """Compute macro F1-score on a dataset."""
    from sklearn.metrics import f1_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    model.to(device)

    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_lbls.extend(lbls.numpy())

    f1 = f1_score(all_lbls, all_preds, average="macro", zero_division=0)
    return f1


def evaluate_all(model, axial_dataset, coronal_dataset, sagittal_dataset, device):
    f1_a = evaluate(model, axial_dataset, device)
    f1_c = evaluate(model, coronal_dataset, device)
    f1_s = evaluate(model, sagittal_dataset, device)
    final = 0.7 * f1_s + 0.3 * (f1_a + f1_c) / 2
    print(f"\n[Evaluation]")
    print(f"  F1 Axial    (A): {f1_a:.4f}")
    print(f"  F1 Coronal  (C): {f1_c:.4f}")
    print(f"  F1 Sagittal (S): {f1_s:.4f}")
    print(f"  Final Score    : {final:.4f}  (0.7×S + 0.3×(A+C)/2)")
    return {"F1_A": f1_a, "F1_C": f1_c, "F1_S": f1_s, "Final": final}


# ══════════════════════════════════════════════════════════
# 5. Lambda Search (optional)
# ══════════════════════════════════════════════════════════


def search_lambda(
    model_init_state,
    axial_dataset,
    coronal_dataset,
    sagittal_dataset,
    device,
    lambdas=(0.01, 0.05, 0.1, 0.3, 0.5),
):
    """
    Grid search over lambda values.
    Tries each lambda, returns best lambda by Final Score.
    Use only if you have enough time budget.
    """
    print("\n[Lambda Search]")
    best_lam = lambdas[0]
    best_score = -1

    for lam in lambdas:
        model = build_model()
        model.load_state_dict(copy.deepcopy(model_init_state))
        model = finetune_l2sp(
            model, sagittal_dataset, device, epochs=30, lam=lam, val_ratio=0.2
        )
        scores = evaluate_all(
            model, axial_dataset, coronal_dataset, sagittal_dataset, device
        )
        print(f"  λ={lam:.3f}  Final={scores['Final']:.4f}")
        if scores["Final"] > best_score:
            best_score = scores["Final"]
            best_lam = lam

    print(f"  Best λ = {best_lam}  (Final={best_score:.4f})")
    return best_lam


# ══════════════════════════════════════════════════════════
# 6. Main Entry Point
# ══════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-Shot Domain Generalization Training"
    )
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Root train dir containing axial/, coronal/, sagittal/, label/",
    )
    parser.add_argument(
        "--save_path", default="./best_model.pth", help="Output model path"
    )
    # MPS (Apple Silicon) > CUDA > CPU 순으로 자동 선택
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument("--device", default=_default_device)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--meta_epochs", type=int, default=40)
    parser.add_argument("--finetune_epochs", type=int, default=150)
    parser.add_argument(
        "--lambda_val", type=float, default=0.05, help="L2-SP regularization weight"
    )
    parser.add_argument(
        "--search_lambda",
        action="store_true",
        help="Run lambda grid search before final training",
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: channels={args.channels}, modes={args.modes}, λ={args.lambda_val}")

    # ── Load datasets ──
    print("\nLoading datasets...")
    t = args.train_dir
    axial_train = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), augment=True
    )
    coronal_train = load_domain(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        augment=True,
    )
    sagittal_ft = load_domain(
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
    print(f"  Sagittal: {len(sagittal_ft)} samples (50-shot per class)")

    # ── Build model ──
    model = build_model(num_classes=11, channels=args.channels, modes=args.modes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params:,}")

    # ── Step 1: Pre-training ──
    model = pretrain(
        model, axial_train, coronal_train, device, epochs=args.pretrain_epochs
    )

    # ── Step 2: FOMAML ──
    model = meta_train(
        model, axial_train, coronal_train, device, meta_epochs=args.meta_epochs
    )

    # Save meta weights (anchor for L2-SP)
    meta_state = copy.deepcopy(model.state_dict())

    # ── Optional: Lambda search ──
    lam = args.lambda_val
    if args.search_lambda:
        lam = search_lambda(
            meta_state,
            axial_eval,
            coronal_eval,
            sagittal_eval,
            device,
            lambdas=(0.01, 0.05, 0.1, 0.3, 0.5),
        )

    # ── Step 3: Fine-tuning with L2-SP ──
    model.load_state_dict(meta_state)
    model = finetune_l2sp(
        model, sagittal_ft, device, epochs=args.finetune_epochs, lam=lam
    )

    # ── Evaluation ──
    scores = evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

    # ── Save model ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_classes": 11,
                "channels": args.channels,
                "modes": args.modes,
            },
            "scores": scores,
        },
        args.save_path,
    )
    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
