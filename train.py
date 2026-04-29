import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.func import functional_call
from model import build_model

# ══════════════════════════════════════════════════════════
# 0. Dataset & Helpers
# ══════════════════════════════════════════════════════════


def _rotate_tensor(img, angle_deg):
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
    grid = F.affine_grid(
        theta.unsqueeze(0), img.unsqueeze(0).shape, align_corners=False
    )
    rotated = F.grid_sample(
        img.unsqueeze(0), grid, align_corners=False, mode="bilinear"
    )
    return rotated.squeeze(0)


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

    def __init__(self, image_dir, label_csv, augment=False):
        import pandas as pd
        from PIL import Image as PILImage

        self.image_dir = image_dir
        self.augment = augment
        raw = pd.read_csv(label_csv, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        one_hot = raw[self.CLASS_NAMES].astype(int).values
        self.labels = torch.tensor(one_hot.argmax(axis=1), dtype=torch.long)

        imgs = []
        filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        for fn in filenames:
            img = (
                PILImage.open(os.path.join(image_dir, fn)).convert("L").resize((28, 28))
            )
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        self.imgs = torch.from_numpy(np.stack(imgs))[:, None, :, :]
        self.imgs = (self.imgs - 0.5) / 0.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, lbl = self.imgs[idx].clone(), self.labels[idx]
        if self.augment:
            if torch.rand(1) < 0.5:
                img = _rotate_tensor(img, (torch.rand(1).item() - 0.5) * 20)
            if torch.rand(1) < 0.5:
                img = (
                    img * (0.8 + torch.rand(1).item() * 0.4)
                    + (torch.rand(1).item() - 0.5) * 0.1
                ).clamp(-1, 1)
        return img, lbl


def load_domain(image_dir, label_csv, augment=False):
    return MedDataset(image_dir, label_csv, augment=augment)


# ══════════════════════════════════════════════════════════
# 1. Training & Meta-Learning Functions
# ══════════════════════════════════════════════════════════


def pretrain(
    model, axial_dataset, coronal_dataset, device, epochs=40, batch_size=64, lr=1e-3
):
    print("\n[Step 1] Pre-training on Axial + Coronal...")
    combined = torch.utils.data.ConcatDataset([axial_dataset, coronal_dataset])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    model.train()
    for epoch in range(epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:  # 5에폭마다 출력
            print(f"  Epoch [{epoch+1}/{epochs}] Last Loss: {loss.item():.4f}")
    return model


class TaskSampler:
    def __init__(self, dataset, n_way=11, k_shot=5, q_query=15):
        self.dataset = dataset
        labels = (
            dataset.labels
            if hasattr(dataset, "labels")
            else torch.cat([d.labels for d in dataset.datasets])
        )
        self.class_indices = {i: np.where(labels.numpy() == i)[0] for i in range(11)}

    def sample_task(self, device):
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
        for new_lbl in range(11):
            idx = np.random.choice(self.class_indices[new_lbl], 20, replace=False)
            for i in idx[:5]:
                img, _ = self.dataset[i]
                s_imgs.append(img)
                s_lbls.append(new_lbl)
            for i in idx[5:]:
                img, _ = self.dataset[i]
                q_imgs.append(img)
                q_lbls.append(new_lbl)
        return (
            torch.stack(s_imgs).to(device),
            torch.tensor(s_lbls).to(device),
            torch.stack(q_imgs).to(device),
            torch.tensor(q_lbls).to(device),
        )


def fomaml_step(model, task_sampler, device, inner_lr=0.01, inner_steps=3, n_tasks=4):
    criterion = nn.CrossEntropyLoss()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    meta_grads = {n: torch.zeros_like(p) for n, p in params.items()}
    total_q_loss = 0.0
    for _ in range(n_tasks):
        s_imgs, s_lbls, q_imgs, q_lbls = task_sampler.sample_task(device)
        fast_params = {n: p.clone() for n, p in params.items()}
        for _s in range(inner_steps):
            logits = functional_call(model, fast_params, (s_imgs,))
            loss = criterion(logits, s_lbls)
            grads = torch.autograd.grad(loss, fast_params.values(), allow_unused=True)
            fast_params = {
                n: p - inner_lr * g if g is not None else p
                for (n, p), g in zip(fast_params.items(), grads)
            }
        q_logits = functional_call(model, fast_params, (q_imgs,))
        q_loss = criterion(q_logits, q_lbls)
        total_q_loss += q_loss.item()
        q_grads = torch.autograd.grad(q_loss, params.values(), allow_unused=True)
        for (n, _), g in zip(params.items(), q_grads):
            if g is not None:
                meta_grads[n] += g / n_tasks
    for n, p in params.items():
        p.grad = meta_grads[n]
    return total_q_loss / n_tasks


def meta_train(model, axial, coronal, device, meta_epochs=30):
    print("\n[Step 2] FOMAML Meta-training...")
    combined = torch.utils.data.ConcatDataset([axial, coronal])
    sampler = TaskSampler(combined)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(meta_epochs):
        q_loss = fomaml_step(model, sampler, device)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Meta-epoch [{epoch+1}/{meta_epochs}] q_loss={q_loss:.4f}")
    return model


def finetune_l2sp(model, dataset, device, epochs=50, batch_size=32, lr=2e-4, lam=0.05):
    print(f"\n[Step 3] Fine-tuning on Sagittal (λ={lam})...")
    theta_meta = {n: p.clone().detach() for n, p in model.named_parameters()}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            l2sp = sum(
                ((p - theta_meta[n].to(device)) ** 2).sum()
                for n, p in model.named_parameters()
            )
            loss = criterion(logits, lbls) + lam * l2sp
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, dataset, device):
    from sklearn.metrics import f1_score

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            all_preds.extend(model(imgs).argmax(1).cpu().numpy())
            all_lbls.extend(lbls.numpy())
    return f1_score(all_lbls, all_preds, average="macro", zero_division=0)


# ══════════════════════════════════════════════════════════
# 2. Main Process
# ══════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./best_model.pth")
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--meta_epochs", type=int, default=60)
    parser.add_argument("--finetune_epochs", type=int, default=250)
    parser.add_argument("--lambda_val", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    t = args.train_dir
    axial_train = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), augment=True
    )
    coronal_train = MedDataset(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        augment=True,
    )

    # Stratified Split for Sagittal
    sag_full_no_aug = MedDataset(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        augment=False,
    )
    sag_full_aug = MedDataset(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        augment=True,
    )

    indices = np.arange(len(sag_full_no_aug))
    train_idx, val_idx = [], []
    for i in range(11):
        c_idx = np.where(sag_full_no_aug.labels.numpy() == i)[0]
        np.random.shuffle(c_idx)
        val_idx.extend(c_idx[:10])
        train_idx.extend(c_idx[10:])

    sag_train = Subset(sag_full_aug, train_idx)
    sag_val = Subset(sag_full_no_aug, val_idx)

    model = build_model(channels=64, modes=12).to(device)
    model = pretrain(
        model, axial_train, coronal_train, device, epochs=args.pretrain_epochs
    )
    model = meta_train(
        model, axial_train, coronal_train, device, meta_epochs=args.meta_epochs
    )
    model = finetune_l2sp(
        model, sag_train, device, epochs=args.finetune_epochs, lam=args.lambda_val
    )

    f1_val = evaluate(model, sag_val, device)
    print(f"\n⭐ Final Validation Sagittal F1: {f1_val:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"channels": 64, "modes": 12},
        },
        args.save_path,
    )
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
