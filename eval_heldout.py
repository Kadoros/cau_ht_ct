"""
eval_heldout.py
Sagittal을 train/test로 나눠서 held-out 평가.
lambda별 모델의 실제 일반화 성능을 비교.

Usage:
  python eval_heldout.py \
      --train_dir ./train \
      --models model_lam0005.pth model_lam001.pth model_lam002.pth model_lam003.pth
"""

import os, sys, copy, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from train import load_domain


def evaluate_f1(model, dataset, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds_all, lbls_all = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            preds_all.extend(preds)
            lbls_all.extend(lbls.numpy())
    return f1_score(lbls_all, preds_all, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="모델 파일들 (예: model_lam001.pth model_lam003.pth)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.3,
        help="Sagittal에서 test로 뗄 비율 (default: 0.3 = 132개)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    t = args.train_dir
    axial_eval = load_domain(
        os.path.join(t, "axial"), os.path.join(t, "label", "axial.csv"), augment=False
    )
    coronal_eval = load_domain(
        os.path.join(t, "coronal"),
        os.path.join(t, "label", "coronal.csv"),
        augment=False,
    )
    sag_full = load_domain(
        os.path.join(t, "sagittal"),
        os.path.join(t, "label", "sagittal.csv"),
        augment=False,
    )

    # Sagittal을 train/test로 분리
    n_test = max(11, int(len(sag_full) * args.test_ratio))  # 최소 클래스당 1개
    n_train = len(sag_full) - n_test
    sag_train, sag_test = random_split(
        sag_full, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"\nSagittal split: {n_train} train / {n_test} test")
    print(f"(test에서 F1 측정 = 실제 대회 환경 근사)\n")

    print(
        f"{'모델':<30} {'F1_S_train':>10} {'F1_S_test':>10} {'F1_A':>8} {'F1_C':>8} {'Final(test)':>12}"
    )
    print("-" * 85)

    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"  {model_path}: 파일 없음, 스킵")
            continue

        ckpt = torch.load(model_path, map_location=device)
        cfg = ckpt.get("config", {"num_classes": 11, "channels": 32, "modes": 8})
        model = build_model(
            cfg.get("num_classes", 11), cfg.get("channels", 32), cfg.get("modes", 8)
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        f1_s_train = evaluate_f1(model, sag_train, device)
        f1_s_test = evaluate_f1(model, sag_test, device)
        f1_a = evaluate_f1(model, axial_eval, device)
        f1_c = evaluate_f1(model, coronal_eval, device)
        final_test = 0.7 * f1_s_test + 0.3 * (f1_a + f1_c) / 2

        name = os.path.basename(model_path)
        print(
            f"{name:<30} {f1_s_train:>10.4f} {f1_s_test:>10.4f} {f1_a:>8.4f} {f1_c:>8.4f} {final_test:>12.4f}"
        )

    print("\n※ Final(test) 가 실제 대회 점수와 가장 가까운 추정치예요.")


if __name__ == "__main__":
    main()
