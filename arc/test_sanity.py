"""
test_sanity.py
더미 PNG + one-hot CSV로 전체 파이프라인 검증.
Usage: python test_sanity.py
"""

import os, csv, tempfile
import numpy as np
import torch
from PIL import Image

from model import build_model
from train import load_domain, pretrain, meta_train, finetune_l2sp, evaluate_all

# Device 자동 인식 (제출 서버와 Mac 동시 지원)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"✅ 사용 중인 Device: {device}")
print(f"✅ PyTorch Version: {torch.__version__}")

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


def make_dummy_domain(domain_dir, label_csv_path, n_per_class=20, n_classes=11):
    os.makedirs(domain_dir, exist_ok=True)
    os.makedirs(os.path.dirname(label_csv_path), exist_ok=True)
    rows = []
    idx = 0
    for cls in range(n_classes):
        for _ in range(n_per_class):
            fname = f"image_{idx:05d}"
            arr = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(
                os.path.join(domain_dir, fname + ".png")
            )
            one_hot = [0] * n_classes
            one_hot[cls] = 1
            rows.append([fname] + one_hot)
            idx += 1
    with open(label_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index"] + CLASS_NAMES)
        writer.writerows(rows)
    print(f"  Created {idx} images -> {domain_dir}")


def main():
    print("=" * 55)
    print("Sanity Check: Full Pipeline (PNG + one-hot CSV)")
    print("=" * 55)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        label_dir = os.path.join(tmpdir, "label")
        axial_dir = os.path.join(tmpdir, "axial")
        coronal_dir = os.path.join(tmpdir, "coronal")
        sagittal_dir = os.path.join(tmpdir, "sagittal")

        print("\nGenerating dummy data...")
        make_dummy_domain(
            axial_dir, os.path.join(label_dir, "axial.csv"), n_per_class=20
        )
        make_dummy_domain(
            coronal_dir, os.path.join(label_dir, "coronal.csv"), n_per_class=20
        )
        make_dummy_domain(
            sagittal_dir, os.path.join(label_dir, "sagittal.csv"), n_per_class=10
        )

        print("\nLoading datasets...")
        axial_train = load_domain(
            axial_dir, os.path.join(label_dir, "axial.csv"), augment=True
        )
        coronal_train = load_domain(
            coronal_dir, os.path.join(label_dir, "coronal.csv"), augment=True
        )
        sagittal_ft = load_domain(
            sagittal_dir, os.path.join(label_dir, "sagittal.csv"), augment=True
        )
        axial_eval = load_domain(
            axial_dir, os.path.join(label_dir, "axial.csv"), augment=False
        )
        coronal_eval = load_domain(
            coronal_dir, os.path.join(label_dir, "coronal.csv"), augment=False
        )
        sagittal_eval = load_domain(
            sagittal_dir, os.path.join(label_dir, "sagittal.csv"), augment=False
        )

        model = build_model(num_classes=11, channels=32, modes=8)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total:,}")

        model = pretrain(
            model, axial_train, coronal_train, device, epochs=2, batch_size=16
        )
        model = meta_train(
            model, axial_train, coronal_train, device, meta_epochs=3, n_tasks=2
        )
        model = finetune_l2sp(model, sagittal_ft, device, epochs=3, lam=0.1)
        scores = evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

        save_path = os.path.join(tmpdir, "model.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {"num_classes": 11, "channels": 32, "modes": 8},
                "scores": scores,
            },
            save_path,
        )

        from inference import load_model, preprocess_images, predict_with_tta

        loaded = load_model(save_path, device)
        raw = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
        preds = predict_with_tta(loaded, preprocess_images(raw), device)
        assert preds.shape == (20,)
        assert all(0 <= p <= 10 for p in preds)

    print("\n✅ All checks passed!")
    print("\n실제 학습:")
    print("  python train.py --train_dir ./train --save_path ./best_model.pth")


if __name__ == "__main__":
    main()
