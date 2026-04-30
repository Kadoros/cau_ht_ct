import os
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from model import build_model
from train import load_domain, evaluate_all
from model_ewc import EWC


# ══════════════════════════════════════════════════════════
# 1. EWC Training Pipeline
# ══════════════════════════════════════════════════════════
def train_with_ewc(
    model, train_loader, ewc, device, epochs=50, lr=1e-4, ewc_lambda=1000
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n[Step 3] Fine-tuning on Sagittal with EWC (lambda={ewc_lambda})...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()

            output = model(imgs)
            ce_loss = criterion(output, lbls)

            # EWC Penalty: Source Domain의 중요 가중치 보호
            ewc_loss = ewc.penalty(model)
            loss = ce_loss + ewc_lambda * ewc_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == lbls).sum().item()
            total += len(lbls)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/total*100:.2f}%"
            )

    return model


# ══════════════════════════════════════════════════════════
# 2. Main Entry Point
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--ft_epochs", type=int, default=100)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 2. Pre-training on Source Domains (A + C)
    print("\n[Step 1] Pre-training on Axial + Coronal...")
    model = build_model(num_classes=11, channels=64, modes=12).to(device)
    source_dataset = ConcatDataset([axial, coronal])
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)

    # (Pre-training loop 생략 - 기존 train.py의 pretrain 함수 활용 가능)
    # 여기서는 데모를 위해 직접 간단히 구현하거나 기존 함수를 호출
    from train import pretrain

    model = pretrain(
        model, axial, coronal, device, epochs=args.pretrain_epochs, lr=1e-3
    )

    # 3. Calculate Fisher Information Matrix (FIM)
    print("\n[Step 2] Calculating Fisher Information Matrix for Source Domains...")
    # FIM 계산 시에는 augmentation이 없는 eval 데이터셋 사용 권장
    fisher_loader = DataLoader(
        ConcatDataset([axial_eval, coronal_eval]), batch_size=32, shuffle=False
    )
    ewc = EWC(model, fisher_loader, device)

    # 4. Fine-tuning on Target Domain (S) with EWC
    target_loader = DataLoader(sagittal, batch_size=32, shuffle=True)
    model = train_with_ewc(
        model,
        target_loader,
        ewc,
        device,
        epochs=args.ft_epochs,
        lr=args.lr,
        ewc_lambda=args.ewc_lambda,
    )

    # 5. Final Evaluation
    print("\n[Step 4] Final Evaluation (F1 Score)...")
    evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

    torch.save(model.state_dict(), "ewc_fno_final.pth")


if __name__ == "__main__":
    main()
