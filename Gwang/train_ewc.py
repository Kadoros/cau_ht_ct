import os
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# [핵심] 우리가 최적화한 최신 모델과 검증용 유틸리티들 임포트
from model_san1 import build_model
from train_san2 import (
    load_domain,
    evaluate_all,
    stratified_split_indices,
    MedSubset,
    pretrain,
)
from model_ewc import EWC


def train_ewc_with_eval(
    model, train_loader, val_datasets, ewc, device, epochs=50, lr=1e-4, ewc_lambda=1000
):
    """검증 로직(Early Stopping)이 추가된 EWC 파인튜닝 함수"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n[Step 3] Fine-tuning on Sagittal with EWC (lambda={ewc_lambda})...")

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(imgs)
            ce_loss = criterion(output, lbls)

            # EWC Penalty
            ewc_loss = ewc.penalty(model)
            loss = ce_loss + ewc_lambda * ewc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(lbls)
            correct += (output.argmax(1) == lbls).sum().item()
            total += len(lbls)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1) * 100.0

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=150)
    parser.add_argument("--ft_epochs", type=int, default=80)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)  # EWC 강도
    parser.add_argument("--lr", type=float, default=5e-5)
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

    # [핵심] 최적화된 채널 64, 모드 12 모델 생성
    model = build_model(num_classes=11, channels=64, modes=12).to(device)

    # 2. Pre-training
    model = pretrain(
        model, axial, coronal, device, epochs=args.pretrain_epochs, lr=1e-3
    )
    pretrained_state = copy.deepcopy(model.state_dict())

    # 3. Fisher Information Matrix 계산
    print("\n[Step 2] Calculating Fisher Information Matrix for Source Domains...")
    fisher_loader = DataLoader(
        ConcatDataset([axial_eval, coronal_eval]),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    ewc = EWC(model, fisher_loader, device)

    # 4. [핵심] Sagittal 데이터를 80(Train) : 20(Val) 로 분할하여 최적의 Epoch 찾기
    print("\n[Step 3] Internal Validation to find best Epoch...")
    train_idx, val_idx = stratified_split_indices(
        sagittal_eval.labels, val_per_class=10, seed=42
    )
    sagittal_train_split = MedSubset(sagittal_eval, train_idx, augment=True)
    sagittal_val_split = MedSubset(sagittal_eval, val_idx, augment=False)

    val_loader = DataLoader(
        sagittal_train_split, batch_size=32, shuffle=True, num_workers=4
    )

    # 20%로 평가하며 최고 성능 지점 찾기
    _, best_epoch = train_ewc_with_eval(
        model,
        val_loader,
        (axial_eval, coronal_eval, sagittal_val_split),
        ewc,
        device,
        epochs=args.ft_epochs,
        lr=args.lr,
        ewc_lambda=args.ewc_lambda,
    )

    # 5. [핵심] 찾은 최적의 Epoch로 전체 Sagittal 데이터 100% 최종 학습 (오버피팅 방지)
    print(f"\n[Final Training] Fine-tune on ALL Sagittal shots for {best_epoch} Epochs")
    model.load_state_dict(pretrained_state)  # 파인튜닝 전으로 롤백
    full_loader = DataLoader(sagittal, batch_size=32, shuffle=True, num_workers=4)

    model, _ = train_ewc_with_eval(
        model,
        full_loader,
        None,
        ewc,
        device,
        epochs=best_epoch,
        lr=args.lr,
        ewc_lambda=args.ewc_lambda,
    )

    # 6. 최종 평가 및 저장
    print("\n[Step 4] Final Evaluation (F1 Score)...")
    evaluate_all(model, axial_eval, coronal_eval, sagittal_eval, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"num_classes": 11, "channels": 64, "modes": 12},
        },
        "./ewc_fno_final.pth",
    )
    print("\nModel saved to: ./ewc_fno_final.pth")


if __name__ == "__main__":
    main()
