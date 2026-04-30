import os, math, copy, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

# model.py에서 필요한 함수들을 임포트합니다.
from model import build_model, grad_reverse


class MedDataset(Dataset):
    CLASS_NAMES = [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-right",
        "lung-left",
        "pancreas",
        "spleen",
    ]

    def __init__(self, img_dir, csv_path, domain_lbl, augment=False):
        self.domain_lbl = domain_lbl
        self.augment = augment
        raw = pd.read_csv(csv_path, index_col=0, dtype=str)
        raw = raw[raw.index != "Index"]
        self.lbls = raw[self.CLASS_NAMES].astype(int).values.argmax(axis=1)
        self.paths = [os.path.join(img_dir, f + ".png") for f in raw.index]
        self.geo_aug = T.RandomApply([T.RandomAffine(15, (0.1, 0.1), shear=10)], p=0.6)

    def __len__(self):
        return len(self.lbls)

    def __getitem__(self, idx):
        img = read_image(self.paths[idx], ImageReadMode.GRAY).float()
        img = F.interpolate(img.unsqueeze(0), (28, 28), mode="bilinear").squeeze(0)
        img = (img / (img.max() + 1e-6) - 0.5) / 0.5
        if self.augment:
            img = self.geo_aug(img)
        return img.clamp(-1.0, 1.0), self.lbls[idx], self.domain_lbl


class MedSubset(Dataset):
    def __init__(self, base, indices, augment=False):
        self.base, self.indices, self.augment = (
            base,
            torch.as_tensor(indices, dtype=torch.long),
            augment,
        )
        self.lbls = base.lbls[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, lbl, dom = self.base[int(self.indices[idx].item())]
        return img, lbl, dom


def stratified_split_indices(labels, val_per_class=10, seed=42):
    rng = np.random.default_rng(seed)
    t_idx, v_idx = [], []
    for cls in range(11):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = min(val_per_class, max(1, len(idx) // 5))
        v_idx.extend(idx[:n_val].tolist())
        t_idx.extend(idx[n_val:].tolist())
    return t_idx, v_idx


# [Step 1] DANN 사전 학습: 뷰-불변 특징을 강제합니다.[span_11](start_span)[span_11](end_span)
def pretrain_dann(model, axial_ds, coronal_ds, device, epochs, lr):
    print("\n[Step 1] DANN Pre-training on Axial + Coronal (OneCycleLR)...")
    combined = torch.utils.data.ConcatDataset([axial_ds, coronal_ds])
    loader = DataLoader(combined, 64, True, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, steps_per_epoch=len(loader), epochs=epochs
    )
    c_org, c_dom = nn.CrossEntropyLoss(label_smoothing=0.1), nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        total_loss = 0
        for i, l, d in loader:
            i, l, d = i.to(device), l.to(device), d.to(device)
            opt.zero_grad()
            feat = model.get_feature(i)
            alpha = 2.0 / (1.0 + math.exp(-10 * (e / epochs))) - 1
            loss = c_org(model.classifier(feat), l) + c_dom(
                model.domain_classifier(grad_reverse(feat, alpha)), d
            )
            loss.backward()
            opt.step()
            sch.step()
            total_loss += loss.item()
        if (e + 1) % 10 == 0:
            print(f" Epoch {e+1:3d}: Loss {total_loss/len(loader):.4f}")
    return model


# [Step 2] Fisher Matrix 계산: 중요 파라미터를 식별합니다.[span_12](start_span)[span_12](end_span)
def compute_fisher(model, ds, device):
    print("\n[Step 2] Computing Fisher Matrix for EWC...")
    loader = DataLoader(ds, 32)
    model.eval()
    fisher = {
        n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
    }
    for i, l, _ in loader:
        model.zero_grad()
        nn.CrossEntropyLoss()(model(i.to(device)), l.to(device)).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().pow(2) / len(loader)
    return fisher


# [Step 3] Exact Proto-Init: 퓨샷 데이터에 즉각 적응합니다.[span_13](start_span)[span_13](end_span)
def exact_proto_init(model, ds, device):
    print("\n[Step 3] Proto-Init (Metric-Based Adaptation)...")
    loader = DataLoader(ds, 32)
    model.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for i, l, _ in loader:
            feats.append(model.get_feature(i.to(device)))
            lbls.append(l.to(device))
    feats, lbls = torch.cat(feats), torch.cat(lbls)
    protos = torch.stack(
        [
            (
                feats[lbls == c].mean(0)
                if (lbls == c).any()
                else torch.zeros(feats.size(1), device=device)
            )
            for c in range(11)
        ]
    )
    model.classifier.weight.data, model.classifier.bias.data = 2.0 * protos, -(
        protos**2
    ).sum(1)
    return model


# [Step 4] EWC Fine-tuning: 소스 지식을 보존하며 타겟을 학습합니다.[span_14](start_span)[span_14](end_span)
def finetune_ewc(model, ds, fisher, theta, device, epochs, lr, lam, val_datasets=None):
    print(f"\n[Step 4] Fine-tuning with EWC (λ={lam})...")
    loader = DataLoader(ds, 32, True)
    opt = torch.optim.AdamW(model.parameters(), lr)
    best_f1, best_state = 0, copy.deepcopy(model.state_dict())

    for e in range(epochs):
        model.train()
        for i, l, _ in loader:
            i, l = i.to(device), l.to(device)
            opt.zero_grad()
            ce = nn.CrossEntropyLoss(label_smoothing=0.1)(model(i), l)
            ewc = sum(
                [
                    (fisher[n] * (p - theta[n]).pow(2)).sum()
                    for n, p in model.named_parameters()
                    if n in fisher
                ]
            )
            (ce + (lam / 2) * ewc).backward()
            opt.step()

        if val_datasets and (e + 1) % 5 == 0:
            scores = evaluate_all(model, *val_datasets, device, False)
            if scores["Final"] > best_f1:
                best_f1 = scores["Final"]
                best_state = copy.deepcopy(model.state_dict())

    if val_datasets:
        model.load_state_dict(best_state)
    return model


def evaluate_all(model, axial_ds, coronal_ds, sagittal_ds, device, verbose=True):
    from sklearn.metrics import f1_score

    def eval_domain(ds):
        loader = DataLoader(ds, 128)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i, l, _ in loader:
                preds.extend(model(i.to(device)).argmax(1).cpu().numpy())
                trues.extend(l.numpy())
        return float(f1_score(trues, preds, average="macro", zero_division=0))

    f1_a, f1_c, f1_s = (
        eval_domain(axial_ds),
        eval_domain(coronal_ds),
        eval_domain(sagittal_ds),
    )
    final = 0.7 * f1_s + 0.3 * (f1_a + f1_c) / 2.0
    if verbose:
        print(
            f" Axial: {f1_a:.4f} | Coronal: {f1_c:.4f} | Sagittal: {f1_s:.4f} | Final: {final:.4f}"
        )
    return {"Final": final, "A": f1_a, "C": f1_c, "S": f1_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--save_path", default="./model.pth")
    parser.add_argument("--lambda_ewc", type=float, default=200.0)
    parser.add_argument("--search_lambda", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = args.train_dir
    A_train = MedDataset(
        os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), 0, True
    )
    C_train = MedDataset(
        os.path.join(args.train_dir, "coronal"),
        os.path.join(args.train_dir, "label/coronal.csv"),
        1,
        True,
    )
    S_train = MedDataset(
        os.path.join(args.train_dir, "sagittal"),
        os.path.join(args.train_dir, "label/sagittal.csv"),
        2,
        True,
    )
    A_val = MedDataset(os.path.join(t, "axial"), os.path.join(t, "label/axial.csv"), 0)
    C_val = MedDataset(
        os.path.join(t, "coronal"), os.path.join(t, "label/coronal.csv"), 1
    )
    S_val = MedDataset(
        os.path.join(t, "sagittal"), os.path.join(t, "label/sagittal.csv"), 2
    )

    model = pretrain_dann(build_model(), A_train, C_train, device, 100, 1e-3)
    fisher = compute_fisher(
        model, torch.utils.data.ConcatDataset([A_train, C_train]), device
    )
    theta_meta = {n: p.detach().clone() for n, p in model.named_parameters()}

    lam = args.lambda_ewc
    if args.search_lambda:
        print("\n[Searching for Best λ...]")
        t_idx, v_idx = stratified_split_indices(S_val.lbls)
        for cand in [100.0, 200.0, 500.0]:
            tmp_model = copy.deepcopy(model)
            tmp_model = exact_proto_init(
                tmp_model, MedSubset(S_val, t_idx, True), device
            )
            tmp_model = finetune_ewc(
                tmp_model,
                MedSubset(S_val, t_idx, True),
                fisher,
                theta_meta,
                device,
                30,
                2e-4,
                cand,
                (A_val, C_val, MedSubset(S_val, v_idx)),
            )
            res = evaluate_all(
                tmp_model, A_val, C_val, MedSubset(S_val, v_idx), device, False
            )
            print(f" Lambda {cand}: S-Val Final {res['Final']:.4f}")

    model = exact_proto_init(model, S_train, device)
    model = finetune_ewc(model, S_train, fisher, theta_meta, device, 60, 2e-4, lam)
    evaluate_all(model, A_val, C_val, S_val, device)
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Saved to {args.save_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
