import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC(object):
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        # 최적 가중치 저장 (Source Domain 학습 후)
        self.params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._fisher_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for imgs, lbls in self.dataloader:
            self.model.zero_grad()
            imgs = imgs.to(self.device)
            # FIM 계산을 위해 log-likelihood의 gradient 사용
            output = self.model(imgs)
            # 여기서는 CrossEntropy를 사용하여 log-likelihood를 근사
            loss = F.cross_entropy(output, lbls.to(self.device))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data**2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._fisher_matrices[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss
