import torch
import torch.nn as nn
from model import HybridFNONet, FNOBlock, SpectralConv2d
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_HybridFNO(nn.Module):
    def __init__(self, num_classes=11, num_domains=3, channels=64, modes=12, dropout=0.3):
        super(DANN_HybridFNO, self).__init__()
        
        # Feature Extractor (HybridFNONet components)
        self.lifting = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fno1 = FNOBlock(channels, modes, residual_scale=0.5)
        self.fno2 = FNOBlock(channels, modes, residual_scale=0.5)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.feature_expand = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
        )

        # Class Classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, num_classes),
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, num_domains),
        )

    def forward(self, x, alpha=1.0):
        # CoordConv logic
        batch_size, _, h, w = x.size()
        grid_h = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w).to(x.device)
        grid_w = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w).to(x.device)
        x = torch.cat([x, grid_h, grid_w], dim=1)

        # Feature Extraction
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = self.feature_expand(x)
        feature = x.mean(dim=(2, 3)) # GAP

        # Class Prediction
        class_output = self.class_classifier(feature)

        # Domain Prediction with GRL
        reverse_feature = GradientReversalLayer.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

