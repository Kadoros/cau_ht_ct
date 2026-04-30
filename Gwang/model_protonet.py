import torch
import torch.nn as nn
from model import FNOBlock
import torch.nn.functional as F

class ProtoNet_FNO(nn.Module):
    def __init__(self, channels=64, modes=12):
        super(ProtoNet_FNO, self).__init__()
        
        # Feature Extractor (Backbone)
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

    def forward(self, x):
        # CoordConv logic
        batch_size, _, h, w = x.size()
        grid_h = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w).to(x.device)
        grid_w = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w).to(x.device)
        x = torch.cat([x, grid_h, grid_w], dim=1)

        # Feature Extraction to Embedding Space
        x = self.lifting(x)
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.spatial_refine(x)
        x = self.feature_expand(x)
        embedding = x.mean(dim=(2, 3)) # Global Average Pooling
        return embedding

def euclidean_dist(x, y):
    # x: [N, D] (Query embeddings)
    # y: [M, D] (Prototypes)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
