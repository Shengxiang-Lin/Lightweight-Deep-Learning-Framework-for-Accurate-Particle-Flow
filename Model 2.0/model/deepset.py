import torch
import torch.nn.functional as F
from torch import nn

'''
输入形状：(batch_size, num_channels, height, width)->(batch_size, set_number, features)->(batch_size, set_number, energy)
'''

import torch
import torch.nn as nn
from typing import Tuple, Optional

def images_to_set6(
    x: torch.Tensor,
    pixel_size: Tuple[float, float] = (1.0, 1.0),
    origin_xy: Tuple[float, float] = (0.0, 0.0)
) -> torch.Tensor:
    """
    x: (B, 4, H, W)  通道顺序: [emcal, hcal, trkn, trkp]
    return:
      feats: (B, N, 6)  特征顺序: [x, y, emcal, hcal, trkn, trkp]
    """
    assert x.dim() == 4 and x.size(1) == 4, "x 应为 (B,4,H,W)"
    B, _, H, W = x.shape
    device = x.device

    # 构造网格坐标 (x,y) —— 以像素中心为坐标
    xs = origin_xy[0] + (torch.arange(W, device=device) + 0.5) * pixel_size[0]
    ys = origin_xy[1] + (torch.arange(H, device=device) + 0.5) * pixel_size[1]
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    Xb = X.view(1, 1, H, W).expand(B, 1, H, W)
    Yb = Y.view(1, 1, H, W).expand(B, 1, H, W)

    emcal = x[:, 0:1]
    hcal  = x[:, 1:2]
    trkn  = x[:, 2:3]
    trkp  = x[:, 3:4]

    feats6 = torch.cat([Xb, Yb, emcal, hcal, trkn, trkp], dim=1)     # (B,6,H,W)
    feats6 = feats6.permute(0, 2, 3, 1).reshape(B, H * W, 6)         # (B,N,6)
    return feats6

class DeepSetPELayer(nn.Module):
    """
    PE DeepSet 层: f(X) = tanh( gamma(X) - lambda(mean(X)) )
    X: (B,N,in_dim) → (B,N,out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, agg: str = "mean"):
        super().__init__()
        self.gamma   = nn.Linear(in_dim, out_dim)
        self.lambda_ = nn.Linear(in_dim, out_dim)
        self.act     = nn.Tanh()
        assert agg in ["mean", "sum"]
        self.agg = agg

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pooled = X.mean(dim=1, keepdim=True) if self.agg == "mean" else X.sum(dim=1, keepdim=True)
        return self.act(self.gamma(X) - self.lambda_(pooled))

class DeepSetPEModel(nn.Module):
    """
    4 层 DS: 6→6→12→8→4；concat(6+12+8+4=30) → per-element MLP → (B,N)
    """
    def __init__(
        self,
        in_dim: int = 6,
        layer_dims = (6, 12, 8, 4),
        readout_hidden = (64, 32),
        out_activation: Optional[str] = None,  # None | 'sigmoid' | 'relu'
    ):
        super().__init__()
        dims = [in_dim] + list(layer_dims)
        self.ds_layers = nn.ModuleList([
            DeepSetPELayer(dims[i], dims[i+1]) for i in range(len(layer_dims))
        ])
        concat_dim = sum(layer_dims)  # 30
        mlp = []
        last = concat_dim
        for h in readout_hidden:
            mlp += [nn.Linear(last, h), nn.ReLU()]
            last = h
        mlp += [nn.Linear(last, 1)]
        self.readout = nn.Sequential(*mlp)

        if out_activation is None:
            self.out_act = None
        elif out_activation.lower() == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_activation.lower() == "relu":
            self.out_act = nn.ReLU()
        else:
            raise ValueError("out_activation 仅支持 None/'sigmoid'/'relu'")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B,N,6)
        return: (B,N)
        """
        outs = []
        H = X
        for layer in self.ds_layers:
            H = layer(H)      # (B,N,d_i)
            outs.append(H)
        H_cat = torch.cat(outs, dim=-1)         # (B,N,30)
        y = self.readout(H_cat).squeeze(-1)     # (B,N)
        if self.out_act is not None:
            y = self.out_act(y)
        return y


class DSReconstruction(nn.Module):
    """
    把预处理 + DeepSet 合在一起：
      输入  (B,4,H,W)
      输出  (B,1,H,W)   —— 与 truth 直接对齐，无需形变 truth
    """
    def __init__(
        self,
        H: int = 56,
        W: int = 56,
        pixel_size: Tuple[float,float] = (2.0/55, 2.0/55),
        origin_xy: Tuple[float,float] = (-1.0, -1.0),
        out_activation: Optional[str] = None,   # None / 'sigmoid' / 'relu'
    ):
        super().__init__()
        self.H, self.W = H, W
        self.pixel_size = pixel_size
        self.origin_xy = origin_xy
        self.backbone = DeepSetPEModel(
            in_dim=6, layer_dims=(6,12,8,4),
            readout_hidden=(64,32),
            out_activation=out_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,4,H,W)
        return: (B,1,H,W)
        """
        B, _, H, W = x.shape
        assert H == self.H and W == self.W, "尺寸需与初始化一致"
        feats = images_to_set6(x, self.pixel_size, self.origin_xy)  # (B,N,6)
        y_flat = self.backbone(feats)                                # (B,N)
        y_map  = y_flat.reshape(B, self.H, self.W).unsqueeze(1)      # (B,1,H,W)
        return y_map
