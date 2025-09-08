import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import torch
from typing import Tuple

def preprocess_gnn_inputs(
    x: torch.Tensor,
    R: float,
    eta_range: Tuple[float, float] = (-2.5, 2.5),
    phi_range: Tuple[float, float] = (-math.pi, math.pi),
    eps: float = 1e-6,
):
    """
    参数
    ----
    x : (B, 4, H, W)  输入通道: [emcal, hcal, trkn, trkp]
    R : 圆筒半径
    eta_range : (eta_min, eta_max)，沿宽度W方向线性分布
    phi_range : (phi_min, phi_max)，沿高度H方向线性分布
    eps : 防止 tan(eta)=0 时分母为0

    返回
    ----
    pos_feats : (B, N, 4)  每节点 [x, y, z, track_position]
    ene_feats : (B, N, 4)  每节点 [emcal, hcal, trkn, trkp]
    """
    assert x.dim() == 4 and x.size(1) == 4, "x 形状应为 (B,4,H,W)，通道=[emcal,hcal,trkn,trkp]"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    N = H * W

    # 1) 构造 (eta, phi) 网格：eta 沿宽度 W，phi 沿高度 H
    eta_min, eta_max = eta_range
    phi_min, phi_max = phi_range
    etas = torch.linspace(eta_min, eta_max, steps=W, device=device, dtype=dtype)  # (W,)
    phis = torch.linspace(phi_min, phi_max, steps=H, device=device, dtype=dtype)  # (H,)
    PHI, ETA = torch.meshgrid(phis, etas, indexing="ij")  # (H,W), (H,W)

    # 2) 圆柱面坐标映射
    #    x = R * sin(phi),  y = R / tan(eta),  z = R * cos(phi)
    tan_eta = torch.tan(ETA)

    Xcoord = R * torch.sin(PHI)         # (H,W)
    Ycoord = R * tan_eta                # (H,W)
    Zcoord = R * torch.cos(PHI)         # (H,W)

    # 扩展到 batch 并与 emcal 对齐
    Xb = Xcoord.view(1, 1, H, W).expand(B, 1, H, W)  # (B,1,H,W)
    Yb = Ycoord.view(1, 1, H, W).expand(B, 1, H, W)
    Zb = Zcoord.view(1, 1, H, W).expand(B, 1, H, W)

    # 3) track position 特征：emcal != 0 → 1，否则 0
    emcal = x[:, 0:1, :, :]                            # (B,1,H,W)
    track_pos = (emcal != 0).to(dtype)                 # (B,1,H,W)

    # 4) 位置分支：拼接 [x, y, z, track_position] → (B,N,4)
    pos_feats = torch.cat([Xb, Yb, Zb, track_pos], dim=1)          # (B,4,H,W)
    pos_feats = pos_feats.permute(0, 2, 3, 1).reshape(B, N, 4)     # (B,N,4)

    # 5) 能量分支：直接展平输入通道 → (B,N,4)
    ene_feats = x.permute(0, 2, 3, 1).reshape(B, N, 4)             # (B,N,4)

    return pos_feats, ene_feats
# ====== KNN & 工具 ======
def knn_idx(feat_pos: torch.Tensor, k: int):
    """feat_pos: (B,N,C) -> idx: (B,N,K)  基于位置分支特征做 KNN"""
    B, N, _ = feat_pos.shape
    k_eff = min(k, N - 1)
    # 距离: (B,N,N)，屏蔽自环
    dist = torch.cdist(feat_pos, feat_pos, p=2)
    eye = torch.eye(N, device=feat_pos.device, dtype=feat_pos.dtype)[None]
    dist = dist + eye * 1e9
    idx = dist.topk(k=k_eff, largest=False, dim=-1).indices  # (B,N,k_eff)
    return idx

def gather_neighbors(x: torch.Tensor, idx: torch.Tensor):
    """x: (B,N,C), idx: (B,N,K) -> (B,N,K,C)"""
    B, N, C = x.shape
    K = idx.size(-1)
    # 利用 batch 维度安全 gather
    gather_index = idx[..., None].expand(B, N, K, C)
    x_expanded = x[:, None, :, :].expand(B, N, N, C)
    return torch.gather(x_expanded, 2, gather_index)

# ====== 小型 MLP ======
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(), bn=True, act=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if bn: layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (..., Cin)
        s = x.shape
        x = x.reshape(-1, s[-1])
        y = self.net(x)
        return y.reshape(*s[:-1], -1)

# ====== 单层 EdgeConv（双分支） ======
class DualEdgeConv(nn.Module):
    """
    位置分支: x^{l+1}_i = max_j Θx(x_j - x_i) + Φx(x_i)
    能量分支: e^{l+1}_i = mean_j Θe(e_j - e_i) + Φe(e_i)
    """
    def __init__(self, pos_in, pos_out, ene_in, ene_out, k=10):
        super().__init__()
        self.k = k
        self.theta_x = MLP(pos_in, pos_out, hidden=(pos_out,))
        self.phi_x   = MLP(pos_in, pos_out, hidden=())
        self.theta_e = MLP(ene_in, ene_out, hidden=())
        self.phi_e   = MLP(ene_in, ene_out, hidden=())

    def forward(self, pos_feat, ene_feat):
        # KNN 只看位置分支
        idx = knn_idx(pos_feat, self.k)                # (B,N,K)
        # 位置分支
        x_i = pos_feat                                 # (B,N,Cx)
        x_j = gather_neighbors(pos_feat, idx)          # (B,N,K,Cx)
        dx  = x_j - x_i.unsqueeze(2)                   # (B,N,K,Cx)
        hx  = self.theta_x(dx).max(dim=2).values       # (B,N,Cx_out)
        skip_x = self.phi_x(x_i)
        pos_next = hx + skip_x
        # 能量分支
        e_i = ene_feat
        e_j = gather_neighbors(ene_feat, idx)          # (B,N,K,Ce)
        de  = e_j - e_i.unsqueeze(2)
        he  = self.theta_e(de).mean(dim=2)             # (B,N,Ce_out)
        skip_e = self.phi_e(e_i)
        ene_next = he + skip_e
        return pos_next, ene_next

# ====== 5层 DGCNN 主干 ======
class DGCNN5DualBackbone(nn.Module):
    """
    位置分支通道: (4,32,64,128,64,3)
    能量分支通道: (4,4,4,4,4,4)
    forward 输入: pos_feats (B,N,4), ene_feats (B,N,4)
    forward 输出: per-layer 特征拼接 (B,N, 319)
    """
    def __init__(self, pos_dims=(4,32,64,128,64,3), ene_dims=(4,4,4,4,4,4), k=10):
        super().__init__()
        assert len(pos_dims) == 6 and len(ene_dims) == 6
        self.blocks = nn.ModuleList([
            DualEdgeConv(pos_dims[i], pos_dims[i+1], ene_dims[i], ene_dims[i+1], k=k)
            for i in range(5)
        ])
        self.pos_dims = pos_dims
        self.ene_dims = ene_dims
        self.concat_dim = sum(pos_dims) + sum(ene_dims)  # 319

    def forward(self, pos_feats, ene_feats):
        pos_all = [pos_feats]     # l0
        ene_all = [ene_feats]     # l0
        p, e = pos_feats, ene_feats
        for blk in self.blocks:
            p, e = blk(p, e)
            pos_all.append(p)
            ene_all.append(e)
        # 拼接所有层（含 l0 到 l5）
        pos_cat = torch.cat(pos_all, dim=-1)  # (B,N, sum pos_dims)
        ene_cat = torch.cat(ene_all, dim=-1)  # (B,N, sum ene_dims)
        feat_cat = torch.cat([pos_cat, ene_cat], dim=-1)  # (B,N,319)
        return feat_cat

# ====== per-node 读出 MLP + 还原到 (B,1,H,W) ======
class NodeEnergyHead(nn.Module):
    def __init__(self, in_dim=319, hidden=(128,64), out_act="relu"):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, x):  # (B,N,319) -> (B,N,1)
        y = self.net(x)
        if self.out_act == "relu":
            y = F.relu(y)
        elif self.out_act == "sigmoid":
            y = torch.sigmoid(y)
        return y

# ====== 整体模型：x -> 预处理 -> DGCNN -> (B,1,H,W) ======
class DGCNNModel(nn.Module):
    def __init__(self, preprocess_fn, R, eta_range=(-2.5, 2.5), phi_range=(-3.14159265, 3.14159265),
                 k=10, readout_hidden=(128,64), out_act="relu"):
        super().__init__()
        self.preprocess_fn = preprocess_fn
        self.R = R
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.backbone = DGCNN5DualBackbone(k=k)
        self.head = NodeEnergyHead(in_dim=self.backbone.concat_dim,
                                   hidden=readout_hidden,
                                   out_act=out_act)
        # 跳跃连接参数
        self.skip_weights = nn.Parameter(torch.ones(4))  # 初始化为1，也可用其他值

    def forward(self, x):
        """
        x: (B,4,H,W)  通道: [emcal, hcal, trkn, trkp]
        return: (B,1,H,W)
        """
        B, _, H, W = x.shape
        pos_feats, ene_feats = self.preprocess_fn(
            x, R=self.R, eta_range=self.eta_range, phi_range=self.phi_range
        )  # pos:(B,N,4), ene:(B,N,4)
        feat_cat = self.backbone(pos_feats, ene_feats)        # (B,N,319)
        y_flat = self.head(feat_cat)                          # (B,N,1)
        y = y_flat.view(B, H, W, 1).permute(0, 3, 1, 2)       # -> (B,1,H,W)

        # 跳跃连接部分
        # x: (B,4,H,W), skip_weights: (4,)
        skip = (x * self.skip_weights.view(1, 4, 1, 1)).sum(dim=1, keepdim=True)  # (B,1,H,W)
        y = y + skip

        return y