import torch
import torch.nn.functional as F
from torch import nn

# 模型4: 三分支网络 (Three-Branch Network)
class MB_ThreeBranch(nn.Module):
    """三分支网络：分别处理EMCal、HCal和Tracker"""
    def __init__(self):
        super(MB_ThreeBranch, self).__init__()
        
        # EMCal分支
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # HCal分支
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Tracker分支
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        
        # 分离数据
        emcal_data = x[:, 0:1, :, :]  # (B, 1, 56, 56)
        hcal_data = x[:, 1:2, :, :]   # (B, 1, 56, 56)
        tracker_data = x[:, 2:, :, :]  # (B, 2, 56, 56)
        
        # 分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 拼接融合
        fused_features = torch.cat([emcal_features, hcal_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# 模型4 V2: 三分支网络 (Three-Branch Network V2)
class MB_ThreeBranch_v2(nn.Module):
    """三分支网络V2：扩大参数量"""
    def __init__(self):
        super(MB_ThreeBranch_v2, self).__init__()
        
        # EMCal分支 - 扩大参数量
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # HCal分支 - 扩大参数量
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Tracker分支 - 扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 融合层 - 扩大参数量
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 64+64+64=192, 96->192
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        
        # 分离数据
        emcal_data = x[:, 0:1, :, :]  # (B, 1, 56, 56)
        hcal_data = x[:, 1:2, :, :]   # (B, 1, 56, 56)
        tracker_data = x[:, 2:, :, :]  # (B, 2, 56, 56)
        
        # 分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 拼接融合
        fused_features = torch.cat([emcal_features, hcal_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output
