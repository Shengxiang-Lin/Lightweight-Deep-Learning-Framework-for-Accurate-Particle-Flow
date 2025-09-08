import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, d_model=16, nhead=1):
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels

        # 降维减少计算量
        self.conv_down = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.conv_up = nn.Conv2d(d_model, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 降维 (B, d_model, H, W)
        x_down = self.conv_down(x)

        # 将特征展开为序列 (B, H*W, d_model)
        x_seq = x_down.reshape(B, -1, x_down.shape[1])  # �� 使用 reshape 确保兼容非连续张量

        # 自注意力计算
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)

        # 将序列恢复为特征图 (B, d_model, H, W)
        attn_feature = attn_output.reshape(B, -1, H, W)  # �� 也改为 reshape

        # 映射回原通道数 (B, C, H, W)
        out = self.conv_up(attn_feature)
        return out

    


class CNN_with_Attention(nn.Module):
    def __init__(self):
        super(CNN_with_Attention, self).__init__()

        # 降低通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # 32 → 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 64 → 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 128 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.attention_block = SelfAttentionBlock(in_channels=64, d_model=16, nhead=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention_block(x)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 16  # 32 → 16
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 7x7 → 3x3
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 仅保留 layer1，其他层减少
        self.layer1 = self._make_layer(16, blocks=1, stride=1)  # 32 → 16, 仅1个块
        self.layer2 = self._make_layer(32, blocks=0, stride=1)  # 64 → 32
        self.layer3 = self._make_layer(64, blocks=0, stride=1)  # 128 → 64
        self.layer4 = self._make_layer(128, blocks=0, stride=1)  # 256 → 128

        num_classes = 1
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        if blocks > 0:
            layers.append(self.ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResNet.ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)  # 3x3 → 1x1
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)  # 3x3 → 1x1
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
            return out

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


# 测试优化后的模型
if __name__ == "__main__":
    model = ResNet()
    x = torch.randn(4, 4, 56, 56)
    y = model(x)
    print("ResNet Output shape:", y.shape)  # (4, 1, 56, 56)

    model_cnn_att = CNN_with_Attention()
    y2 = model_cnn_att(x)
    print("CNN_with_Attention Output shape:", y2.shape)  # (4, 1, 56, 56)

