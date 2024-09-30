import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Self-Attention layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Define the layers for query, key, and value
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 1x1 convolution to generate queries
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)    # 1x1 convolution to generate keys
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 1x1 convolution to generate values
        self.softmax = nn.Softmax(dim=-1)  # Softmax for attention scores

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Generate queries, keys, and values
        queries = self.query(x).view(batch_size, C, -1)  # (B, C, H*W)
        keys = self.key(x).view(batch_size, C, -1)      # (B, C, H*W)
        values = self.value(x).view(batch_size, C, -1)  # (B, C, H*W)

        # Compute attention scores
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys)  # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)  # Apply softmax

        # Compute the attention output
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))  # (B, C, H*W)
        return out.view(batch_size, C, H, W)  # Reshape to original dimensions

# Define the U-Net with Transformer integration
class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels=1):  # Output channels set to 1
        super(UNetTransformer, self).__init__()
        # Encoder part
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # Input has 4 channels
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Self-attention layer
        self.attention = SelfAttention(256)  # Input has 256 channels
        # Decoder part
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # Adjust number of channels after concatenation
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Final output layer
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)  # Adjust number of channels after concatenation

    def forward(self, x):
        # Encoding
        enc1 = F.relu(self.enc1(x))  # 4x56x56 -> 64x56x56
        enc2 = F.relu(self.enc2(F.max_pool2d(enc1, 2)))  # 64x28x28 -> 128x28x28
        enc3 = F.relu(self.enc3(F.max_pool2d(enc2, 2)))  # 128x14x14 -> 256x14x14

        # Self-attention
        attn_out = self.attention(enc3)  # 256x14x14 -> 256x14x14

        # Decoding
        dec1 = F.relu(self.dec1(F.upsample(attn_out, scale_factor=2, mode='bilinear', align_corners=False)))  # 256x14x14 -> 128x28x28
        dec1 = torch.cat([dec1, enc2], dim=1)  # Concatenate with corresponding encoder output
        dec1 = self.conv1(dec1)  # Adjust channels after concatenation
        dec2 = F.relu(self.dec2(F.upsample(dec1, scale_factor=2, mode='bilinear', align_corners=False)))  # 128x28x28 -> 64x56x56
        dec2 = torch.cat([dec2, enc1], dim=1)  # Concatenate with corresponding encoder output
        dec2 = self.conv2(dec2)  # Adjust channels after concatenation
        out = self.dec3(dec2)  # 64x56x56 -> 1x56x56
        
        return out  # Output the final result