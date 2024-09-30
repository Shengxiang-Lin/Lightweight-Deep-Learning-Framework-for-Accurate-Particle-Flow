import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Self-Attention layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Convolutional layers for generating query, key, and value matrices from the input feature maps
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # For the query matrix (Q)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)    # For the key matrix (K)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # For the value matrix (V)
        # Softmax layer to normalize the attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()  # Get the dimensions of the input tensor

        # Generate query, key, and value matrices by applying respective convolutions
        queries = self.query(x).view(batch_size, C, -1)  # Reshape to (B, C, H*W)
        keys = self.key(x).view(batch_size, C, -1)       # Reshape to (B, C, H*W)
        values = self.value(x).view(batch_size, C, -1)   # Reshape to (B, C, H*W)

        # Compute the attention scores using matrix multiplication between queries and keys
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys)  # Output: (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)  # Apply softmax to get the attention weights

        # Multiply values with attention scores and reshape back to the original size
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))  # Output: (B, C, H*W)
        return out.view(batch_size, C, H, W)  # Reshape to (B, C, H, W) without changing the original shape

# Define the CNN with Self-Attention model
class CNNattention(nn.Module):
    def __init__(self, in_channels, out_channels=1):  # Default output channels set to 1
        super(CNNattention, self).__init__()
        # Encoder part (downsampling with convolutional layers)
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # First encoding layer, input has 4 channels
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)          # Second encoding layer
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)         # Third encoding layer

        # Self-Attention layer applied after the encoder
        self.attention = SelfAttention(256)  # Self-Attention applied to 256 channels

        # Decoder part (upsampling with convolutional layers)
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # First decoding layer
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Second decoding layer
        self.dec3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Final layer, output has 1 channel
        nn.Sigmoid()  # Sigmoid activation (not used in forward)

    def forward(self, x):
        # Encoder: apply convolutional layers and relu activation
        enc1 = F.relu(self.enc1(x))  # (4, 56, 56) -> (64, 56, 56)
        enc2 = F.relu(self.enc2(F.max_pool2d(enc1, 2)))  # Downsampling: (64, 28, 28) -> (128, 28, 28)
        enc3 = F.relu(self.enc3(F.max_pool2d(enc2, 2)))  # Downsampling: (128, 14, 14) -> (256, 14, 14)

        # Self-Attention layer to refine feature maps
        attn_out = self.attention(enc3)  # Attention output: (256, 14, 14)

        # Decoder: upsample and apply convolutional layers
        dec1 = F.relu(self.dec1(F.upsample(attn_out, scale_factor=2, mode='bilinear', align_corners=False)))  # Upsample: (128, 28, 28)
        dec2 = F.relu(self.dec2(F.upsample(dec1, scale_factor=2, mode='bilinear', align_corners=False)))  # Upsample: (64, 56, 56)
        out = self.dec3(dec2)  # Final output: (64, 56, 56) -> (1, 56, 56)

        return out  # Return the final output