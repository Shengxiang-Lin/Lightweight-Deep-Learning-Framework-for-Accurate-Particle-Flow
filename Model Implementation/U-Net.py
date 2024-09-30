import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolutional block with two convolutional layers, batch normalization, dropout, and LeakyReLU activation
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            # First 3x3 convolution layer with batch normalization, dropout, and activation
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1),  # 3x3 convolution, padding=1 keeps size unchanged
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
            # Second 3x3 convolution layer with batch normalization, dropout, and activation
            nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1),  # Again, keep size unchanged
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

# Downsampling block to reduce the feature map size by half using a 2x2 convolution with stride 2
class DownSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 2x2 convolution with stride 2 reduces the spatial dimensions
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2 convolution, stride=2 reduces size by half
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)

# Upsampling block to increase the feature map size by a factor of 2 using nearest neighbor interpolation
class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        # 1x1 convolution to adjust the number of channels after upsampling
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 1x1 convolution to change the number of channels

    def forward(self, x, r):
        # Upsample using nearest neighbor interpolation
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # Use nearest neighbor interpolation to upsample
        x = self.Up(up)  # Change the number of output channels
        x = torch.cat([x, r], dim=1)  # Concatenate the upsampled feature map with the skip connection
        return x

# U-Net architecture for image segmentation or similar tasks
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Initial convolution block with 4 input channels (for example, emcal, hcal, trkn, trkp)
        self.C1 = Conv(4, 64)  # First convolution block, input has 4 channels
        # First downsampling layer to reduce spatial size
        self.D1 = DownSampling(64, 128)  # First downsampling layer
        # Second convolution block
        self.C2 = Conv(128, 128)  # Second convolution block
        # Skipping a second downsampling and upsampling for simplicity in this version
        # Final upsampling layer to increase the size back and perform concatenation with the previous layer
        self.U2 = UpSampling(128, 64)  # Second upsampling layer
        # Final convolution block after concatenation
        self.C5 = Conv(128, 64)  # Final convolution block
        # Prediction layer that outputs a single channel
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output layer, predicts 1 channel
        # Apply sigmoid activation for final output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def forward(self, x):
        # Encoder part: First convolution and downsampling
        R1 = self.C1(x)  # First convolution block
        R2 = self.C2(self.D1(R1))  # Downsampling followed by the second convolution block
        
        # Decoder part: Upsample and concatenate with previous layer (skip connection)
        up2 = self.U2(R2, R1)  # Second upsampling with skip connection to the first convolution block
        
        # Final convolution and output
        c = self.C5(up2)  # Final convolution block
        return self.sigmoid(self.pred(c))  # Output prediction with sigmoid activation