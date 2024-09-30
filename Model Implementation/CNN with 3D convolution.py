import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the 3D CNN model
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        # Define three 3D convolutional layers with different kernel sizes
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))  # 3x3x3 kernel (3D)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))  # 3x5x5 kernel
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))  # 3x7x7 kernel

        # Encoder part (2D convolution, batch normalization, and ReLU layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, padding=2),  # First 2D convolution (input 12 channels)
            nn.BatchNorm2d(32),                          # Batch normalization for stable training
            nn.ReLU(inplace=True),                       # ReLU activation

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # Second 2D convolution
            nn.BatchNorm2d(64),                          # Batch normalization
            nn.ReLU(inplace=True)                        # ReLU activation
        )

        # Decoder part (two 2D convolutional layers to reconstruct output)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2), # First decoder layer
            nn.Conv2d(16, 1, kernel_size=5, padding=2),  # Second decoder layer (output is 1 channel)
            nn.Sigmoid()                                 # Sigmoid activation to normalize output
        )

    def forward(self, x):  # Input x has shape (batch_size, 4, 56, 56) [channels: emcal, hcal, trkn, trkp]
        x = x.unsqueeze(1)  # Add an extra dimension to x (batch_size, 1, 4, 56, 56) for 3D convolutions

        # Select subsets of the input data for 3D convolutions
        x_e_h_n = x[:, :, :3, :, :]  # Select channels: emcal, hcal, trkn (first 3 channels)
        x_e_h_p = x[:, :, [0, 1, 3], :, :]  # Select channels: emcal, hcal, trkp (channels 0, 1, 3)

        # Apply 3D convolutions on these subsets
        x2 = self.conv3x3x3(x_e_h_n)  # Apply 3x3x3 convolution on emcal, hcal, trkn
        x3 = self.conv3x5x5(x_e_h_n)  # Apply 3x5x5 convolution on emcal, hcal, trkn
        x4 = self.conv3x7x7(x_e_h_n)  # Apply 3x7x7 convolution on emcal, hcal, trkn

        x5 = self.conv3x3x3(x_e_h_p)  # Apply 3x3x3 convolution on emcal, hcal, trkp
        x6 = self.conv3x5x5(x_e_h_p)  # Apply 3x5x5 convolution on emcal, hcal, trkp
        x7 = self.conv3x7x7(x_e_h_p)  # Apply 3x7x7 convolution on emcal, hcal, trkp

        # Concatenate all the resulting feature maps (12 channels) along the channel dimension
        x = torch.cat((x2, x3, x4, x5, x6, x7), dim=1).view(-1, 12, 56, 56)

        # Pass through the encoder
        x = self.encoder(x)

        # Pass through the decoder to reconstruct the output
        x = self.decoder(x)

        return x  # Output shape: (batch_size, 1, 56, 56)