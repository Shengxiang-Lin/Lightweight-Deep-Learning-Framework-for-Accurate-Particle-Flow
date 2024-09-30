import torch.nn as nn
import torch.nn.functional as F
import torch

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Three convolutional layers with different kernel sizes (3x3, 5x5, 7x7)
        self.cov3x3 = nn.Conv2d(4, 2, kernel_size=3, padding=1)  # 3x3 kernel with padding for same size output
        self.cov5x5 = nn.Conv2d(4, 2, kernel_size=5, padding=2)  # 5x5 kernel with padding for same size output
        self.cov7x7 = nn.Conv2d(4, 2, kernel_size=7, padding=3)  # 7x7 kernel with padding for same size output

        # Encoder part (a sequence of convolutional, batch normalization, and ReLU layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, padding=2),  # First layer after concatenation of cov layers
            nn.BatchNorm2d(32),                          # Batch normalization for stable training
            nn.ReLU(inplace=True),                       # ReLU activation function

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # Second convolutional layer
            nn.BatchNorm2d(64),                          # Batch normalization
            nn.ReLU(inplace=True)                        # ReLU activation
        )

        # Decoder part (two convolutional layers for reconstruction)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2), # First layer of decoder
            nn.Conv2d(16, 1, kernel_size=5, padding=2),  # Second layer, outputting one channel
            nn.Sigmoid()                                 # Sigmoid activation for the output (for normalization)
        )

    def forward(self, x):  # Input x is a 4-channel image, i.e., concatenated (emcal, hcal, trkn, trkp)
        # Apply three convolutional layers with different kernel sizes
        x1 = self.cov3x3(x)  # Apply 3x3 convolution
        x2 = self.cov5x5(x)  # Apply 5x5 convolution
        x3 = self.cov7x7(x)  # Apply 7x7 convolution

        # Concatenate the outputs of the three convolutional layers along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)  # (2+2+2 = 6 channels)

        # Pass the concatenated output through the encoder
        x = self.encoder(x)

        # Pass the encoded features through the decoder to reconstruct the output
        x = self.decoder(x)

        return x  # Final output (1 channel, 56x56)