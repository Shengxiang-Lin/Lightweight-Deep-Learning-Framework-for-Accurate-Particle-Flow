import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RNN, self).__init__()
        # Initialize the encoder LSTM layer
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Initialize the decoder LSTM layer
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Initialize a fully connected layer to map the hidden states to the output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Forward pass through the encoder LSTM
        # enc_out: Encoded output sequences
        # hidden: Hidden states of the encoder
        enc_out, hidden = self.encoder(x)
        
        # Forward pass through the decoder LSTM
        # dec_out: Decoded output sequences
        # _ : Decoder hidden states (not used here)
        dec_out, _ = self.decoder(enc_out, hidden)
        
        # Apply the fully connected layer to map decoder outputs to the desired output dimension
        out = self.fc(dec_out)
        return out