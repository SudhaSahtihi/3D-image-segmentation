import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class MySimpleUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MySimpleUNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Conv3d(16, out_channels, kernel_size=1)    
        
    def forward(self, x):
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        output = self.output(dec_features)
        return output

# Initialize the model
my_model = MySimpleUNet3D(in_channels=1, out_channels=1)

# Check if GPU is available and move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model.to(device)

# Ensure model has trainable parameters
if not list(my_model.parameters()):
    raise ValueError("Model has no trainable parameters. Check your model architecture.")

# Initialize optimizer after moving the model to the correct device
optimizer = optim.Adam(my_model.parameters(), lr=0.001)

# Setup TensorBoard
writer = SummaryWriter('runs/MySimple_UNet3D_experiment')

# Define criterion
criterion = nn.MSELoss()

# Define gradient accumulation steps
accumulation_steps = 4  # Accumulate gradients over 4 batches