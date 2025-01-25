import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchio.transforms import Resample, CropOrPad, ZNormalization, Resize
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchio
from mydataset import MyMRIHeartDataset, get_transforms
from mymodell import MySimpleUNet3D

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

accumulation_steps = 4  # Accumulate gradients over 4 batches

def train_my_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Initialize data loaders with the specified batch size
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            images = batch
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)  # Using MSE loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() 

        avg_train_loss = total_loss / len(train_loader) 

        # Validation phase
        model.eval()
        val_loss = 0        
        with torch.no_grad():
            for batch in val_loader:
                images = batch
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)  # Using MSE loss
                val_loss += loss.item()        
        avg_val_loss = val_loss / len(val_loader)        
        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Close the TensorBoard writer
writer.close()