import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchio.transforms import Resize, CropOrPad, ZNormalization
from torch.utils.data import random_split

class MyMRIHeartDataset(Dataset):
    def __init__(self, input_folder, transform=None):
        self.input_folder = input_folder
        self.transform = transform
        self.files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
        print("MRI files found:")
        for file in self.files:
            print(file)
        print(f"Initialized dataset with {len(self.files)} images.")    
        
    def __len__(self):
        return len(self.files)    
        
    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = nib.load(image_path).get_fdata(dtype=np.float32)
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = torch.from_numpy(image)  # Convert numpy array to torch tensor        
        
        if self.transform:
            image = self.transform(image)        
            print(f"Loaded and transformed image {idx+1}/{len(self)}.")
        return image

def get_transforms():
    print("Creating transformations for the dataset...")
    transform = Compose([
        Resize((128, 128, 128)),  # Resize images to a smaller size
        CropOrPad((128, 128, 128)),  # Crop or pad images to have the same shape
        ZNormalization()  # Normalize each image to have zero mean and unit variance
    ])
    return transform

def split_dataset(dataset, num_val_samples=3):
    num_samples = len(dataset)
    val_length = num_val_samples  # Set the exact number of validation samples
    train_length = num_samples - val_length  # Remaining samples are for training    
    if val_length > num_samples:
        raise ValueError("Number of validation samples exceeds the total number of samples in the dataset.")    
        
    train_set, val_set = random_split(dataset, [train_length, val_length])
    print(f"Split dataset into {(train_length)} training and {(val_length)} validation samples.")
    return train_set, val_set
    
# Example usage
input_folder = 'Task02_Heart/imagesTr' # Change this to the path where your data is stored
dataset = MyMRIHeartDataset(input_folder, transform=get_transforms())
train_set, val_set = split_dataset(dataset, num_val_samples=3)