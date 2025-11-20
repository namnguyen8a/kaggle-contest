import os
import torch
from torch.utils.data import DataLoader
from dataset import LiverDataset, get_transforms
from model import UNet
import traceback

def main():
    try:
        print("Initializing Dataset...")
        ds = LiverDataset('f:/kaggle-dataset/aio2025liverseg', phase='train', transform=get_transforms('train'))
        print(f"Dataset size: {len(ds)}")
        
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        
        print("Fetching one batch...")
        batch = next(iter(loader))
        print("Batch fetched successfully.")
        
        images = batch['image']
        masks = batch['mask']
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        
        print("Initializing Model...")
        model = UNet(n_channels=1, n_classes=1)
        
        print("Forward pass...")
        output = model(images)
        print(f"Output shape: {output.shape}")
        
        print("Success!")
        
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
