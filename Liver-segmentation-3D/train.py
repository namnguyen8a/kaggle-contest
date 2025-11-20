import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from dataset import LiverDataset, get_transforms
from model import UNet
from utils import set_seed

# --- Config ---
DATA_DIR = 'f:/kaggle-dataset/aio2025liverseg'
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 20
DEVICE = 'cpu'  # Forced CPU due to RTX 5050 (sm_120) incompatibility with PyTorch 2.5.1
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: logits
        inputs = torch.sigmoid(inputs)
        
        # flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            dice_score += dice_coeff(preds.float(), masks).item()
            
    return running_loss / len(loader), dice_score / len(loader)

def main():
    set_seed(42)
    
    # Dataset
    full_dataset = LiverDataset(DATA_DIR, phase='train', transform=get_transforms('train'))
    
    # Split Train/Val (e.g., 80/20)
    # Note: Splitting by slices is risky if slices from same patient end up in both sets.
    # Ideally split by patient ID. For simplicity here, we use random split but be aware of data leakage.
    # A better approach: Split by patient ID in Dataset class.
    # Let's do a simple split for now as a starter.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = DiceBCELoss()
    
    best_dice = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == '__main__':
    main()
