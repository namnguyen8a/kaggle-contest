import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms.functional as TF
import random

class LiverDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            phase (str): 'train' or 'test'.
            transform (callable): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.samples = []
        
        if phase == 'train':
            self.volume_dir = os.path.join(root_dir, 'train', 'volume')
            self.mask_dir = os.path.join(root_dir, 'train', 'segmentation')
            self.volume_paths = sorted(glob.glob(os.path.join(self.volume_dir, '*.nii')))
            self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.nii')))
        else:
            self.volume_dir = os.path.join(root_dir, 'test')
            self.volume_paths = sorted(glob.glob(os.path.join(self.volume_dir, '*.nii')))
            self.mask_paths = None

        print(f"Indexing {phase} dataset...")
        for idx, vol_path in enumerate(self.volume_paths):
            try:
                img = nib.load(vol_path)
                n_slices = img.shape[2]
                for s in range(n_slices):
                    self.samples.append({
                        'vol_path': vol_path,
                        'mask_path': self.mask_paths[idx] if self.mask_paths else None,
                        'slice_idx': s,
                        'case_id': os.path.basename(vol_path).replace('.nii', '')
                    })
            except Exception as e:
                print(f"Error loading {vol_path}: {e}")
                
        print(f"Indexed {len(self.samples)} slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vol_path = sample['vol_path']
        mask_path = sample['mask_path']
        slice_idx = sample['slice_idx']
        
        img_obj = nib.load(vol_path)
        image = img_obj.dataobj[..., slice_idx]
        image = np.array(image).astype(np.float32)
        
        if mask_path:
            mask_obj = nib.load(mask_path)
            mask = mask_obj.dataobj[..., slice_idx]
            mask = np.array(mask).astype(np.float32)
            # Normalize mask to binary [0, 1]
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros_like(image)

        # Preprocessing
        min_hu, max_hu = -100, 400
        image = np.clip(image, min_hu, max_hu)
        image = (image - min_hu) / (max_hu - min_hu)

        if image.ndim == 3:
            image = image.squeeze()
            
        # Convert to Tensor (C, H, W)
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return {
            'image': image,
            'mask': mask,
            'case_id': sample['case_id'],
            'slice_idx': slice_idx
        }

class Transforms:
    def __init__(self, phase='train'):
        self.phase = phase

    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, [512, 512], antialias=True)
        mask = TF.resize(mask, [512, 512], interpolation=TF.InterpolationMode.NEAREST)

        if self.phase == 'train':
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Rotate
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        return image, mask

def get_transforms(phase='train'):
    return Transforms(phase)
