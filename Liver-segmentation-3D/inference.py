import os
import glob
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
import torchvision.transforms.functional as TF

from model import UNet
from utils import rle_encode

# --- Config ---
DATA_DIR = 'f:/kaggle-dataset/aio2025liverseg'
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
OUTPUT_CSV = 'submission.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_image(image):
    # Resize to 512x512
    # Input: (H, W) numpy array
    # Output: (1, 512, 512) tensor
    
    # Convert to tensor
    image = torch.tensor(image).unsqueeze(0) # (1, H, W)
    
    # Resize
    image = TF.resize(image, [512, 512], antialias=True)
    
    return image

def main():
    # Load Model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Using random weights.")
        
    model.eval()
    
    test_volume_dir = os.path.join(DATA_DIR, 'test')
    if os.path.exists(os.path.join(test_volume_dir, 'volume')):
        test_volume_dir = os.path.join(test_volume_dir, 'volume')
        
    test_files = sorted(glob.glob(os.path.join(test_volume_dir, '*.nii')))
    print(f"Found {len(test_files)} test files.")
    
    submission_data = []
    
    for vol_path in tqdm(test_files, desc="Inference"):
        vol_id = os.path.basename(vol_path).replace('.nii', '')
        
        try:
            # Load volume
            img_obj = nib.load(vol_path)
            volume = img_obj.get_fdata() # (H, W, D)
            
            # Iterate slices
            n_slices = volume.shape[2]
            for s in range(n_slices):
                slice_img = volume[..., s]
                
                # Preprocess
                min_hu, max_hu = -100, 400
                slice_img = np.clip(slice_img, min_hu, max_hu)
                slice_img = (slice_img - min_hu) / (max_hu - min_hu)
                
                # Transform
                input_tensor = preprocess_image(slice_img.astype(np.float32))
                input_tensor = input_tensor.unsqueeze(0).to(DEVICE) # (1, 1, 512, 512)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
                    
                rle = rle_encode(pred_mask)
                submission_data.append([f"{vol_id}_{s}", rle])
        except Exception as e:
            print(f"Error processing {vol_path}: {e}")
            
    df = pd.DataFrame(submission_data, columns=['id', 'rle'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Submission saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
