import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rle_encode(mask: np.ndarray) -> str:
    """
    Run-length encode a 2D binary mask (1 for foreground, 0 for background)
    Empty mask -> '1 0'.
    """
    assert mask.ndim == 2, "rle_encode expects a 2D mask"
    pixels = mask.astype(np.uint8).flatten(order='F')  # column-major
    if pixels.max() == 0:
        return "1 0"
    # Pad with zeros at both ends to catch transitions cleanly
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))

def show_sample(image, mask=None, title=None):
    """
    Helper to visualize a 2D slice and its mask.
    image: 2D numpy array
    mask: 2D numpy array (optional)
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis('off')
    
    if mask is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='gray')
        plt.imshow(mask, alpha=0.5, cmap='jet') # Overlay mask
        plt.title("Mask Overlay")
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.show()
