# Project Codebase Explanation

This document provides a detailed technical explanation of the codebase for the Kaggle AIO2025 Liver Segmentation project. It is intended to help other Large Language Models (LLMs) or developers understand the structure, logic, and purpose of each file in the repository.

## Project Overview
**Goal**: Segment liver regions from 3D CT scan volumes (NIfTI format).
**Approach**: 2D U-Net model trained on slices extracted from 3D volumes.
**Input**: CT volumes (`.nii` files).
**Output**: Run-Length Encoded (RLE) masks for submission.

---

## File Descriptions

### 1. `dataset.py`
**Purpose**: Handles data loading, preprocessing, and augmentation. It converts 3D NIfTI volumes into 2D slices suitable for training a 2D network.

**Key Components**:
- **`LiverDataset` Class**:
    - **`__init__`**:
        - Scans the `train` or `test` directories for `.nii` files.
        - **Indexing**: Iterates through all 3D volumes and pre-indexes every slice. It stores metadata (volume path, slice index) in a list `self.samples`. This allows random access to any slice across all volumes during training.
    - **`__getitem__`**:
        - Loads the specific slice from the NIfTI volume using `nibabel`.
        - **Preprocessing**:
            - **Windowing**: Clips Hounsfield Units (HU) to the range `[-100, 400]` (broad liver window) to focus on relevant tissue contrast.
            - **Normalization**: Normalizes pixel values to `[0, 1]`.
        - **Augmentation**: Applies transformations (resize, flip, rotate) if provided.
        - Returns a dictionary containing the image tensor, mask tensor (if training), case ID, and slice index.
- **`get_transforms(phase)`**:
    - Returns `albumentations` composition.
    - **Train**: Resize to 512x512, Horizontal Flip, slight Rotation.
    - **Test/Val**: Resize to 512x512 only.

### 2. `model.py`
**Purpose**: Defines the neural network architecture.

**Key Components**:
- **`UNet` Class**:
    - A standard U-Net implementation for biomedical image segmentation.
    - **Encoder (Downsampling)**: 4 levels of `DoubleConv` (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU) followed by MaxPooling. Channels increase: 64 -> 128 -> 256 -> 512 -> 1024.
    - **Decoder (Upsampling)**: 4 levels of bilinear upsampling followed by concatenation with skip connections from the encoder, then `DoubleConv`. Channels decrease: 1024 -> 512 -> 256 -> 128 -> 64.
    - **Output Layer**: 1x1 Convolution to map 64 features to `n_classes` (1 for binary segmentation). Note: It returns raw logits (no Sigmoid activation included in the model itself).

### 3. `train.py`
**Purpose**: Orchestrates the training process.

**Key Components**:
- **Configuration**: Constants for `BATCH_SIZE`, `LR` (Learning Rate), `EPOCHS`, etc.
- **`DiceBCELoss` Class**:
    - A combined loss function: `BCEWithLogitsLoss + Dice Loss`.
    - **BCE**: Handles pixel-wise classification accuracy.
    - **Dice**: Optimizes for overlap, helpful for class imbalance (small liver vs large background).
- **`train_one_epoch`**:
    - Iterates through the `DataLoader`.
    - Performs forward pass, loss calculation, backward pass, and optimizer step.
- **`validate`**:
    - Evaluates model on validation set.
    - Calculates Loss and Dice Coefficient (metric).
- **`main`**:
    - Sets random seeds.
    - Initializes `LiverDataset` and splits it into Train (80%) and Validation (20%) sets.
    - **Note on Split**: Currently uses random split of slices. *Potential Improvement*: Split by patient ID to prevent data leakage.
    - Runs the training loop for `EPOCHS`.
    - Saves the model with the best validation Dice score to `checkpoints/best_model.pth`.

### 4. `inference.py`
**Purpose**: Generates predictions for the test set and creates the submission file.

**Key Components**:
- **`main`**:
    - Loads the trained model from `checkpoints/best_model.pth`.
    - Iterates through all test volumes in `aio2025liverseg/test`.
    - **Slice-by-Slice Inference**:
        - Loads volume, extracts slices.
        - Preprocesses (Windowing `[-100, 400]`, Normalization).
        - Resizes to 512x512.
        - Runs model inference (Sigmoid > 0.5 threshold).
    - **RLE Encoding**: Encodes the binary mask using `rle_encode` from `utils.py`.
    - **Submission**: Aggregates results into a DataFrame and saves `submission.csv`.

### 5. `utils.py`
**Purpose**: Utility functions used across the project.

**Key Components**:
- **`set_seed`**: Ensures reproducibility by fixing seeds for `random`, `numpy`, and `torch`.
- **`rle_encode`**: Converts a binary mask into Run-Length Encoding format required for Kaggle submission.
    - **Logic**: Flattens mask (column-major), finds runs of 1s, and formats as "start length start length...".
- **`show_sample`**: Matplotlib helper to visualize an image and its mask overlay.

### 6. `verify_setup.py`
**Purpose**: A sanity check script to ensure the environment is ready.

**Key Components**:
- Checks if required libraries (`numpy`, `torch`, `nibabel`, etc.) are installed.
- Verifies that the dataset paths (`train/volume`, `train/segmentation`, `test`) exist and contain files.

---

## Execution Flow
1.  **Setup**: Run `python verify_setup.py` to check dependencies and data.
2.  **Train**: Run `python train.py` to train the U-Net. This saves `best_model.pth`.
3.  **Inference**: Run `python inference.py` to generate `submission.csv` using the trained model.
