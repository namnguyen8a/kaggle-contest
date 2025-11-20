# Kaggle Liver Segmentation - Detailed Guide

This guide provides step-by-step instructions on how to set up, run, and understand the Liver Segmentation project.

## 1. Prerequisites & Setup

### Requirements
You need Python installed (which you have). The project relies on several external libraries listed in `requirements.txt`.

### Installation
To install all necessary dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

*Note: If you have a GPU and want to use CUDA, ensure you have the appropriate version of PyTorch installed. The standard command above usually installs a compatible version, but you can verify at [pytorch.org](https://pytorch.org).*

### Verification
After installation, run the verification script to ensure everything is ready:

```bash
python verify_setup.py
```
If this prints "You are ready to run 'python train.py'!", you are good to go.

---

## 2. How to Run

### Step 1: Training
To train the model from scratch:

```bash
python train.py
```
- This will load the data, train the U-Net model for 20 epochs (default), and save the best model to `checkpoints/best_model.pth`.
- **Output**: You will see a progress bar and loss metrics for each epoch.

### Step 2: Inference (Generating Predictions)
Once you have a trained model (or if you have a pre-trained `best_model.pth` in the `checkpoints` folder):

```bash
python inference.py
```
- This reads the test volumes, generates segmentation masks, encodes them in RLE format, and saves them to `submission.csv`.
- **Output**: `submission.csv` ready for Kaggle submission.

---

## 3. File Structure & Explanations

Here is a detailed breakdown of every file in the project:

### Core Scripts

*   **`train.py`**: The main training script.
    *   **What it does**: Sets up the data loader, initializes the U-Net model and optimizer, and runs the training loop. It monitors validation performance (Dice score) and saves the best model.
    *   **Key Variables**: `BATCH_SIZE`, `LR` (Learning Rate), `EPOCHS`.

*   **`inference.py`**: The prediction script.
    *   **What it does**: Loads the saved model, processes test images slice-by-slice, predicts the liver mask, and creates the submission file.
    *   **Key Output**: `submission.csv`.

*   **`dataset.py`**: Data handling logic.
    *   **What it does**: Defines the `LiverDataset` class. It reads the 3D NIfTI files (`.nii`), extracts 2D slices, applies preprocessing (windowing/normalization), and handles data augmentation (flipping/rotating) during training.

*   **`model.py`**: Neural Network Architecture.
    *   **What it does**: Defines the `UNet` class using PyTorch. This is a standard architecture for medical image segmentation consisting of an encoder (downsampling) and a decoder (upsampling) with skip connections.

### Utilities

*   **`utils.py`**: Helper functions.
    *   **What it does**: Contains `set_seed` for reproducibility, `rle_encode` for formatting predictions for Kaggle, and `show_sample` for visualizing images.

*   **`verify_setup.py`**: Environment checker.
    *   **What it does**: Checks if all libraries are installed and if the dataset files are in the correct locations.

*   **`requirements.txt`**: Dependency list.
    *   **What it does**: Lists all Python packages required to run the project.

### Documentation

*   **`project_explanation.md`**: A technical deep-dive into the codebase (useful for developers/LLMs).
*   **`detailed_guide.md`**: This file! A user-friendly guide to running the project.

---

## 4. Troubleshooting

*   **Missing Libraries**: Run `pip install -r requirements.txt`.
*   **CUDA/GPU Errors**: If you don't have a GPU, the code automatically falls back to CPU (`device = 'cpu'`). If you do have a GPU but it's not being used, check your PyTorch installation.
*   **Memory Errors**: If you run out of memory (OOM), try reducing `BATCH_SIZE` in `train.py` (e.g., change 8 to 4 or 2).
