# Beginner's Guide to Liver Segmentation with Deep Learning

Welcome! This guide will walk you through the entire process of training a Deep Learning model to segment livers from 3D CT scans. We will use a **U-Net** architecture, which is the gold standard for medical image segmentation.

## 1. Project Structure

Here is the layout of the files we have created:

- **`dataset.py`**: Handles loading the CT scans (NIfTI files). It processes the 3D volumes by slicing them into 2D images so our 2D model can learn from them. It also handles "windowing" (adjusting contrast for liver tissue) and data augmentation (flipping/rotating images to make the model robust).
- **`model.py`**: Defines the **U-Net** architecture. This is a neural network that takes an image as input and outputs a "mask" (a binary image where 1=liver, 0=background).
- **`utils.py`**: Contains helper functions, most importantly `rle_encode` which converts our predicted masks into the text format required for the Kaggle submission.
- **`train.py`**: The engine room. This script trains the model. It feeds data to the model, calculates the error (Loss), and updates the model's weights to minimize that error.
- **`inference.py`**: Used after training. It loads the best saved model, predicts masks for the test set, and creates the `submission.csv` file.

## 2. Prerequisites

Before running the code, you need to ensure you have the necessary Python libraries installed. Open your terminal or command prompt and run:

```bash
pip install numpy pandas torch matplotlib nibabel albumentations tqdm
```

*Note: If you have a GPU (NVIDIA), make sure you have installed the version of PyTorch that supports CUDA for faster training.*

## 3. Understanding the Data

Medical images are often stored in `.nii` (NIfTI) format.
- **CT Scans**: These are 3D volumes. Think of them as a loaf of bread. We can slice them to get 2D images.
- **Hounsfield Units (HU)**: CT scans measure density. Water is 0, Air is -1000, Bone is +1000. The liver is usually between approx. -20 and 200.
- **Preprocessing**: In `dataset.py`, we "clip" the values to the range [-100, 400] to focus on the soft tissue and liver, ignoring bone and air. Then we normalize them to [0, 1].

## 4. The Model: U-Net

We use a **U-Net**. It's called that because its architecture looks like the letter U.
1. **Encoder (Left side)**: Takes the image and shrinks it down (using Max Pooling), extracting features like edges and textures.
2. **Bottleneck (Bottom)**: The most compressed representation of the image content.
3. **Decoder (Right side)**: Expands the features back up to the original size. Crucially, it uses "Skip Connections" to copy high-resolution details from the Encoder to the Decoder. This helps the model draw precise boundaries around the liver.

## 5. How to Train

To start training the model, simply run:

```bash
python train.py
```

**What happens during training?**
- The script splits the training data (80 cases) into a **Training Set** (to learn) and a **Validation Set** (to check progress).
- It runs for `20` epochs (passes through the data).
- It saves the model with the highest "Dice Score" (a measure of overlap between prediction and ground truth) as `checkpoints/best_model.pth`.

## 6. How to Create a Submission

Once training is done, generate your submission file for Kaggle:

```bash
python inference.py
```

This will:
1. Load `checkpoints/best_model.pth`.
2. Go through every file in the `test` folder.
3. Predict the liver mask slice-by-slice.
4. Save the results to `submission.csv`.

## 7. Tips for Improvement

- **More Epochs**: Try training for 50 or 100 epochs.
- **Data Augmentation**: Add more transforms in `dataset.py` (e.g., scaling, elastic deformations).
- **3D Model**: We are using a 2D approach (slice-by-slice). A full 3D U-Net (V-Net) might capture 3D context better but requires more GPU memory.
- **Loss Function**: Experiment with different weights for the Dice and BCE loss.

Good luck!
