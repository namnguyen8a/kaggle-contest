Here is the detailed technical documentation for **Version 1** of your Liver Tumor Segmentation solution. This document serves as a "lab report" or a "model card" that explains exactly how you achieved the **Public LB Score of 0.95766**.

This serves as the foundation for future improvements.

---

# 📝 Project Report: Liver Segmentation - Version 1 (Baseline)
**Status:** Baseline Established
**Public Leaderboard Score:** `0.95766` (Dice Score)
**Technique:** 2.5D Segmentation (Slice-by-Slice) with U-Net

---

## 1. Data Preprocessing Pipeline
Medical images (CT scans) are not standard images. This is the most critical step in achieving a high score.

### A. Format Handling (3D to 2D)
*   **Raw Data:** 3D NIfTI files (`.nii`) containing volumetric data $(Height, Width, Depth)$.
*   **The Challenge:** 3D volumes are too large to fit into GPU memory for standard training.
*   **The Solution (2.5D Approach):** We treated the 3D volume as a stack of independent 2D images. We sliced every volume along the Z-axis (axial plane).
    *   *Input to Model:* Single Slice $(512 \times 512 \times 1)$
    *   *Output:* Single Mask $(512 \times 512 \times 1)$

### B. Normalization (The "Liver Window")
CT Scans use **Hounsfield Units (HU)**, which range from -1000 (Air) to +3000 (Bone). Neural networks expect values between 0 and 1. We cannot just divide by 3000 because the liver contrast would be lost.
*   **Strategy:** We used a specific "Window" to highlight soft tissue.
*   **Clip Range:** `[-100, 200]`
    *   Values below -100 (Air/Fat) become -100.
    *   Values above 200 (Bone/Metal) become 200.
    *   *Why?* The liver density is usually between 30 and 150 HU. This window makes the liver the primary focus of the image.
*   **Scaling:** Min-Max normalization to `[0, 1]`.
    $$ Pixel_{new} = \frac{Pixel_{raw} - (-100)}{200 - (-100)} $$

### C. Resizing
*   **Input Size:** `512 x 512`.
*   *Note:* We maintained the original resolution of the CT scans to preserve fine edges of the liver.

---

## 2. Model Architecture
We used a standard encoder-decoder architecture which is the "Gold Standard" for medical segmentation.

*   **Architecture:** **U-Net**
    *   *Why?* U-Net has skip-connections that concatenate high-level features (context) with low-level features (edges/texture). This helps the model know *where* the liver is and exactly *what shape* it is.
*   **Encoder (Backbone):** **EfficientNet-B0**
    *   *Weights:* Pre-trained on **ImageNet**.
    *   *Why?* EfficientNet-B0 is lightweight (fast training) but learns features very quickly. Using ImageNet weights allows the model to recognize basic shapes/edges immediately, even though CT scans look different from natural images.
*   **Input Channels:** 1 (Grayscale).
*   **Output Channels:** 1 (Binary Mask: Liver vs Background).
*   **Activation:** Sigmoid (outputs a probability 0.0 to 1.0 for every pixel).

---

## 3. Training Strategy

### Hyperparameters
*   **Image Size:** 512x512
*   **Batch Size:** 16 (Fits comfortably on Kaggle P100/T4 GPUs).
*   **Learning Rate:** `1e-4` (Standard safe starting point for AdamW).
*   **Epochs:** 15 (Convergence usually happens around epoch 10-12).
*   **Optimizer:** **AdamW** (Adam with Weight Decay) - helps prevent overfitting.

### Loss Function (Hybrid)
We used a combined loss function to handle the fact that the liver is small compared to the black background.
$$ Loss = 0.5 \times \text{BCE} + 0.5 \times \text{DiceLoss} $$
1.  **BCE (Binary Cross Entropy):** Measures pixel-by-pixel accuracy. Good for general classification.
2.  **Dice Loss:** Directly optimizes the evaluation metric (Overlap). It ignores the massive amount of black background pixels and focuses on the liver region.

### Augmentations (Albumentations)
To prevent the model from memorizing specific patient orientations, we applied:
*   **Horizontal Flip:** 50% chance.
*   **Vertical Flip:** 50% chance.
*   **Rotation:** +/- 10 degrees.

### Validation
*   **Scheme:** K-Fold Cross Validation (5 Folds).
*   **Baseline shortcut:** To save time for Version 1, we trained only on **Fold 0**. (Training on all 5 folds and averaging them usually boosts score by +0.005 to +0.01).

---

## 4. Inference Strategy (Prediction)

### TTA (Test Time Augmentation)
This is a key reason for the high **0.95+** score. During prediction, we didn't just predict once. For every slice:
1.  **Pass 1:** Predict the image normally.
2.  **Pass 2:** Flip the image horizontally -> Predict -> Flip the result back.
3.  **Result:** Average the two predictions.
*   *Why?* This smooths out noise and corrects minor errors at the edges of the liver.

### Post-Processing
*   **Thresholding:** `Predictions > 0.5` become 1 (Liver), else 0.
*   **RLE Encoding:** Standard Kaggle format for submission.

---

## 5. Summary Checklist for Newcomers
If you want to reproduce this result, ensure you have these settings:

| Component | Setting / Choice | Reason |
| :--- | :--- | :--- |
| **Data** | 3D NIfTI -> 2D Slice | Reduce memory usage |
| **Preprocessing** | Clip `[-100, 200]` | Isolate Liver tissue density |
| **Backbone** | EfficientNet-B0 | Fast, robust feature extractor |
| **Loss** | 0.5 BCE + 0.5 Dice | Balance pixel accuracy with overlap |
| **Inference** | TTA (Horizontal Flip) | Boosts accuracy without re-training |
| **Image Size** | 512x512 | Preserve detail |

---

## 6. Path to Improvement (Version 2 Ideas)
To improve from **0.957** to **0.96+** or **0.97+**, consider these steps:

1.  **Full K-Fold Training:** Train Folds 0, 1, 2, 3, 4 and ensemble them (average their predictions).
2.  **2.5D Context:** Instead of feeding 1 slice, feed 3 slices (Current, Previous, Next) as a 3-channel image. This gives the model "3D context".
3.  **Heavy Augmentations:** Add GridDistortion or ElasticTransform (livers are soft and deformable).
4.  **Post-Processing:** Use "Largest Connected Component" analysis. If the model predicts a small speck of liver floating far away from the main liver, remove it (it's likely noise).
5.  **Larger Backbone:** Switch from `efficientnet-b0` to `efficientnet-b4` or `mit_b2` (SegFormer).