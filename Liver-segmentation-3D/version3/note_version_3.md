# Ref:
- https://www.kaggle.com/code/nnguyen2605/aio2025-warm-up-liver-tumor-segmentation-v3/edit (training)

# 📝 Experiment Report: Version 3 (2.5D Context - Optimized) - score 0.95840 - time 15307.3s

**Status:** Running (Disk Optimization Applied)
**Core Concept:** **2.5D Segmentation** (Pseudo-3D)
**Key Improvement:** Solved "Disk Full" error by constructing data on-the-fly.

---

## 1. The Hypothesis: Why "2.5D"?
In the Baseline (V1), the model looks at one slice at a time.
*   **Problem:** A slice of the liver might look identical to a slice of the stomach or spleen if you lack context.
*   **Solution:** By feeding the model the **Current Slice**, the **Slice Above**, and the **Slice Below**, we give it 3D context without the heavy memory cost of full 3D models (like V-Net).
*   **Analogy:** It's like watching a 3-frame movie clip instead of a single still photo. You can see how the organ shape changes.

---

## 2. The "Disk Full" Fix (Technical Detail)
The previous attempt failed because it tried to save pre-stacked 3-channel images to the Kaggle disk.
*   *Old Math (Failed):* 80 Vols $\times$ 3 Channels $\times$ `float32` = **~45 GB** (Crash).
*   **New Math (Optimized):**
    1.  Save only **1 Channel** (Current Slice).
    2.  Convert to **`float16`** (Half Precision) before saving.
    3.  **On-the-Fly Construction:** The `Dataset` class reads 3 separate files (Prev, Curr, Next) during training and stacks them in RAM.
    *   *Result:* **~6 GB** (Safe).

---

## 3. Data Pipeline & Preprocessing

### A. Caching Phase
We iterate through every volume and save every slice as an individual `.npy` file.
*   **Normalization:** Standard Liver Window `[-100, 200]` scaled to `[0, 1]`.
*   **Storage:** `img_{id}_{slice}.npy` (containing only 1 slice).

### B. The Dataset Loader (The Magic Part)
When the model asks for **Slice $i$**, the DataLoader does this:
1.  **Identify Neighbors:** It calculates indices for $i-1$ (Previous) and $i+1$ (Next).
    *   *Edge Case:* If $i=0$, "Previous" is clamped to 0. If $i=Max$, "Next" is clamped to Max.
2.  **Load Files:** It loads 3 separate `.npy` files from the disk.
3.  **Stack:** It combines them into a single array of shape `(Height, Width, 3)`.
4.  **Return:** A 3-channel image that looks like an RGB photo to the model.

---

## 4. Model Architecture
*   **Architecture:** **U-Net** (Standard).
*   **Backbone:** **EfficientNet-B0**.
    *   *Note:* We kept the backbone small (B0) to isolate the effect of the "3-Channel Input." If this scores high, we know it's because of the *Context*, not just model size.
*   **Input Channels:** **3** (Crucial Change).
    *   Channel 0: Slice $i-1$
    *   Channel 1: Slice $i$
    *   Channel 2: Slice $i+1$
*   **Output:** 1 Channel (Binary Mask).

---

## 5. Training Configuration
*   **Augmentations:** Light.
    *   Horizontal Flip (50%)
    *   Vertical Flip (50%)
    *   Rotation (+/- 10 deg)
    *   *Note:* We use `Albumentations`. Since our input is 3-channel (like RGB), standard RGB augmentations work perfectly here.
*   **Loss Function:** `0.5 * BCE + 0.5 * Dice`.
*   **Batch Size:** 16.

---

## 6. Inference Strategy
We cannot just predict slice-by-slice normally because the model *needs* neighbors.
1.  **Padding:** We pad the input 3D volume with one empty slice at the start and one at the end.
    *   *Why?* To predict the very first slice (Index 0), we need [Index -1, Index 0, Index 1]. Padding handles this.
2.  **Sliding Window:** We slide a window of size 3 across the Z-axis.
3.  **Permutation:** The model expects `(Batch, Channels, Height, Width)`. We reshape the stack from `(H, W, 3)` to `(1, 3, H, W)`.

---

## 7. What to Watch For
When the results come back:
*   **If Score > 0.96:** The 3D context is working. The model is successfully using the neighboring slices to define boundaries better.
*   **If Score < 0.95:** The alignment might be off, or the "clamping" logic at the edges of the liver is confusing the model.

**Next Step:** If this version wins, your final "Gold" submission should combine **Version 2 (Big Model)** with **Version 3 (3-Channel Context)**. That combination is usually a competition winner.
