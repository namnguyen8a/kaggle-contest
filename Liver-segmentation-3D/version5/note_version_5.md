# Ref:
- https://www.kaggle.com/code/youpre/aio2025-warm-up-liver-tumor-segmentation-v5/edit (training)

# 📝 Experiment Report: Hybrid Loss Functions

**Common Foundation (Both Versions):**
*   **Architecture:** U-Net.
*   **Backbone:** EfficientNet-B0 (Pre-trained on ImageNet).
*   **Input Data:** **2.5D Context** (Stack of 3 Slices: $i-1, i, i+1$).
*   **System Optimizations:** `num_workers=0` (Crash Prevention), `float16` Cache (Disk Saving).

---

## 📊 Version 5: The "Edge Refiner"
**Strategy:** **0.7 Dice Loss + 0.3 Boundary-Aware Loss**

### 1. The Motivation
*   **Observation:** In previous versions, the model found the liver location well, but the contours (edges) were sometimes "blobby" or lacked precision.
*   **The Problem:** Standard Dice loss focuses on **Area Overlap**. It doesn't care if the boundary is off by 1-2 pixels as long as the total number of internal pixels is correct.

### 2. The Solution (Loss Function)
We use a hybrid loss:
*   **70% Dice Loss:** Keeps the global shape and ensures high overlap score.
*   **30% Focal Loss (`gamma=2.0`):**
    *   *Note on Boundary Loss:* True "Hausdorff Distance Loss" is extremely slow to calculate. In deep learning, **Focal Loss with high gamma** acts as a "Boundary Proxy."
    *   **Why?** The pixels in the *center* of the liver are easy (probability $\approx 1.0$). The pixels at the *edges* are hard (probability $\approx 0.5$). Focal loss minimizes the gradient for the easy center and **maximizes the gradient for the hard edges**. This forces the model to sharpen the boundaries.

### 3. Hypothesis
*   **Expectation:** This model should produce the cleanest, most visually accurate masks.
*   **Best For:** If the Leaderboard metric penalizes rough edges heavily.