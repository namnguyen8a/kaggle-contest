# Ref:
- https://www.kaggle.com/code/nnguyen2605/aio2025-warm-up-liver-tumor-segmentation-v6/edit (training)

---

# 📝 Experiment Report: Hybrid Loss Functions

**Common Foundation (Both Versions):**
*   **Architecture:** U-Net.
*   **Backbone:** EfficientNet-B0 (Pre-trained on ImageNet).
*   **Input Data:** **2.5D Context** (Stack of 3 Slices: $i-1, i, i+1$).
*   **System Optimizations:** `num_workers=0` (Crash Prevention), `float16` Cache (Disk Saving).

---

## 📊 Version 6: The "Robust Metric"
**Strategy:** **0.6 Tversky Loss + 0.4 Lovász-Softmax Loss**

### 1. The Motivation
*   **Observation:**
    *   **V3.2 (Tversky)** was your 2nd best model (`0.9679`), proving that handling False Negatives (Recall) is vital.
    *   **V3.3 (Lovász)** is a SOTA loss function that directly optimizes the competition metric (IoU/Jaccard) using a convex surrogate.
*   **The Idea:** Combine the **Stability and Recall** of Tversky with the **Metric Optimization** of Lovász.

### 2. The Solution (Loss Function)
*   **60% Tversky ($\alpha=0.3, \beta=0.7$):**
    *   This acts as the "Base." It ensures the model finds *all* the liver tissue and doesn't leave holes (High Recall).
*   **40% Lovász-Softmax:**
    *   This acts as the "Fine-Tuner." It sorts the pixel errors and optimizes the Jaccard Index directly. It prevents the Tversky loss from over-predicting (making the mask too fat).

### 3. Hypothesis
*   **Expectation:** This is mathematically the strongest combination. It covers Class Imbalance (via Tversky) and Metric Alignment (via Lovász).
*   **Risk:** Lovász is slower to compute. This version will take the longest to train.

---