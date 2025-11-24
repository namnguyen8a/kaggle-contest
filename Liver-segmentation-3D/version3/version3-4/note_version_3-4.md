# Ref:
- https://www.kaggle.com/code/nnguyen26/aio2025-warm-up-liver-tumor-segmentation-v3-4/edit (training)
- https://www.kaggle.com/code/nnguyen26/liver-segment-infer-v3-4/edit (inference)

# 📝 Experiment Report: Version 3.4 (Stable Imbalance Handling) - time 22757.9s - score 0.95981

**Status:** Running (Stability Optimized)
**Core Concept:** **Recall Maximization via Class Weighting**
**Key Change:** Replaced Focal Loss with **Weighted Binary Cross Entropy (`pos_weight=5.0`)**.

---

## 1. The Problem: Why V3.1 Failed
In the previous attempt (V3.1), we tried to use **Focal Loss** to solve the class imbalance problem (small liver, huge background).
*   **The Crash:** The training log showed `Train Loss: nan`.
*   **The Reason:** Focal Loss uses exponents ($power^\gamma$) and logarithms. When training with Mixed Precision (FP16), predictions very close to 0 or 1 caused mathematical overflows (numbers too big for the GPU to handle), leading to `NaN`.

## 2. The Solution: Weighted BCE
**Version 3.4** achieves the exact same goal as V3.1 (focusing on the liver) but uses a mathematically safer method.

### The Math
Standard BCE calculates error equally for Background (0) and Liver (1).
$$ \text{Loss} = - [ y \cdot \log(p) + (1-y) \cdot \log(1-p) ] $$

**Weighted BCE** adds a multiplier ($W$) only to the Liver term:
$$ \text{Loss} = - [ \mathbf{W} \cdot y \cdot \log(p) + (1-y) \cdot \log(1-p) ] $$

*   **We set $W = 5.0$**.
*   **Impact:** Every time the model fails to detect a pixel of liver, it is penalized **5 times harder** than if it confuses background.
*   **Stability:** This uses simple multiplication, which never explodes into `NaN` like the exponents in Focal Loss.

---

## 3. Technical Architecture (Inherited from V3)
This version retains the successful "Context" logic from Version 3.

*   **Architecture:** U-Net.
*   **Backbone:** EfficientNet-B0.
*   **Input Data:** **2.5D Context** (3 Channels).
    *   Channel 0: Slice $i-1$
    *   Channel 1: Slice $i$
    *   Channel 2: Slice $i+1$
*   **Benefits:** The model sees the shape of the liver changing across slices, helping it distinguish the liver from the spleen/stomach.

---

## 4. Implementation Safeguards
To ensure this run finishes successfully within the 12-hour limit, we added:

1.  **Gradient Clipping:** `clip_grad_norm_(model.parameters(), 2.0)`
    *   If the loss spikes suddenly, this forces the gradients to stay small, preventing the model from "exploding" or diverging.
2.  **Smooth Metric:** `dice_coef_metric(..., smooth=1.0)`
    *   Ensures we never see `Val Dice: NaN` in the logs, even if the ground truth mask is empty.
3.  **Workers=0:** Prevents the multiprocessing deadlock that caused the hanging issue in earlier runs.

---

## 5. Hypothesis & Expectations

### Goal
We want to beat the Version 3 score (`0.95840`).

### Expected Behavior
*   **High Recall:** Because of the `pos_weight=5.0`, this model will be very aggressive. It will likely find *all* the liver pixels.
*   **Risk (False Positives):** It might be *too* aggressive and predict liver in slightly ambiguous areas (like the heart or spleen) because the penalty for missing liver is so high.
*   **Correction:** The **Dice Loss** component (50% of the total loss) acts as a counterbalance, encouraging the model to keep the shape accurate and not just splash white pixels everywhere.

### Comparison Table

| Version | Loss Strategy | Focus | Stability |
| :--- | :--- | :--- | :--- |
| **V3.0** | Standard BCE + Dice | Balanced | High |
| **V3.1** | Focal + Dice | Hard Mining | **Unstable (Failed)** |
| **V3.4** | **Weighted BCE (5x) + Dice** | **Recall (No Missing Liver)** | **High** |

---

### ⏱️ Next Steps
Once V3.2, V3.3, and V3.4 finish training:
1.  Compare their **Public LB Scores**.
2.  If V3.4 wins, it means the dataset has "hard to find" liver edges.
3.  If V3.3 (Lovász) wins, it means the dataset requires very sharp boundary precision.