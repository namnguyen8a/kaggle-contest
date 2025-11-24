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

# Log + note:
98.3s	26	  warnings.warn(
98.3s	27	
98.3s	28	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
98.3s	29	  warnings.warn(
98.3s	30	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
98.3s	31	  warnings.warn(
101.1s	32	[0mCaching Data V5 (2.5D Context)...
541.5s	33	
541.5s	34	=== FOLD 0 STARTED ===
2063.0s	35	Fold 0 | Epoch 1 | Train Loss: nan | Val Dice: 0.5289
2063.0s	36	--> Best Model Saved! (0.5289)
3569.5s	37	Fold 0 | Epoch 2 | Train Loss: nan | Val Dice: 0.5289


The issue with Version 5 is the same one that killed Version 3.1: Mathematical Instability (Exploding Gradients) caused by the FocalLoss component inside your BoundaryAwareDiceLoss.
💥 The Diagnosis
You defined:
code
Python
self.boundary_proxy = smp.losses.FocalLoss(mode='binary', gamma=2.0)
When training with Mixed Precision (AMP), FocalLoss with gamma=2.0 involves calculating 
(
1
−
p
)
2
⋅
log
⁡
(
p
)
(1−p) 
2
 ⋅log(p)
. If the model predicts 
p
≈
0
p≈0
 or 
p
≈
1
p≈1
 perfectly, the gradient calculation divides by zero or creates an infinitely small number that underflows float16, resulting in NaN.
🛠️ The Fix: Use "Log-Cosh Dice Loss" instead of Focal
Since you want to refine boundaries (edges) without using unstable Focal loss, the best alternative is Log-Cosh Dice Loss.
Math: 
log
⁡
(
cosh
⁡
(
DiceLoss
)
)
log(cosh(DiceLoss))
.
Why? It is a smoothing function. It works similarly to L2 regression for segmentation. It penalizes errors accurately but is mathematically bounded (it cannot explode to infinity), making it 100% stable in FP16.
Here is the Corrected Version 5 Script.

---

Here is the detailed **Experiment Tracking Report** for **Version 5 (Fixed)**.

This document records the shift from the unstable "Boundary-Aware" attempt (which crashed with `NaN`) to the mathematically safe "Log-Cosh" strategy.

---

# 📝 Experiment Report: Version 5 (Fixed) - time 23861.7s - score 0.96073

**Status:** Running (Stability Optimized)
**Core Concept:** **Boundary Smoothing & Robustness**
**Key Change:** Replaced unstable Focal Loss with **Log-Cosh Dice Loss**.

---

## 1. The Problem: Why the Original V5 Failed
In the first attempt at Version 5, we tried to use **Focal Loss** to approximate boundary awareness.
*   **The Crash:** `Train Loss: nan` at Epoch 1.
*   **The Reason:** Focal Loss calculates $-(1-p)^\gamma \log(p)$. In Mixed Precision (FP16) training, if the model predicts a probability $p$ very close to 0 or 1 (which happens often with simple backgrounds), the gradient calculation creates values that are too large for FP16 storage ("Overflow"), resulting in `NaN` (Not a Number). Once `NaN` appears, the model weights are destroyed.

## 2. The Solution: Log-Cosh Dice Loss
**Version 5 (Fixed)** uses a loss function designed for regression-like smoothness in segmentation.

### The Math
$$ L_{lc-dice} = \log(\cosh(\text{DiceLoss})) $$

*   **$\cosh(x)$ (Hyperbolic Cosine):** $\frac{e^x + e^{-x}}{2}$. It is a smooth, convex function that looks like a parabola near zero.
*   **$\log(\cdot)$:** Dampens the effect of outliers.

### Why is it better?
1.  **Stability:** Unlike Focal Loss, `cosh` and `log` are defined for all real numbers and do not explode to infinity easily. It is safe for FP16.
2.  **Smoothing:** It acts similarly to L2 regularization (Mean Squared Error) but for Dice scores. It creates very smooth gradients even when the model is close to convergence.
3.  **Boundary Effect:** Because it is a non-linear mapping of the Dice score, it penalizes errors *more* when accuracy is low (easy examples) and *smoothly* when accuracy is high (fine-tuning edges), effectively acting as a stable edge-refiner.

---

## 3. Technical Architecture
This version retains the successful "Context" logic from Version 3.

*   **Architecture:** U-Net.
*   **Backbone:** EfficientNet-B0 (Kept small to isolate the Loss function impact).
*   **Input Data:** **2.5D Context** (3 Channels: $i-1, i, i+1$).
*   **Optimizations:**
    *   `num_workers=0`: Prevents multiprocessing deadlocks.
    *   `float16` Caching: Reduces disk usage to ~7GB.
    *   **Gradient Clipping:** Added `clip_grad_norm_` to the training loop as a final safety net against explosions.

---

## 4. Hypothesis & Expectations

### Goal
To beat the current 2nd best score (`0.9679` from V3.2 Tversky) by producing cleaner edges.

### Comparison to V3.2 (Tversky)
*   **V3.2 (Tversky):** Focuses on **Recall** (Finding *all* the liver pixels, even if it means over-segmenting slightly). Results in high coverage but potentially "fat" masks.
*   **V5 (Log-Cosh):** Focuses on **Accuracy & Smoothness**. It tries to fit the mask perfectly without over-punishing small errors linearly.

### Expected Outcome
*   **Pros:** This model should have the most stable training curve (loss decreases smoothly without spikes). The masks should look visually cleaner than Tversky.
*   **Cons:** It might not be as aggressive at finding tiny, disconnected liver sections as Tversky or Weighted BCE.

---

## 5. Next Steps (After Training)
Once this training finishes (approx. 5-6 hours):
1.  Check the **Validation Dice**.
2.  If `Val Dice > 0.970`, this is your new best single model.
3.  **Final Submission Idea:** Ensemble **V2 (Big Model)** + **V3.2 (Recall Model)** + **V5 (Smooth Model)**. Averaging these three distinct "viewpoints" is the classic recipe for a Kaggle Gold/Silver finish.
