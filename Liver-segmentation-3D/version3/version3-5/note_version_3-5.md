# Ref:
- https://www.kaggle.com/code/nnguyen2605/aio2025-warm-up-liver-tumor-segmentation-v3-5/edit (training)

# 📝 Experiment Report: Advanced Hybrid Losses

**Base Architecture:**
*   **Model:** U-Net with EfficientNet-B0 Encoder.
*   **Input:** **2.5D Context** (3 Channels: Prev/Curr/Next Slices).
*   **Optimizations:** `float16` Caching (Disk Safe), `num_workers=0` (Process Safe).

---

## 📊 Version 3.5: The "Hard Recall" Strategy - time 23676.2s - score
**Loss Function:** **Focal-Tversky Loss**
**Formula:** $ \text{Loss} = (1 - \text{TverskyIndex})^\gamma $

### 1. The Logic
This version combines the best feature of your current winner (V3.2 - Tversky) with the intent of your failed run (V3.1 - Focal).
*   **From Tversky (V3.2):** We keep $\alpha=0.3, \beta=0.7$. This forces the model to prioritize **Recall** (finding the liver is more important than ignoring background).
*   **From Focal (V3.1):** We add the Gamma exponent ($\gamma=0.75$). This acts as a "Hard Mining" mechanism.
    *   *Easy Examples:* If the model effectively finds the liver (Tversky Index $\approx$ 0.95), the gradient is down-weighted significantly.
    *   *Hard Examples:* If the model struggles (Tversky Index $\approx$ 0.4), the gradient is emphasized.

### 2. Why this is better than V3.1 (Focal)
*   **Stability:** Standard Focal Loss failed because pixel probabilities ($p$) approached 0 or 1, causing numerical explosions ($log(0)$).
*   **Safety:** The Tversky Index is calculated over the *entire batch* (or image), not per pixel. It is a smooth number between 0 and 1. Raising $(1 - Tversky)$ to a power is numerically safe and will not cause `NaN`.

### 3. Expected Outcome
*   **Goal:** To beat V3.2 (`0.96790`).
*   **Hypothesis:** V3.2 was good at finding the liver but might have been "lazy" on difficult edges. V3.5 forces the model to work harder on those difficult edges while maintaining high recall.

---