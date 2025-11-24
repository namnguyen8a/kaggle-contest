# Ref:
- https://www.kaggle.com/code/nnguyen2605/aio2025-warm-up-liver-tumor-segmentation-v3-2/edit (training)

## 📊 Version 3.2: The "Recall" Strategy - time 23552.0s - score 0.96790
**Loss Function:** `Tversky Loss (alpha=0.3, beta=0.7)`

### 1. The Logic
The Dice coefficient gives equal weight to False Positives (predicting liver where there is none) and False Negatives (missing the liver).
*   **Tversky Index:** $ \frac{TP}{TP + \alpha FP + \beta FN} $
*   We set **$\beta = 0.7$** (High penalty for False Negatives) and **$\alpha = 0.3$** (Low penalty for False Positives).

### 2. Why use it?
*   In medical segmentation, **missing a part of the organ (FN)** is usually worse than predicting a few extra pixels of background (FP).
*   By analyzing V1/V2 results, we often see "holes" in the liver mask. This loss function tells the model: *"I will punish you severely if you leave holes in the liver."*

### 3. Expected Outcome
*   **Pros:** Highly contiguous masks; very few holes; higher Recall.
*   **Cons:** The masks might be slightly "fatter" (slightly larger than the ground truth) because the model is afraid to miss pixels.