# Ref:
- https://www.kaggle.com/code/youpre/aio2025-warm-up-liver-tumor-segmentation-v3-3/edit (training)

## 📊 Version 3.3: The "Boundary" Strategy - time 23782.9s - score 0.96486
**Loss Function:** `0.5 * BCE + 0.5 * Lovász-Softmax`

### 1. The Logic
Most loss functions (like BCE) optimize pixel-by-pixel accuracy. They don't "care" about the shape or the overlap metric (IoU) directly because IoU is not differentiable (you can't calculate gradients for it).
*   **Lovász-Softmax** is a mathematical breakthrough that acts as a **differentiable surrogate for the Jaccard Index (IoU)**.
*   It directly optimizes the metric used on the Leaderboard.

### 2. Why use it?
*   It is widely considered the **SOTA (State of the Art)** loss function for Kaggle segmentation competitions.
*   It creates extremely crisp, accurate boundaries, whereas Dice loss can sometimes result in "blobby" or rounded shapes.

### 3. Expected Outcome
*   **Pros:** Highest potential spatial overlap score; sharpest edges.
*   **Cons:** It is computationally expensive (training might be slightly slower) and sometimes tricky to converge from scratch (which is why we combine it with BCE for stability).