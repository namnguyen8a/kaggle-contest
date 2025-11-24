# Ref:
- https://www.kaggle.com/code/youpre/aio2025-warm-up-liver-tumor-segmentation-v3-6/edit (training)

---

# 📝 Experiment Report: Advanced Hybrid Losses

**Base Architecture:**
*   **Model:** U-Net with EfficientNet-B0 Encoder.
*   **Input:** **2.5D Context** (3 Channels: Prev/Curr/Next Slices).
*   **Optimizations:** `float16` Caching (Disk Safe), `num_workers=0` (Process Safe).

---

## 📊 Version 3.6: The "Correlation" Strategy - time 23689.2s - score 0.95358
**Loss Function:** **0.5 Dice + 0.5 MCC (Matthews Correlation Coefficient)**

### 1. The Logic
Most loss functions (BCE, Focal) look at pixel error. Dice/Tversky/Jaccard look at Overlap. **MCC looks at Statistical Correlation.**
*   **MCC:** It considers all four quadrants of the confusion matrix: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
*   **Difference from Dice:** Dice ignores True Negatives (the black background). MCC includes them. This makes MCC a more "balanced" metric for imbalanced datasets.

### 2. Why use it?
*   **Gradient Diversity:** The gradient signal provided by MCC is mathematically very different from Dice.
*   **Convergence:** When a model gets stuck in a local minimum using Dice (e.g., it learns a general blob shape but stops improving), MCC can often provide the "push" needed to refine the shape further because it creates a steeper slope in the loss landscape as the correlation improves.

### 3. Expected Outcome
*   **Goal:** Robustness.
*   **Hypothesis:** This version produces the most "statistically correct" masks. It is less likely to produce wild false positives (artifacts) than the Tversky-based models because MCC penalizes False Positives heavily if they ruin the correlation.

---

# Log:
98.5s	82	[0mCaching Data V3.6...
530.0s	83	
530.0s	84	========== FOLD 0 STARTED ==========
2078.4s	85	Fold 0 | Epoch 1 | Train Loss: 0.2560 | Val Dice: 0.9420
2078.4s	86	--> Best Model Saved! (0.9420)
3571.0s	87	Fold 0 | Epoch 2 | Train Loss: 0.0502 | Val Dice: 0.9120
5068.2s	88	Fold 0 | Epoch 3 | Train Loss: 0.0414 | Val Dice: 0.9498
5068.2s	89	--> Best Model Saved! (0.9498)
6572.0s	90	Fold 0 | Epoch 4 | Train Loss: 0.0716 | Val Dice: 0.1923
8066.3s	91	Fold 0 | Epoch 5 | Train Loss: 0.4758 | Val Dice: 0.8173
9568.9s	92	Fold 0 | Epoch 6 | Train Loss: 0.6635 | Val Dice: 0.4123
11059.6s	93	Fold 0 | Epoch 7 | Train Loss: 0.6808 | Val Dice: 0.4500
12567.3s	94	Fold 0 | Epoch 8 | Train Loss: 0.6817 | Val Dice: 0.4736
14054.6s	95	Fold 0 | Epoch 9 | Train Loss: 0.6821 | Val Dice: 0.4578
15564.2s	96	Fold 0 | Epoch 10 | Train Loss: 0.6812 | Val Dice: 0.4459
17051.7s	97	Fold 0 | Epoch 11 | Train Loss: 0.6807 | Val Dice: 0.4707
18562.9s	98	Fold 0 | Epoch 12 | Train Loss: 0.6820 | Val Dice: 0.4643
20054.9s	99	Fold 0 | Epoch 13 | Train Loss: 0.6816 | Val Dice: 0.4669
21566.1s	100	Fold 0 | Epoch 14 | Train Loss: 0.6827 | Val Dice: 0.4729
23057.5s	101	Fold 0 | Epoch 15 | Train Loss: 0.6818 | Val Dice: 0.4741
23062.7s	102	
23062.7s	103	--- Starting Inference ---
23675.3s	104	Submission Saved.