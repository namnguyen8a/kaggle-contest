Here is the detailed technical breakdown of all 12 versions in a structured comparison table.

### 📋 Experiment Master Table

| Version | **Model Architecture** | **Data Processing (Input)** | **Normalization Strategy** | **Augmentations** | **Loss Function** | **Inference Method** | **Status / Problem** | **Final Score** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | **U-Net**<br>Encoder: EffNet-B0 | **2D**<br>(1 Slice, 1 Channel) | **MinMax**<br>`(vol-min)/(max-min)`<br>Range: [0,1] | **Standard**<br>Resize 512, HFlip, VFlip, Rotate (10) | **BCE + Dice**<br>(50/50) | **TTA**<br>(Horizontal Flip) | ✅ Success<br>Solid Baseline. | **0.95766** |
| **v2** | **U-Net++**<br>Encoder: EffNet-B4 | **2D**<br>(1 Slice, 1 Channel) | **Formula**<br>`(vol+100)/300`<br>Range: [0,1] | **Standard**<br>Resize 512, HFlip, VFlip, Rotate (15) | **BCE + Dice**<br>(50/50) | **TTA**<br>(Horizontal Flip) | ⚠️ **Timeout/Stuck**<br>Heavy model, training stopped at Ep 8, but performed best. | **0.96800**<br>(Best) |
| **v3** | **U-Net**<br>Encoder: EffNet-B0 | **2.5D**<br>(Stack: Prev, Curr, Next)<br>3 Channels | **Formula**<br>`(vol+100)/300` | **Standard**<br>Resize 512, HFlip, VFlip, Rotate (10) | **BCE + Dice**<br>(50/50) | **TTA**<br>(Horizontal Flip) | ✅ Success<br>Multiprocessing crash fixed by `num_workers=0`. | 0.95840 |
| **v3.1** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Focal Loss** | N/A | ❌ **Failed**<br>Loss became `NaN`. Focal Loss is unstable with AMP (Float16). | Failed |
| **v3.2** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Tversky Loss**<br>(Alpha 0.3, Beta 0.7) | **TTA** | ✅ Success<br>Very stable training. High Recall. | **0.96790**<br>(2nd Best) |
| **v3.3** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **BCE + Lovasz**<br>(50/50) | **TTA** | ✅ Success<br>Slow convergence, but good generalization. | 0.96486 |
| **v3.4** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Weighted BCE + Dice**<br>(PosWeight=5.0) | **TTA** | ⚠️ **Unstable**<br>Loss exploded to `NaN`/`Inf` at Epoch 13. | 0.95981 |
| **v3.5** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Focal Tversky**<br>(Gamma 0.75) | **TTA** | ✅ Success<br>Stable, but no gain over pure Tversky. | 0.96107 |
| **v3.6** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Dice + MCC**<br>(Matthews Corr. Coeff) | **TTA** | ❌ **Collapsed**<br>Model weights corrupted after Epoch 3. Val Score dropped to 0.47. | 0.95358 |
| **v4** | **U-Net**<br>Encoder: **ResNet34** | **2D**<br>(1 Slice, 1 Channel) | Formula | **Heavy**<br>GridDistortion, ElasticTransform | **BCE + Dice**<br>(50/50) | **TTA + LCC**<br>(Keep Largest Component) | ✅ Success<br>Heavy augs likely confused the model (Underfitting). | 0.95197 |
| **v5** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **LogCosh Dice**<br>(Smoothed Dice) | **TTA** | ✅ Success<br>Very smooth loss curve, average performance. | 0.96073 |
| **v6** | U-Net<br>Encoder: EffNet-B0 | **2.5D**<br>3 Channels | Formula | Standard | **Tversky + Lovasz**<br>(0.6 / 0.4) | **TTA** | ✅ Success<br>Highest Val Score (0.98), slight overfitting to Fold 0. | **0.96628**<br>(3rd Best) |

---

### 🔍 Key Differences Explained

#### **1. Data Processing (2D vs 2.5D)**
*   **2D (v1, v2, v4):** The model sees one slice at a time. It doesn't know if the liver is getting bigger or smaller in the next slice.
*   **2.5D (v3, v5, v6):** The model sees the **Previous** and **Next** slices as extra color channels (RGB). This gives it "3D context" without the heavy memory cost of full 3D training.
    *   *Result:* 2.5D generally improved stability (v3.2 vs v1), but **v2** proved that a bigger model (EffNet-B4) on 2D data is still more powerful than a small model (EffNet-B0) on 2.5D data.

#### **2. Normalization Strategy**
*   **v1 (MinMax):** `(x - min) / (max - min)`. This is risky if there is metal or bone artifacts (super bright pixels), which squeezes the liver contrast into a tiny range.
*   **v2-v6 (Formula):** `(x - (-100)) / (200 - (-100))`. This hard-clips the intensity to the specific Hounsfield Unit range of the liver. This is the **medical standard** and much better.

#### **3. Loss Function Evolution**
*   **BCE+Dice:** The standard baseline. Good, but struggles if the liver is very small.
*   **Tversky (v3.2):** Penalizes False Negatives (missing liver pixels) more than False Positives. This boosted the score significantly (0.9679).
*   **Lovasz (v3.3, v6):** Directly optimizes the IoU metric used in competition. Combined with Tversky (v6), it achieved the highest validation accuracy.
*   **Focal/Weighted (v3.1, v3.4):** These try to force the model to look at hard pixels using exponents ($loss^\gamma$). This caused mathematical overflows (NaN) because the numbers became too large for the GPU's Float16 precision.

#### **4. Inference Method**
*   **TTA (Test Time Augmentation):** Every version used this. The model predicts the image, then flips the image horizontally, predicts again, and averages the result. This usually adds +0.002 to +0.005 to the score.
*   **LCC (v4):** Keeps only the largest blob. This removes small noise artifacts outside the liver. It didn't help v4 much, likely because the underlying ResNet model wasn't accurate enough.

---

### 📉 Why Val 0.98 but Leaderboard 0.96?

In **Version 6**, you saw:
*   **Validation Dice:** 0.9820
*   **Leaderboard Score:** 0.9662

**The Reason:**
1.  **Metric Definition Difference:**
    *   Your logs calculate the average Dice of **2D slices**.
    *   Many slices in a CT scan are empty (black). Predicting "all black" on an empty slice gives a Dice of **1.0**. These perfect 1.0s inflate your average in the logs.
    *   The Leaderboard calculates the Dice of the **3D Volume**. It ignores the empty slices that make the score look good and focuses on the actual liver volume.

2.  **Overfitting Fold 0:**
    *   You are training on the *same* 20% of patients (Fold 0).
    *   By Version 6, you have tuned the loss function (Tversky+Lovasz) perfectly for *those specific patients*.
    *   When the model sees new patients (Test Set), it struggles slightly because it optimized too much for the Fold 0 patients.
    *   *Fix:* You must train on Folds 1, 2, 3, and 4 and average the models to fix this.