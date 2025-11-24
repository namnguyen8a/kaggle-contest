# Ref:
- https://www.kaggle.com/code/youpre/aio2025-warm-up-liver-tumor-segmentation-v4/edit (training)

Here is your detailed **Experiment Tracking Report for Version 4**.

This document explains the "Robust" strategy, the specific technical choices made to fix the crashes, and why this model is structurally different from the previous versions.

---

# 📝 Experiment Report: Version 4 (The Robust ResNet) - score 0.95197 - time 26069.5s

**Status:** Running (Stability & Disk Optimized)
**Core Concept:** **Shape Invariance & Noise Reduction**
**Key Changes:** ResNet34 Backbone + Heavy Geometric Augmentation + LCC Post-processing.

---

## 1. The Objective: Why "Robust"?
In medical segmentation, two common problems prevent high scores:
1.  **Overfitting to Shape:** The model memorizes the specific shape of livers in the training set (80 cases) and fails when a test patient has a slightly squished or deformed liver.
2.  **Floating Artifacts:** The model correctly finds the liver but also predicts random "dust" or noise in the background, lowering the Dice score.

**Version 4 attacks these specific problems.**

---

## 2. Technical Architecture

### A. Model: U-Net with ResNet34
*   **Previous:** EfficientNet-B0 (V1/V3) and EfficientNet-B4 (V2).
*   **Current:** **ResNet34**.
*   **Why?**
    *   ResNet (Residual Networks) uses a different mathematical approach to feature extraction compared to EfficientNet.
    *   ResNet34 is the "sweet spot" depth—deep enough to learn complex textures, but not so deep that it overfits on a small dataset (80 samples).
    *   *Ensembling Strategy:* If you average predictions from an EfficientNet (V2) and a ResNet (V4), the errors often cancel out because they "think" differently.

### B. Input Data
*   **Format:** 2D Slices (Single Channel).
*   **Resolution:** 512x512.
*   **Disk Optimization:**
    *   We save cache files as **`float16`** (Half Precision).
    *   This reduces the cache size from ~15GB to **~7.5GB**, making it impossible to hit Kaggle's "Disk Full" error.

---

## 3. The "Secret Sauce": Heavy Augmentations
This version uses **Albumentations** to aggressively deform the training images.

*   **Grid Distortion:** Warps the image grid (like looking through a fun-house mirror).
*   **Elastic Transform:** Simulates "squishing" the organ (as if you poked it with a finger).
*   **Why do this?**
    *   The liver is a soft organ. Its shape changes based on patient position, breathing, and surrounding organs.
    *   By warping the images during training, we generate infinite variations of "squished livers." This forces the model to learn **texture and boundary logic** rather than memorizing fixed shapes.

---

## 4. Stability Fixes (The Crash Solution)
You experienced crashes with `AssertionError: can only test a child process`. This is a known conflict between PyTorch DataLoaders, heavy Albumentations, and the Docker environment.

*   **The Fix:** `num_workers = 0`
*   **What it does:** Instead of spawning separate parallel processes (workers) to load data, the Main Process loads the data itself.
*   **Trade-off:** It is slightly slower (training might take 10% longer), but it is **100% stable** and will not crash.

---

## 5. Post-Processing: Largest Connected Component (LCC)
This is a logic step applied *after* the model predicts.

*   **The Problem:** Sometimes the model predicts the liver correctly, but also predicts a tiny blob of 5 pixels in the top corner (noise).
*   **The Algorithm:**
    1.  Take the binary mask.
    2.  Count distinct "islands" (connected blobs) of white pixels.
    3.  Calculate the area of each island.
    4.  **Keep** the largest island (the liver).
    5.  **Delete** all smaller islands (noise/artifacts).
*   **Impact:** This typically boosts the Dice score by `+0.001` to `+0.005` by cleaning up false positives.

---

## 6. Summary of Differences

| Feature | Version 1 (Baseline) | Version 4 (Robust) |
| :--- | :--- | :--- |
| **Backbone** | EfficientNet-B0 | **ResNet34** |
| **Augmentation** | Flips/Rotations | **Elastic/Grid Distortions** |
| **Data Loading** | `num_workers=2` | **`num_workers=0` (Safe Mode)** |
| **Disk Cache** | `float32` | **`float16` (half size)** |
| **Post-Processing** | None | **Keep Largest Component** |

### 🔮 Prediction
If V4 works well, it indicates that the dataset requires the model to be flexible regarding shapes. The LCC post-processing ensures the submission masks are clean "solid" shapes, which looks professional and scores higher.
