# Ref:
- https://www.kaggle.com/code/nnguyen26/aio2025-warm-up-liver-tumor-segmentation-v3-1/edit (training)

## 📊 Version 3.1: The "Focus" Strategy
**Loss Function:** `Dice Loss + Focal Loss`

### 1. The Logic
Standard Cross-Entropy (BCE) treats every pixel equally. In a CT scan, 95% of the pixels are black background (easy) and 5% are liver (hard). The model can achieve 95% accuracy by just predicting "Background" everywhere.
*   **Focal Loss** introduces a modulating factor $(1 - p_t)^\gamma$.
*   If the model is confident ($p_t \approx 1$), the loss drops to near zero.
*   If the model is struggling (edges, small tumors), the loss remains high.

### 2. Why use it?
*   It stops the massive number of background pixels from overwhelming the gradients.
*   It forces the model to learn the **"Hard Examples"**—usually the tricky boundaries between the liver and the stomach, or small disjointed sections of the liver.

### 3. Expected Outcome
*   **Pros:** Better detection of small details; sharper definition in low-contrast areas.
*   **Cons:** Might be slightly unstable in early epochs if the learning rate is too high.

## Log:
92.8s	102	[0mCaching Data V3.1...
541.7s	103	
541.8s	104	========== FOLD 0 STARTED ==========
2089.0s	105	Fold 0 | Epoch 1/15 | Train Loss: nan | Val Dice: 0.5289
2089.0s	106	--> Best Model Saved! (0.5289)
3567.8s	107	Fold 0 | Epoch 2/15 | Train Loss: nan | Val Dice: 0.5289
5056.7s	108	Fold 0 | Epoch 3/15 | Train Loss: nan | Val Dice: 0.5289
6528.3s	109	Fold 0 | Epoch 4/15 | Train Loss: nan | Val Dice: 0.5289
7998.3s	110	Fold 0 | Epoch 5/15 | Train Loss: nan | Val Dice: 0.5289
89.9s	111	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
89.9s	112	  warnings.warn(
89.9s	113	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
89.9s	114	  warnings.warn(
89.9s	115	
89.9s	116	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
89.9s	117	  warnings.warn(
89.9s	118	/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
89.9s	119	  warnings.warn(

## Estimated problem:
- The crash you experienced with Dice + Focal Loss (Train Loss: nan) happens because Focal Loss is mathematically unstable when using Mixed Precision (FP16) training. It uses exponents and logarithms; if the model predicts 0.0000001 or 0.9999999, the math explodes to Infinity or NaN, killing your training instantly.