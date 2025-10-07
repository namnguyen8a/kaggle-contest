# **Phân vùng Tổn thương Da - [AIMA Challenge]**

**Kết quả tốt nhất trên Public Leaderboard: 0.91557**

## **1. Tổng quan về Cuộc thi & Dữ liệu**

-   **Nhiệm vụ:** Binary Semantic Segmentation. Xây dựng một mô hình Trí tuệ Nhân tạo để tạo ra mặt nạ (mask) nhị phân, trong đó pixel màu trắng đánh dấu vùng tổn thương và pixel màu đen là da lành.
-   **Dữ liệu:**
    -   **Train:** 2595 ảnh da liễu (.jpg) và mặt nạ tương ứng (.png) với nhiều độ phân giải khác nhau.
    -   **Test:** 1100 ảnh không có nhãn.
-   **Thước đo:** Dice Coefficient (Dice Score).
-   **Định dạng nộp bài:** Run-Length Encoding (RLE) của mask dự đoán, sau khi đã được resize về kích thước 512x512.

## **2. Phương pháp luận Chung**

Toàn bộ quá trình được xây dựng dựa trên một phương pháp luận lặp đi lặp lại: **Thử nghiệm -> Phân tích -> Cải tiến**. Chiến lược cốt lõi đã chứng minh hiệu quả bao gồm:

1.  **K-Fold Cross-Validation:** Thay vì chỉ chia dữ liệu một lần, toàn bộ tập train được chia thành 5 phần (folds). Quá trình training được lặp lại 5 lần, mỗi lần sử dụng 4 phần để huấn luyện và 1 phần để kiểm tra. Điều này giúp tận dụng tối đa dữ liệu và tạo ra các mô hình mạnh mẽ, ổn định.
2.  **Ensemble Learning:** Kết quả cuối cùng không dựa trên một mô hình duy nhất. Thay vào đó, các mô hình tốt nhất từ mỗi fold được kết hợp lại (bằng cách lấy trung bình dự đoán) để tạo ra một quyết định tổng hợp, giúp giảm lỗi và tăng độ chính xác.
3.  **Test Time Augmentation (TTA):** Ở khâu dự đoán, không chỉ ảnh gốc được sử dụng. Nhiều phiên bản biến đổi của ảnh test (lật, xoay, thay đổi kích thước) được đưa vào mô hình. Kết quả từ tất cả các phiên bản này được tổng hợp lại để tạo ra dự đoán cuối cùng, cực kỳ mạnh mẽ và ổn định.
---

## **3. Hai Giải pháp Tốt nhất được chọn cho Private Leaderboard**

Dưới đây là mô tả chi tiết về hai submission đã được chọn để đánh giá cuối cùng trên tập dữ liệu ẩn.

### **Submission 1: `submission_V-FINAL_GrandEnsemble_thresh0.6.csv`**

-   **Public Score:** **0.91557**
-   **Private Score:** `0.89965`

#### **A. Phân tích Chi tiết Hướng tiếp cận**

Đây là phương pháp **"Grand Ensemble"**, kết hợp 10 model từ 2 phiên bản training khác nhau (V8.1 và V9.1).

1.  **Bộ Model 1 (Từ V8.1): 5 model `Unet` với backbone `EfficientNet-B4`.**
    -   Được huấn luyện ở kích thước 512x512, `EPOCHS=25`, `BATCH_SIZE=12`.
    -   Sử dụng `AdvancedLoss` (Dice+Focal+Tversky).
    -   Đã chứng tỏ sự ổn định và hiệu năng cao.
2.  **Bộ Model 2 (Từ V9.1): 5 model `UnetPlusPlus` với backbone `timm-efficientnet-b5`.**
    -   Được huấn luyện ở kích thước 512x512, `EPOCHS=20`, `BATCH_SIZE=6`.
    -   Sử dụng kiến trúc `Unet++` tiên tiến hơn và backbone `B5` mạnh mẽ hơn.
3.  **Chiến lược Suy luận:**
    -   **Ultimate TTA:** Áp dụng TTA đa kích thước và đa biến thể (`Multi-Scale` & `Multi-Transform`) cho **từng bộ model riêng biệt**.
    -   **Grand Ensemble:** Lấy kết quả cuối cùng của bộ model V8 và V9, sau đó **lấy trung bình cộng** một lần nữa để ra quyết định cuối cùng.
    -   **Thresholding & Post-processing:** Sử dụng ngưỡng tối ưu `0.6` và các kỹ thuật hậu xử lý (làm mịn, lấp lỗ, xóa nhiễu).

#### **B. Mã nguồn Tham chiếu (Grand Ensemble)**
<details>
<summary>Click để xem Mã nguồn Suy luận Grand Ensemble</summary>

```python
# ====================================================================
# SKIN SEGMENTATION V-FINAL - GRAND ENSEMBLE (V8 + V9)
# ====================================================================
# PHẦN 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH
print(">>> [PHẦN 1] Bắt đầu cài đặt...")
!pip install -q segmentation-models-pytorch albumentations timm scikit-image scikit-learn
import os, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

class Config:
    IMAGE_SIZE = 512
    N_SPLITS = 5
    TTA_SCALES = [480, 512, 640]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_PATH = "/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation"
    TEST_IMG_PATH = os.path.join(DATASET_PATH, "Test/Test/Image")
    
    # Cần cập nhật các đường dẫn này cho đúng với output notebooks của bạn
    MODEL_V8_DIR = "/kaggle/input/best-model-5-fold-v8"
    MODEL_V9_DIR = "/kaggle/input/best-model-v9"
    
    V8_CONFIG = {'arch': 'Unet', 'encoder': 'efficientnet-b4'}
    V9_CONFIG = {'arch': 'UnetPlusPlus', 'encoder': 'timm-efficientnet-b5'}
cfg = Config()

# PHẦN 2: CÁC HÀM TIỆN ÍCH
def mask2rle(mask):
    pixels = mask.flatten(); pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1; runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def create_model(arch, encoder):
    return smp.create_model(arch, encoder_name=encoder, encoder_weights='imagenet', in_channels=3, classes=1)
def predict_with_ultimate_tta(models, image_np):
    final_predictions = []
    for scale in cfg.TTA_SCALES:
        transform_scale = A.Compose([A.Resize(scale, scale), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        scaled_tensor = transform_scale(image=image_np)['image'].unsqueeze(0).to(cfg.DEVICE)
        transforms = [lambda x: x, lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-2]), lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, 3, [-2, -1])]
        reverse_transforms = [lambda x: x, lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-2]), lambda x: torch.rot90(x, -1, [-2, -1]), lambda x: torch.rot90(x, -3, [-2, -1])]
        tta_preds_for_scale = []
        with torch.no_grad(), autocast():
            for transform, reverse_transform in zip(transforms, reverse_transforms):
                aug_tensor = transform(scaled_tensor)
                fold_preds = [torch.sigmoid(model(aug_tensor)) for model in models]
                ensembled_pred = torch.stack(fold_preds).mean(0)
                tta_preds_for_scale.append(reverse_transform(ensembled_pred))
        avg_pred_for_scale = torch.stack(tta_preds_for_scale).mean(0)
        restored_pred = F.interpolate(avg_pred_for_scale, size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), mode='bilinear', align_corners=False)
        final_predictions.append(restored_pred)
    return torch.stack(final_predictions).mean(0)
def advanced_postprocess(mask, min_size=100):
    binary_mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(3))
    return binary_mask.astype(np.uint8)

# PHẦN 3: PIPELINE SUY LUẬN "GRAND ENSEMBLE"
models_v8 = []
for fold in range(1, cfg.N_SPLITS + 1):
    model = create_model(cfg.V8_CONFIG['arch'], cfg.V8_CONFIG['encoder']).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_V8_DIR, f"best_model_fold_{fold}.pth")))
    model.eval(); models_v8.append(model)
print(f"✅ Đã tải thành công {len(models_v8)} models V8.")
models_v9 = []
for fold in range(1, cfg.N_SPLITS + 1):
    model = create_model(cfg.V9_CONFIG['arch'], cfg.V9_CONFIG['encoder']).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_V9_DIR, f"best_model_fold_{fold}.pth")))
    model.eval(); models_v9.append(model)
print(f"✅ Đã tải thành công {len(models_v9)} models V9.")
test_files = [f for f in os.listdir(cfg.TEST_IMG_PATH) if f.endswith('.jpg')]
test_ids = [os.path.splitext(f)[0] for f in test_files]
results = []
OPTIMAL_THRESHOLD = 0.6
for test_id in tqdm(test_ids, desc=f"Grand Ensemble Inference"):
    img_path = os.path.join(cfg.TEST_IMG_PATH, f"{test_id}.jpg")
    image_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    pred_v8 = predict_with_ultimate_tta(models_v8, image_numpy)
    pred_v9 = predict_with_ultimate_tta(models_v9, image_numpy)
    final_pred_tensor = (pred_v8 + pred_v9) / 2.0
    final_mask_np = final_pred_tensor.cpu().numpy().squeeze()
    binary_mask_with_threshold = (final_mask_np > OPTIMAL_THRESHOLD)
    final_mask_processed = advanced_postprocess(binary_mask_with_threshold)
    rle = mask2rle(final_mask_processed)
    results.append({"ID": f"{test_id}_segmentation", "Predicted_Mask": rle})
submission_df = pd.DataFrame(results)
submission_filename = f"submission_V-FINAL_GrandEnsemble_thresh{OPTIMAL_THRESHOLD}.csv"
submission_df.to_csv(submission_filename, index=False)
```

</details>

---

### **Submission 2: `submission_v8.3_ultimateTTA_thresh0.6.csv`**

-   **Public Score:** **0.91447**
-   **Private Score:** `0.90117`
#### **A. Phân tích Chi tiết Hướng tiếp cận**

1.  **Bộ Model (Từ V8.1):** Sử dụng 5 model `Unet` với backbone `EfficientNet-B4`. Đây là bộ model được huấn luyện với các thông số cân bằng, đảm bảo hoàn thành trong thời gian cho phép và có chất lượng cao.
2.  **Chiến lược Suy luận (Ultimate TTA):**
    -   **Ensemble:** Tổ hợp kết quả từ 5 model đã huấn luyện.
    -   **Multi-Scale & Multi-Transform TTA:** Mỗi ảnh test được dự đoán 15 lần (`3 scales * 5 transforms`), và mỗi lần dự đoán lại là kết quả ensemble của 5 model. Kỹ thuật này giúp mô hình "nhìn" ảnh ở nhiều bối cảnh và góc độ khác nhau, tạo ra một dự đoán tổng hợp cực kỳ mạnh mẽ.
    -   **Thresholding & Post-processing:** Sử dụng ngưỡng `0.6` và các kỹ thuật hậu xử lý.

#### **B. Mã nguồn Tham chiếu**

<details>
<summary>Click để xem Mã nguồn Phiên bản 8</summary>

```python
# ====================================================================
# SKIN SEGMENTATION V8.1 (TRAINING) & V8.3 (INFERENCE)
# ====================================================================

# --------------------------------------------------------------------
# PHẦN 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH (CHO TRAINING)
# --------------------------------------------------------------------
print(">>> [PHẦN 1] Bắt đầu cài đặt thư viện và cấu hình...")
!pip install -q segmentation-models-pytorch albumentations timm scikit-image scikit-learn
import os, numpy as np, pandas as pd, cv2, gc, torch, torch.nn as nn, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

class Config:
    ARCHITECTURE = 'Unet'
    ENCODER = 'efficientnet-b4'
    PRETRAINED_WEIGHTS = 'imagenet'
    IMAGE_SIZE = 512
    N_SPLITS = 5
    BATCH_SIZE = 12
    EPOCHS = 25
    PATIENCE = 5
    MIN_DELTA = 1e-4
    LEARNING_RATE = 1e-4
    GRADIENT_CHECKPOINTING = True
    TTA_SCALES = [480, 512, 640]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_PATH = "/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation"
    TRAIN_IMG_PATH = os.path.join(BASE_PATH, "Train/Train/Image")
    TRAIN_MASK_PATH = os.path.join(BASE_PATH, "Train/Train/Mask")
    TEST_IMG_PATH = os.path.join(BASE_PATH, "Test/Test/Image")
    MODEL_OUTPUT_DIR = "/kaggle/working/models_v8.1/"

cfg = Config()
os.makedirs(cfg.MODEL_OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# PHẦN 2: CÁC HÀM TIỆN ÍCH, LOSS, AUGMENTATION, TTA
# --------------------------------------------------------------------
def mask2rle(mask):
    pixels = mask.flatten(); pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1; runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
class AdvancedLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.focal = smp.losses.FocalLoss(smp.losses.BINARY_MODE, alpha=0.25, gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, alpha=0.7, beta=0.3)
    def forward(self, pred, target): return 0.4*self.dice(pred, target) + 0.3*self.focal(pred, target) + 0.3*self.tversky(pred, target)
def get_transforms_v8():
    train_transform = A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ShiftScaleRotate(p=0.5),
        A.OneOf([A.ElasticTransform(p=0.2), A.GridDistortion(p=0.2)], p=0.3),
        A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.3), A.CLAHE(p=0.4),
        A.OneOf([A.GaussNoise(), A.CoarseDropout(max_holes=8, max_height=32, max_width=32)], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
    ])
    val_transform = A.Compose([A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    return train_transform, val_transform
def predict_with_ultimate_tta(models, image_np):
    final_predictions = []
    for scale in cfg.TTA_SCALES:
        transform_scale = A.Compose([A.Resize(scale, scale), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        scaled_tensor = transform_scale(image=image_np)['image'].unsqueeze(0).to(cfg.DEVICE)
        transforms = [lambda x: x, lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-2]), lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, 3, [-2, -1])]
        reverse_transforms = [lambda x: x, lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-2]), lambda x: torch.rot90(x, -1, [-2, -1]), lambda x: torch.rot90(x, -3, [-2, -1])]
        tta_preds_for_scale = []
        with torch.no_grad(), autocast():
            for transform, reverse_transform in zip(transforms, reverse_transforms):
                aug_tensor = transform(scaled_tensor)
                fold_preds = [torch.sigmoid(model(aug_tensor)) for model in models]
                ensembled_pred = torch.stack(fold_preds).mean(0)
                tta_preds_for_scale.append(reverse_transform(ensembled_pred))
        avg_pred_for_scale = torch.stack(tta_preds_for_scale).mean(0)
        restored_pred = F.interpolate(avg_pred_for_scale, size=(512, 512), mode='bilinear', align_corners=False)
        final_predictions.append(restored_pred)
    return torch.stack(final_predictions).mean(0)
def advanced_postprocess(mask, min_size=100):
    binary_mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(3))
    return binary_mask.astype(np.uint8)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience, self.min_delta, self.counter, self.best_score = patience, min_delta, 0, None
        self.best_weights = None
    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score, self.counter = val_score, 0; self.best_weights = model.state_dict().copy(); return False
        else: self.counter += 1; return self.counter >= self.patience
class SkinLesionDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir, transform=None):
        self.ids, self.img_dir, self.mask_dir, self.transform = ids, img_dir, mask_dir, transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]; img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)
        if self.transform: augmented = self.transform(image=image, mask=mask); image, mask = augmented['image'], augmented['mask']
        return image, mask.unsqueeze(0)
def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float(); intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def validate_model(model, loader, device):
    model.eval(); total_dice = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with autocast(): preds = torch.sigmoid(model(images))
            for pred, mask in zip(preds, masks): total_dice += dice_coefficient(pred, mask).item()
    return total_dice / len(loader.dataset)
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device):
    model.train(); total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(): loss = loss_fn(model(images), masks)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if scheduler is not None: scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)
def create_model():
    model = smp.Unet(encoder_name=cfg.ENCODER, encoder_weights=cfg.PRETRAINED_WEIGHTS, in_channels=3, classes=1, activation=None)
    if cfg.GRADIENT_CHECKPOINTING:
        try:
            if hasattr(model.encoder, 'set_grad_checkpointing'): model.encoder.set_grad_checkpointing(enable=True); print("✅ Gradient checkpointing enabled.")
        except Exception as e: print(f"⚠️ Không thể kích hoạt gradient checkpointing: {e}")
    return model

# --------------------------------------------------------------------
# PHẦN 3: PIPELINE HUẤN LUYỆN K-FOLD
# --------------------------------------------------------------------
print(">>> [PHẦN 3] Bắt đầu pipeline huấn luyện K-Fold...")
START_FOLD = 0 
all_files = [f for f in os.listdir(cfg.TRAIN_IMG_PATH) if f.endswith('.jpg')]
all_ids = [os.path.splitext(f)[0] for f in all_files]
kf = KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=42)
train_transform, val_transform = get_transforms_v8()
full_dataset = SkinLesionDataset(all_ids, cfg.TRAIN_IMG_PATH, cfg.TRAIN_MASK_PATH, transform=train_transform)
val_dataset_template = SkinLesionDataset(all_ids, cfg.TRAIN_IMG_PATH, cfg.TRAIN_MASK_PATH, transform=val_transform)
overall_val_dice = 0.0
all_splits = list(kf.split(all_ids))
for fold in range(START_FOLD, cfg.N_SPLITS):
    print(f"\n{'='*25} FOLD {fold+1}/{cfg.N_SPLITS} {'='*25}")
    train_idx, val_idx = all_splits[fold]
    train_subset, val_subset = Subset(full_dataset, train_idx), Subset(val_dataset_template, val_idx)
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
    model = create_model().to(cfg.DEVICE)
    loss_fn = AdvancedLoss().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.LEARNING_RATE*5, epochs=cfg.EPOCHS, steps_per_epoch=len(train_loader))
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA)
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, cfg.DEVICE)
        val_dice = validate_model(model, val_loader, cfg.DEVICE)
        print(f"Fold {fold+1} Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        if early_stopper(val_dice, model): print(f"⏰ Early stopping! Best Dice: {early_stopper.best_score:.4f}"); break
    model.load_state_dict(early_stopper.best_weights)
    torch.save(model.state_dict(), os.path.join(cfg.MODEL_OUTPUT_DIR, f"best_model_fold_{fold+1}.pth"))
    print(f"🎉 Đã lưu model tốt nhất cho Fold {fold+1} với Dice: {early_stopper.best_score:.4f}")
    overall_val_dice += early_stopper.best_score
    gc.collect(); torch.cuda.empty_cache()
print(f"\n✅ Training K-Fold V8.1 hoàn tất! Điểm Dice trung bình: {overall_val_dice/cfg.N_SPLITS:.4f}")

# --------------------------------------------------------------------
# PHẦN 4: PIPELINE SUY LUẬN TỐI THƯỢNG (TƯƠNG ĐƯƠNG V8.3)
# --------------------------------------------------------------------
print("\n>>> [PHẦN 4] Bắt đầu pipeline suy luận tối thượng...")
fold_models = []
for fold in range(1, cfg.N_SPLITS + 1):
    model_path = os.path.join(cfg.MODEL_OUTPUT_DIR, f"best_model_fold_{fold}.pth")
    if os.path.exists(model_path):
        model = create_model().to(cfg.DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval(); fold_models.append(model)
if fold_models:
    test_files = [f for f in os.listdir(cfg.TEST_IMG_PATH) if f.endswith('.jpg')]
    test_ids = [os.path.splitext(f)[0] for f in test_files]
    results_v8 = []
    OPTIMAL_THRESHOLD = 0.6
    for test_id in tqdm(test_ids, desc="Ultimate Inference V8.3"):
        img_path = os.path.join(cfg.TEST_IMG_PATH, f"{test_id}.jpg")
        image_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        final_pred_tensor = predict_with_ultimate_tta(fold_models, image_numpy)
        final_mask_np = final_pred_tensor.cpu().numpy().squeeze()
        binary_mask_with_threshold = (final_mask_np > OPTIMAL_THRESHOLD)
        final_mask_processed = advanced_postprocess(binary_mask_with_threshold)
        rle = mask2rle(final_mask_processed)
        results_v8.append({"ID": f"{test_id}_segmentation", "Predicted_Mask": rle})
    submission_df_v8 = pd.DataFrame(results_v8)
    submission_filename = f"submission_v8.3_ultimateTTA_thresh{OPTIMAL_THRESHOLD}.csv"
    submission_df_v8.to_csv(submission_filename, index=False)
```

</details>