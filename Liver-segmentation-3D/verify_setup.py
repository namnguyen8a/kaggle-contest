import os
import sys
import glob

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name} imported successfully.")
        return True
    except ImportError:
        print(f"[FAIL] {module_name} is missing. Please install it.")
        return False

def check_path(path, description):
    if os.path.exists(path):
        print(f"[OK] {description} found at: {path}")
        return True
    else:
        print(f"[FAIL] {description} NOT found at: {path}")
        return False

def main():
    print("--- Verifying Setup ---")
    
    # 1. Check Libraries
    modules = ['numpy', 'torch', 'nibabel', 'albumentations', 'tqdm', 'pandas', 'matplotlib']
    all_modules_ok = all([check_import(m) for m in modules])
    
    if not all_modules_ok:
        print("\n[CRITICAL] Some libraries are missing. Run: pip install numpy torch nibabel albumentations tqdm pandas matplotlib")
    
    # 2. Check Data
    data_root = 'f:/kaggle-dataset/aio2025liverseg'
    train_vol = os.path.join(data_root, 'train', 'volume')
    train_seg = os.path.join(data_root, 'train', 'segmentation')
    test_dir = os.path.join(data_root, 'test')
    
    paths_ok = True
    paths_ok &= check_path(train_vol, "Train Volumes")
    paths_ok &= check_path(train_seg, "Train Segmentations")
    paths_ok &= check_path(test_dir, "Test Directory")
    
    if paths_ok:
        # Check count
        n_train = len(glob.glob(os.path.join(train_vol, '*.nii')))
        n_test = len(glob.glob(os.path.join(test_dir, '*.nii')))
        print(f"\nFound {n_train} training volumes.")
        print(f"Found {n_test} test volumes.")
        
        if n_train == 0 or n_test == 0:
            print("[WARNING] Dataset seems empty?")
    
    print("\n--- Verification Complete ---")
    if all_modules_ok and paths_ok:
        print("You are ready to run 'python train.py'!")
    else:
        print("Please fix the issues above before proceeding.")

if __name__ == '__main__':
    main()
