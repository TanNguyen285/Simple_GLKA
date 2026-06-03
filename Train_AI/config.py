import os
import torch

class Config:
    # =====================================================================
    # 1. PATH CONFIGURATION
    # =====================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    
    # Logic tự sinh thư mục dạng runs/exp1, runs/exp2... mang từ train.py sang
    RUNS_BASE_DIR = os.path.join(BASE_DIR, "runs")
    
    # =====================================================================
    # 2. TRAINING HYPERPARAMETERS
    # =====================================================================
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # =====================================================================
    # 3. HARDWARE CONFIGURATION
    # =====================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8
    PIN_MEMORY = True

    # =====================================================================
    # 4. DATA AUGMENTATION CONFIGURATION
    # =====================================================================
    CROP_SCALE = (0.8, 1.0)
    CROP_RATIO = (0.75, 1.33)
    FLIP_PROB = 0.5
    ERASING_PROB = 0.3
    
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


config = Config()

# Thực thi logic tự động tạo thư mục exp khi khởi tạo cấu hình
if not os.path.exists(config.RUNS_BASE_DIR):
    os.makedirs(config.RUNS_BASE_DIR)

exp_num = 1
while True:
    SAVE_DIR = os.path.join(config.RUNS_BASE_DIR, f'exp{exp_num}')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[*] Đã tạo thư mục lưu kết quả tự động: {SAVE_DIR}")
        config.SAVE_DIR = SAVE_DIR  # Gán thẳng đường dẫn cụ thể vào config để file khác sài
        break
    exp_num += 1