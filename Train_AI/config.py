import os
import torch

class Config:
    # =====================================================================
    # 1. PATH CONFIGURATION
    # =====================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)  # Go up one level to CNN_edge folder
    DATA_DIR = os.path.join(PARENT_DIR, "dataset")
    
    # Logic tự sinh thư mục dạng runs/exp1, runs/exp2...
    RUNS_BASE_DIR = os.path.join(BASE_DIR, "runs")
    
    # =====================================================================
    # 2. MODEL HYPERPARAMETERS
    # =====================================================================
    NUM_CLASSES = 2
    
    # =====================================================================
    # 3. TRAINING HYPERPARAMETERS
    # =====================================================================
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 40
    
    # =====================================================================
    # 4. OPTIMIZER CONFIGURATION (ĐÃ FIX CHO CẢ SGD VÀ ADAMW)
    # =====================================================================
    OPTIMIZER_TYPE = "SGD"  # Bạn chỉ cần đổi giữa "SGD" hoặc "AdamW" ở đây
    _SGD_CONFIG = {
        "lr": 0.015,           # SGD cần learning rate lớn hơn
        "momentum": 0.9,
        "weight_decay": 5e-4  
    }
    
    _ADAMW_CONFIG = {
        "lr": 3e-4,          
        "weight_decay": 1e-2  
    }
    
    # Thuộc tính động: Tự động trả về cấu hình chuẩn dựa theo OPTIMIZER_TYPE
    @property
    def OPTIMIZER_CONFIG(self):
        if self.OPTIMIZER_TYPE == "SGD":
            return self._SGD_CONFIG
        elif self.OPTIMIZER_TYPE == "AdamW":
            return self._ADAMW_CONFIG
        else:
            raise ValueError(f"Unknown optimizer type: {self.OPTIMIZER_TYPE}")
    
    # =====================================================================
    # 5. SCHEDULER CONFIGURATION
    # =====================================================================
    SCHEDULER_TYPE = "CosineAnnealingLR"  # "CosineAnnealingLR", "StepLR"
    
    _COSINE_CONFIG = {
        "T_max": EPOCHS,
        "eta_min": 1e-6
    }
    
    _STEP_CONFIG = {
        "step_size": 7,
        "gamma": 0.1
    }
    
    # Thuộc tính động cho Scheduler
    @property
    def SCHEDULER_CONFIG(self):
        if self.SCHEDULER_TYPE == "CosineAnnealingLR":
            return self._COSINE_CONFIG
        elif self.SCHEDULER_TYPE == "StepLR":
            return self._STEP_CONFIG
        else:
            raise ValueError(f"Unknown scheduler type: {self.SCHEDULER_TYPE}")
    
    # =====================================================================
    # 6. HARDWARE CONFIGURATION
    # =====================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8
    PIN_MEMORY = True

    # =====================================================================
    # 7. DATA AUGMENTATION CONFIGURATION
    # =====================================================================
    CROP_SCALE = (0.8, 1.0)
    CROP_RATIO = (0.75, 1.33)
    FLIP_PROB = 0.5
    ERASING_PROB = 0.3
    
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


config = Config()

def create_save_dir():
    """Tạo thư mục lưu kết quả tự động (chỉ gọi một lần lúc khởi động)"""
    if not os.path.exists(config.RUNS_BASE_DIR):
        os.makedirs(config.RUNS_BASE_DIR)

    exp_num = 1
    while True:
        SAVE_DIR = os.path.join(config.RUNS_BASE_DIR, f'exp{exp_num}')
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            print(f"[*] Đã tạo thư mục lưu kết quả tự động: {SAVE_DIR}")
            config.SAVE_DIR = SAVE_DIR
            break
        exp_num += 1