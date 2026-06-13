import os
import torch


class Config:
    # =====================================================================
    # 1. PATH CONFIGURATION
    # =====================================================================
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR  = os.path.dirname(BASE_DIR)
    DATA_DIR    = os.path.join(PARENT_DIR, "dataset")   # dataset/train/ và dataset/val/
                                                          # mỗi class là 1 subfolder chứa *.mp4 / *.avi
    RUNS_BASE_DIR = os.path.join(BASE_DIR, "runs")

    # =====================================================================
    # 2. MODEL
    # =====================================================================
    NUM_CLASSES = 3   # Đổi theo số class của bạn

    # =====================================================================
    # 3. VIDEO / FRAME CONFIGURATION
    # =====================================================================
    VIDEO_DURATION   = 30       # giây — dùng để validate khi load
    CLIP_DURATION    = 30       # giây clip thực sự dùng (có thể < VIDEO_DURATION)
    FPS_SAMPLE       = 2        # số frame lấy mỗi giây  →  30s * 2fps = 60 frames/video
    IMG_SIZE         = 224

    # diff_mode: "subtract" (I_t - I_{t-1}) | "optical_flow" (Farneback)
    DIFF_MODE        = "subtract"

    # =====================================================================
    # 4. TRAINING HYPERPARAMETERS
    # =====================================================================
    BATCH_SIZE = 8              # Video nặng hơn ảnh — batch nhỏ hơn
    EPOCHS     = 30

    # =====================================================================
    # 5. OPTIMIZER
    # =====================================================================
    OPTIMIZER_TYPE = "AdamW"

    _SGD_CONFIG = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }
    _ADAMW_CONFIG = {
        "lr": 3e-4,
        "weight_decay": 1e-2,
    }

    @property
    def OPTIMIZER_CONFIG(self):
        if self.OPTIMIZER_TYPE == "SGD":
            return self._SGD_CONFIG
        elif self.OPTIMIZER_TYPE == "AdamW":
            return self._ADAMW_CONFIG
        raise ValueError(f"Unknown optimizer: {self.OPTIMIZER_TYPE}")

    # =====================================================================
    # 6. SCHEDULER
    # =====================================================================
    SCHEDULER_TYPE = "CosineAnnealingLR"

    _COSINE_CONFIG = {"T_max": EPOCHS, "eta_min": 1e-6}
    _STEP_CONFIG   = {"step_size": 7, "gamma": 0.1}

    @property
    def SCHEDULER_CONFIG(self):
        if self.SCHEDULER_TYPE == "CosineAnnealingLR":
            return self._COSINE_CONFIG
        elif self.SCHEDULER_TYPE == "StepLR":
            return self._STEP_CONFIG
        raise ValueError(f"Unknown scheduler: {self.SCHEDULER_TYPE}")

    # =====================================================================
    # 7. HARDWARE
    # =====================================================================
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4      # Video decode nặng — không nên dùng quá nhiều workers
    PIN_MEMORY  = True

    # =====================================================================
    # 8. AUGMENTATION (áp dụng trên từng frame)
    # =====================================================================
    CROP_SCALE    = (0.8, 1.0)
    CROP_RATIO    = (0.75, 1.33)
    FLIP_PROB     = 0.5
    ERASING_PROB  = 0.3

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD  = [0.229, 0.224, 0.225]


config = Config()


def create_save_dir():
    """Tạo thư mục lưu kết quả tự động runs/exp1, exp2, ..."""
    os.makedirs(config.RUNS_BASE_DIR, exist_ok=True)
    exp_num = 1
    while True:
        save_dir = os.path.join(config.RUNS_BASE_DIR, f"exp{exp_num}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"[*] Thư mục lưu kết quả: {save_dir}")
            config.SAVE_DIR = save_dir
            return
        exp_num += 1
