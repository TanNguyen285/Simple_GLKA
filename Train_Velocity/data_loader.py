"""
data_loader.py — GLKAFlow v2
Dataset video: mỗi class là subfolder chứa *.mp4 / *.avi
    dataset/
        train/
            ket/        ← video 30s
            khong_ket/
        val/
            ket/
            khong_ket/

Mỗi sample trả về:
    frame : Tensor (3, H, W)   — frame hiện tại (đã normalize)
    diff  : Tensor (3, H, W)   — frame difference hoặc optical flow
    label : int
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from config import config


# -----------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}


def _scan_videos(root_dir):
    """Trả về list (path, class_idx), class_names."""
    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples = []
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1] in VIDEO_EXTS:
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
    return samples, class_names, class_to_idx


def _extract_frames(video_path, n_frames, img_size):
    """
    Đọc video, sample đều `n_frames` frame.
    Trả về list PIL Image (RGB), hoặc None nếu lỗi.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    # Lấy đúng n_frames chỉ số frame, trải đều trên video
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(Image.fromarray(frame))
        prev_idx = idx

    cap.release()
    return frames if len(frames) >= 2 else None


def _compute_diff_subtract(pil_a, pil_b):
    """Frame difference: |I_t - I_{t-1}|, trả về PIL Image."""
    a = np.array(pil_a, dtype=np.float32)
    b = np.array(pil_b, dtype=np.float32)
    diff = np.abs(b - a)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def _compute_diff_optical_flow(pil_a, pil_b):
    """Farneback optical flow, encode (mag, angle) → RGB, trả về PIL Image."""
    a_gray = cv2.cvtColor(np.array(pil_a), cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        a_gray, b_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*a_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # hue = direction
    hsv[..., 1] = 255                              # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb)


def compute_diff(pil_a, pil_b, mode="subtract"):
    if mode == "optical_flow":
        return _compute_diff_optical_flow(pil_a, pil_b)
    return _compute_diff_subtract(pil_a, pil_b)


# -----------------------------------------------------------------------
# AUGMENTATION  (áp dụng đồng thời lên frame VÀ diff để giữ consistency)
# -----------------------------------------------------------------------

class PairedTransform:
    """Áp dụng cùng random transform cho (frame, diff)."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
        self.erasing = transforms.RandomErasing(
            p=config.ERASING_PROB,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0
        )

    def __call__(self, frame_pil, diff_pil):
        if self.is_train:
            # RandomResizedCrop — cùng params
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                frame_pil,
                scale=config.CROP_SCALE,
                ratio=config.CROP_RATIO
            )
            frame_pil = TF.resized_crop(frame_pil, i, j, h, w, (self.img_size, self.img_size))
            diff_pil  = TF.resized_crop(diff_pil,  i, j, h, w, (self.img_size, self.img_size))

            # RandomHorizontalFlip — cùng flip
            if random.random() < config.FLIP_PROB:
                frame_pil = TF.hflip(frame_pil)
                diff_pil  = TF.hflip(diff_pil)

            # ColorJitter chỉ trên frame (diff là motion signal — không jitter)
            cj = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            frame_pil = cj(frame_pil)
        else:
            frame_pil = TF.resize(frame_pil, (self.img_size, self.img_size))
            diff_pil  = TF.resize(diff_pil,  (self.img_size, self.img_size))

        frame_t = self.normalize(self.to_tensor(frame_pil))
        diff_t  = self.normalize(self.to_tensor(diff_pil))

        if self.is_train:
            # RandomErasing độc lập (không cần paired)
            frame_t = self.erasing(frame_t)

        return frame_t, diff_t


# -----------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------

class VideoFrameDataset(Dataset):
    """
    Mỗi __getitem__ chọn ngẫu nhiên 1 cặp (frame_t, frame_{t-1}) từ video.
    Train: random pair | Val: pair ở giữa video (deterministic).
    """

    def __init__(self, root_dir, is_train=True, diff_mode="subtract"):
        self.samples, self.class_names, self.class_to_idx = _scan_videos(root_dir)
        if not self.samples:
            raise RuntimeError(f"Không tìm thấy video trong {root_dir}")

        self.is_train  = is_train
        self.diff_mode = diff_mode
        self.n_frames  = config.FPS_SAMPLE * config.CLIP_DURATION   # ví dụ 60
        self.transform = PairedTransform(config.IMG_SIZE, is_train)

        # Pre-validate — bỏ video không đọc được
        self._validate()

    def _validate(self):
        valid = []
        broken = []
        for path, label in self.samples:
            cap = cv2.VideoCapture(path)
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 2
            cap.release()
            if ok:
                valid.append((path, label))
            else:
                broken.append(path)
        if broken:
            print(f"[⚠] {len(broken)} video lỗi đã bị bỏ qua:")
            for p in broken:
                print(f"    {p}")
        self.samples = valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = _extract_frames(path, self.n_frames, config.IMG_SIZE)

        if frames is None or len(frames) < 2:
            # Fallback: black frame
            black = Image.fromarray(np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8))
            frame_t, diff_t = self.transform(black, black)
            return frame_t, diff_t, label

        if self.is_train:
            t = random.randint(1, len(frames) - 1)
        else:
            t = len(frames) // 2   # deterministic

        frame_pil = frames[t]
        prev_pil  = frames[t - 1]
        diff_pil  = compute_diff(prev_pil, frame_pil, mode=self.diff_mode)

        frame_t, diff_t = self.transform(frame_pil, diff_pil)
        return frame_t, diff_t, label


# -----------------------------------------------------------------------
# DATALOADER FACTORY
# -----------------------------------------------------------------------

def get_data_loaders(handle_imbalance=False):
    train_dir = os.path.join(config.DATA_DIR, "train")
    val_dir   = os.path.join(config.DATA_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Không tìm thấy 'train' hoặc 'val' trong {config.DATA_DIR}"
        )

    train_dataset = VideoFrameDataset(train_dir, is_train=True,  diff_mode=config.DIFF_MODE)
    val_dataset   = VideoFrameDataset(val_dir,   is_train=False, diff_mode=config.DIFF_MODE)

    # ---- Class imbalance ----
    train_sampler = None
    if handle_imbalance:
        labels = [s[1] for s in train_dataset.samples]
        class_counts = np.bincount(labels)
        ratio = class_counts.max() / (class_counts.min() + 1e-8)
        if ratio > 1.5:
            weights = 1.0 / torch.tensor(class_counts[labels], dtype=torch.float)
            train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
            print(f"[⚠] Mất cân bằng (ratio={ratio:.1f}x) → WeightedRandomSampler")
        else:
            print(f"[✓] Classes cân bằng (ratio={ratio:.1f}x)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    print(f"\n{'='*60}")
    print(f"[✓] Data Config:")
    print(f"    Train videos : {len(train_dataset)}")
    print(f"    Val videos   : {len(val_dataset)}")
    print(f"    Classes      : {train_dataset.class_names}")
    print(f"    Frames/video : {train_dataset.n_frames}  ({config.FPS_SAMPLE} fps × {config.CLIP_DURATION}s)")
    print(f"    Diff mode    : {config.DIFF_MODE}")
    print(f"    Batch size   : {config.BATCH_SIZE} train | {config.BATCH_SIZE*2} val")
    print(f"{'='*60}\n")

    return train_loader, val_loader, train_dataset.class_to_idx


# -----------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, class_map = get_data_loaders(handle_imbalance=True)
    frames, diffs, labels = next(iter(train_loader))
    print(f"frames : {frames.shape}  {frames.dtype}")
    print(f"diffs  : {diffs.shape}   {diffs.dtype}")
    print(f"labels : {labels}")
