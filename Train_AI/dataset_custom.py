import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as F
from PIL import Image
from config import config
import warnings
import numpy as np

class SmartSquareTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        w, h = img.size
        if abs(w - h) < (0.01 * max(w, h)):
            return F.resize(img, [self.img_size, self.img_size])
        img = F.resize(img, self.img_size)
        return F.center_crop(img, (self.img_size, self.img_size))


class ValidatedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, skip_broken=True):
        self.skip_broken = skip_broken
        self.broken_files = []
        super().__init__(root, transform=transform)

    def _load_target(self, path):
        try:
            with Image.open(path) as img:
                img.load()
            return super()._load_target(path)
        except Exception as e:
            self.broken_files.append((path, str(e)))
            if self.skip_broken:
                return None
            raise


def get_augmentation_config(dataset_size):

    if dataset_size < 5000:
        print(f"[⚠] Dataset nhỏ ({dataset_size} ảnh) - Augmentation fine-tune cho camera cố định")
        return {
            'crop_scale': (0.8, 1.0),       # Giữ context — KHÔNG crop aggressive
            'crop_ratio': (0.75, 1.33),
            'flip_prob': 0.5,               # Tự nhiên hơn 0.7
            'erasing_prob': 0.3,            # Thấp — không che mất xe
            'color_jitter': (0.2, 0.2, 0.2, 0.05),  # Hue thấp giữ màu đèn đỏ
            'affine_degrees': 5,            # Xoay nhẹ — camera đôi khi lệch
            'affine_translate': (0.05, 0.05),
        }
    elif dataset_size < 20000:
        print(f"[*] Dataset vừa ({dataset_size} ảnh) - Augmentation TRUNG BÌNH")
        return {
            'crop_scale': (0.7, 1.0),
            'crop_ratio': (0.8, 1.25),
            'flip_prob': 0.5,
            'erasing_prob': 0.3,
            'color_jitter': (0.3, 0.3, 0.15, 0.08),
            'affine_degrees': 5,
            'affine_translate': (0.05, 0.05),
        }
    else:
        print(f"[✓] Dataset lớn ({dataset_size} ảnh) - Augmentation NHẸ")
        return {
            'crop_scale': (0.8, 1.0),
            'crop_ratio': (0.9, 1.11),
            'flip_prob': 0.3,
            'erasing_prob': 0.1,
            'color_jitter': (0.2, 0.2, 0.1, 0.05),
            'affine_degrees': 3,
            'affine_translate': (0.03, 0.03),
        }


def get_batch_size(dataset_size):
    if dataset_size < 5000:
        return min(32, config.BATCH_SIZE)
    elif dataset_size < 20000:
        return min(64, config.BATCH_SIZE)
    else:
        return min(128, config.BATCH_SIZE)


def get_data_loaders(validate_images=True, handle_imbalance=False):

    # ============ PATH CHECK ============
    train_dir = os.path.join(config.DATA_DIR, 'train')
    val_dir   = os.path.join(config.DATA_DIR, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"LỖI: Không tìm thấy 'train' hoặc 'val' bên trong {config.DATA_DIR}"
        )

    # ============ LOAD TRAIN DATASET (để biết size) ============
    print("[*] Đang tải dataset...")
    train_dataset_temp = datasets.ImageFolder(train_dir)
    dataset_size = len(train_dataset_temp)

    # ============ ADAPTIVE AUGMENTATION ============
    aug = get_augmentation_config(dataset_size)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            config.IMG_SIZE,
            scale=aug['crop_scale'],
            ratio=aug['crop_ratio']
        ),
        transforms.RandomAffine(                    # Thêm vào — quan trọng cho camera cố định
            degrees=aug['affine_degrees'],
            translate=aug['affine_translate']
        ),
        transforms.RandomHorizontalFlip(p=aug['flip_prob']),
        transforms.ColorJitter(*aug['color_jitter']),
        # AutoAugment đã bỏ — làm nhiễu distribution ảnh giao thông thực
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=aug['erasing_prob'],
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0
        ),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),
    ])

    # ============ VALIDATION TRANSFORMS ============
    val_transforms = transforms.Compose([
        SmartSquareTransform(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),
    ])

    # ============ LOAD DATASETS ============
    if validate_images:
        train_dataset = ValidatedImageFolder(train_dir, transform=train_transforms)
        val_dataset   = ValidatedImageFolder(val_dir,   transform=val_transforms)

        if train_dataset.broken_files:
            print(f"[⚠] Tìm thấy {len(train_dataset.broken_files)} ảnh corrupted trong train")
        if val_dataset.broken_files:
            print(f"[⚠] Tìm thấy {len(val_dataset.broken_files)} ảnh corrupted trong val")
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    # ============ HANDLE CLASS IMBALANCE ============
    train_sampler = None
    if handle_imbalance:
        class_counts = np.bincount(train_dataset.targets)
        ratio = class_counts.max() / class_counts.min()
        if ratio > 1.5:
            print(f"[⚠] Mất cân bằng classes (ratio={ratio:.1f}x): {dict(zip(train_dataset.classes, class_counts))}")
            weights = 1.0 / torch.tensor(class_counts[train_dataset.targets], dtype=torch.float)
            train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
            print(f"[✓] Dùng WeightedRandomSampler")
        else:
            print(f"[✓] Classes cân bằng (ratio={ratio:.1f}x)")

    # ============ ADAPTIVE BATCH SIZE ============
    batch_size = get_batch_size(dataset_size)

    # ============ NUM WORKERS ============
    num_workers = min(config.NUM_WORKERS, 8)

    # ============ CREATE DATALOADERS ============
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(num_workers > 0)
    )

    # ============ REPORT ============
    print(f"\n{'='*60}")
    print(f"[✓] Data Loading Config:")
    print(f"    Dataset size      : {len(train_dataset):,} train | {len(val_dataset):,} val")
    print(f"    Batch size        : {batch_size} train | {batch_size*2} val")
    print(f"    Num workers       : {num_workers}")
    print(f"    Classes           : {len(train_dataset.classes)}")
    print(f"    Class distribution: {dict(zip(train_dataset.classes, np.bincount(train_dataset.targets)))}")
    print(f"    Augmentation      : {'FINE-TUNE (<5k)' if dataset_size < 5000 else 'MEDIUM' if dataset_size < 20000 else 'LIGHT'}")
    print(f"    Crop scale        : {aug['crop_scale']}")
    print(f"    Affine            : degrees={aug['affine_degrees']}, translate={aug['affine_translate']}")
    print(f"    Device            : {config.DEVICE}")
    print(f"{'='*60}\n")

    return train_loader, val_loader, train_dataset.class_to_idx


if __name__ == "__main__":
    try:
        train_loader, val_loader, class_map = get_data_loaders(
            validate_images=True,
            handle_imbalance=True
        )

        images, labels = next(iter(train_loader))
        print(f"[*] Batch test:")
        print(f"    Shape      : {images.shape}")
        print(f"    Dtype      : {images.dtype}")
        print(f"    Min/Max    : {images.min():.2f} / {images.max():.2f}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()