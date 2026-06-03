import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

# >>> IMPORT FILE CONFIG VÀO ĐÂY <<<
from config import config


class SmartSquareTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        w, h = img.size
        if abs(w - h) < (0.01 * max(w, h)):
            return F.resize(img, [self.img_size, self.img_size])
        img = F.resize(img, self.img_size)
        return F.center_crop(img, (self.img_size, self.img_size))


def get_data_loaders():
    # 1. Train transforms (Đọc toàn bộ tham số từ config)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            config.IMG_SIZE, 
            scale=config.CROP_SCALE, 
            ratio=config.CROP_RATIO
        ),
        
        transforms.RandomHorizontalFlip(p=config.FLIP_PROB),

        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.IMAGENET
        ),

        transforms.ToTensor(),

        transforms.RandomErasing(
            p=config.ERASING_PROB, 
            scale=(0.02, 0.2), 
            ratio=(0.3, 3.3), 
            value=0
        ),

        transforms.Normalize(mean=config.NORMALIZE_MEAN,
                             std=config.NORMALIZE_STD),
    ])

    # 2. Val transforms — không augment, chỉ smart crop
    val_transforms = transforms.Compose([
        SmartSquareTransform(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN,
                             std=config.NORMALIZE_STD),
    ])

    # 3. Kiểm tra đường dẫn thông qua config.DATA_DIR
    train_dir = os.path.join(config.DATA_DIR, 'train')
    val_dir   = os.path.join(config.DATA_DIR, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"LỖI: Không tìm thấy thư mục 'train' hoặc 'val' bên trong {config.DATA_DIR}"
        )

    # 4. Dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    # 5. DataLoader (Sử dụng cấu hình phần cứng từ config)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    # 6. Báo cáo
    print(f"[*] Đã tải xong dữ liệu từ: {config.DATA_DIR}")
    print(f"    - Số ảnh Train : {len(train_dataset)}")
    print(f"    - Số ảnh Val   : {len(val_dataset)}")
    print(f"    - Classes      : {train_dataset.class_to_idx}")

    return train_loader, val_loader, train_dataset.class_to_idx


if __name__ == "__main__":
    try:
        # Không cần truyền tham số thủ công nữa, hàm tự đọc từ file config
        train_loader, val_loader, class_map = get_data_loaders()
        
        images, labels = next(iter(train_loader))
        print(f"\n[*] Thiết bị hiện tại đang cấu hình: {config.DEVICE}")
        print(f"[*] Kích thước 1 batch: {images.shape}")
        
    except Exception as e:
        print(f"Lỗi: {e}")