import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image

# --- Giữ nguyên logic Smart Crop để xử lý ảnh vuông/chữ nhật ---
class SmartSquareTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        w, h = img.size
        # Nếu ảnh đã vuông (sai số dưới 1%), chỉ resize
        if abs(w - h) < (0.01 * max(w, h)):
            return F.resize(img, [self.img_size, self.img_size])
        # Nếu ảnh chưa vuông, resize cạnh ngắn rồi cắt giữa
        img = F.resize(img, self.img_size)
        return F.center_crop(img, (self.img_size, self.img_size))

def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=8):
    """
    Hàm chuẩn bị DataLoader nâng cao:
    - Giữ nguyên tên biến và cấu trúc cũ của ông.
    - Tích hợp SmartSquareTransform cho tập Val.
    """
    
    # 1. Data Augmentation cho tập Train
    train_transforms = transforms.Compose([
        # Giữ scale cao để ưu tiên vùng ảnh ông đã tiền xử lý
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Jitter màu sắc (Hue thấp để giữ màu đèn phanh đỏ)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        
        transforms.ToTensor(),
        
        # RandomErasing đặt sau ToTensor
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Tập Val: Sử dụng SmartSquareTransform để không crop ảnh 500x500
    val_transforms = transforms.Compose([
        SmartSquareTransform(img_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Kiểm tra đường dẫn và đọc dữ liệu (Giữ nguyên logic cũ)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_path := train_dir) or not os.path.exists(val_path := val_dir):
        raise FileNotFoundError(f"LỖI: Không tìm thấy thư mục 'train' hoặc 'val' bên trong {data_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # 4. Đóng gói vào DataLoader (Giữ nguyên tham số num_workers=8 của ông)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # 5. In báo cáo tóm tắt
    print(f"[*] Đã tải xong Dữ liệu Traffic (Custom Smart Crop)!")
    print(f"    - Số ảnh Train: {len(train_dataset)}")
    print(f"    - Số ảnh Validation: {len(val_dataset)}")
    print(f"    - Danh sách các nhãn (Classes): {train_dataset.class_to_idx}")
    
    return train_loader, val_loader, train_dataset.class_to_idx

if __name__ == "__main__":
    DATA_DIR = "dataset" 
    try:
        # Giữ nguyên cách gọi hàm của ông
        train_loader, val_loader, class_map = get_data_loaders(DATA_DIR, batch_size=32, img_size=224)
        images, labels = next(iter(train_loader))
        print(f"\n[*] Kích thước 1 batch ảnh: {images.shape}")
    except Exception as e:
        print(f"Lỗi: {e}")