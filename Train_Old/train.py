import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE # <--- THÊM MỚI
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset_custom import get_data_loaders
from Simple_Beta import Simple_GLKA

def create_run_dir(base_dir='runs'):
    """Tự động tạo thư mục runs/exp1, runs/exp2... giống YOLO"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    exp_num = 1
    while True:
        run_dir = os.path.join(base_dir, f'exp{exp_num}')
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            print(f"[*] Đã tạo thư mục lưu kết quả: {run_dir}")
            return run_dir
        exp_num += 1

def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir):
    """
    ===== PLOT 1: Vẽ biểu đồ Loss và Accuracy =====
    - Subplot 1: Train/Val Loss
    - Subplot 2: Validation Accuracy
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    plt.title('Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(all_preds, all_labels, class_names, save_dir):
    """
    ===== PLOT 2: Vẽ ma trận nhầm lẫn (Confusion Matrix) =====
    Hiển thị độ chính xác phân loại giữa các lớp
    """
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def save_classification_report(all_preds, all_labels, class_names, save_dir, epoch=None, train_loss=None, val_loss=None):
    """
    ===== LƯU CLASSIFICATION REPORT CHI TIẾT =====
    In chi tiết Precision, Recall, F1-score, Accuracy cho mỗi lớp
    - Lưu report của best epoch vào classification_report.txt
    - Lưu từng epoch vào epoch_reports.txt
    """
    from sklearn.metrics import accuracy_score
    
    # Tính toán accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Tạo classification report chi tiết
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, digits=4)
    
    # Nếu là epoch report, thêm vào file epoch_reports.txt
    if epoch is not None:
        epoch_file = os.path.join(save_dir, 'epoch_reports.txt')
        with open(epoch_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EPOCH {epoch+1}\n")
            f.write(f"{'='*80}\n")
            if train_loss is not None:
                f.write(f"Train Loss:      {train_loss:.4f}\n")
            if val_loss is not None:
                f.write(f"Val Loss:        {val_loss:.4f}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"\n{report}\n")
    else:
        # Lưu best epoch report vào classification_report.txt
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"\n{report}\n")


def plot_tsne_features(all_features, all_labels, class_names, save_dir, epoch=None):
    """
    ===== OPTIONAL: Vẽ t-SNE để visualize đặc trưng =====
    Chỉ gọi khi cần thiết (không gọi mỗi epoch để tiết kiệm thời gian)
    """
    try:
        if len(all_features) == 0:
            return
        
        print("[*] Đang tính toán t-SNE...")
        feat_arr = np.array(all_features)
        label_arr = np.array(all_labels)
        
        # Giới hạn tối đa 500 mẫu để vẽ nhanh
        if len(feat_arr) > 500:
            idx = np.random.choice(len(feat_arr), 500, replace=False)
            feat_arr = feat_arr[idx]
            label_arr = label_arr[idx]
        
        # Tính t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                    init='pca', learning_rate='auto')
        embedded = tsne.fit_transform(feat_arr)
        
        # Vẽ t-SNE
        plt.figure(figsize=(10, 8))
        for i, name in enumerate(class_names):
            mask = label_arr == i
            plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                       label=name, alpha=0.6, s=50)
        
        plt.title(f't-SNE: Feature Separation (Epoch {epoch+1})', 
                 fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        filename = f'tsne_epoch_{epoch+1:02d}.png' if epoch is not None else 'tsne_final.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"[!] Lỗi t-SNE: {e}")

def generate_gradcam_heatmap(model, val_loader, device, save_dir, epoch, class_names):
    """Sinh Grad-CAM heatmap từ 10 cặp ảnh cố định (1 từ mỗi class) - cách 40 ảnh lấy 1 ảnh"""
    try:
        model.eval()
        
        # Thu thập tất cả ảnh từ từng class theo thứ tự
        class_images = {i: [] for i in range(len(class_names))}
        
        with torch.no_grad():
            for images, labels in val_loader:
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    class_images[label_idx].append(images[i:i+1])
        
        # Chọn 10 cặp ảnh từ indices cố định: 0, 40, 80, 120, 160, 200, 240, 280, 320, 360
        fixed_indices = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
        selected_images_and_labels = []
        
        for idx in fixed_indices:
            pair = []
            for class_idx in range(len(class_names)):
                if idx < len(class_images[class_idx]):
                    pair.append((class_images[class_idx][idx], class_idx))
                else:
                    # Nếu không đủ ảnh, dùng ảnh cuối cùng có sẵn
                    if class_images[class_idx]:
                        pair.append((class_images[class_idx][-1], class_idx))
            
            if len(pair) == len(class_names):
                selected_images_and_labels.append(pair)
        
        # Tạo wrapper model để GradCAM chỉ nhận logits (không tuple)
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super(ModelWrapper, self).__init__()
                self.original_model = original_model
            
            def forward(self, x):
                logits, _ = self.original_model(x)  # Chỉ lấy logits, bỏ features
                return logits
        
        wrapper_model = ModelWrapper(model)
        
        # Khởi tạo Grad-CAM trên lớp blocks (feature extraction cuối cùng)
        target_layers = [model.blocks[-1]]  # Hook vào EB6 (lớp cuối của blocks)
        cam = GradCAM(model=wrapper_model, target_layers=target_layers)
        
        # Vẽ Grad-CAM cho 10 cặp ảnh (2 hàng x 10 cột)
        fig, axes = plt.subplots(len(class_names), len(selected_images_and_labels), figsize=(24, 8))
        
        for col, pair in enumerate(selected_images_and_labels):
            for row, (input_image, label_idx) in enumerate(pair):
                input_image = input_image.to(device)
                
                # Tính toán Grad-CAM
                grayscale_cam = cam(input_tensor=input_image, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                # Chuẩn hóa ảnh gốc về [0, 1]
                input_image_np = input_image[0].cpu().permute(1, 2, 0).numpy()
                input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min() + 1e-8)
                
                # Vẽ Grad-CAM trên ảnh gốc
                visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
                
                # Hiển thị trên subplot
                axes[row, col].imshow(visualization)
                axes[row, col].set_title(f'{class_names[label_idx]} (idx={fixed_indices[col]})', fontsize=9)
                axes[row, col].axis('off')
        
        plt.suptitle(f'Grad-CAM: 10 cặp ảnh cố định - Epoch {epoch+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        gradcam_path = os.path.join(save_dir, f'gradcam_epoch_{epoch+1:02d}.png')
        plt.savefig(gradcam_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        
        print(f"  -> Đã lưu Grad-CAM heatmap (10 cặp ảnh): {gradcam_path}")
        
    except Exception as e:
        print(f"[!] Lỗi khi sinh Grad-CAM: {e}")

def train_model():
    """
    ========== MAIN TRAINING PIPELINE ==========
    """
    
    # ===== 1. CẤU HÌNH TRAINING =====
    DATA_DIR = './dataset'  # Đường dẫn đến dataset đã được chuẩn bị
    BATCH_SIZE = 32        
    IMG_SIZE = 224          
    EPOCHS = 25             
    LEARNING_RATE = 0.0015   
    WEIGHT_DECAY = 7.e-4  
    MOMENTUM = 0.9       
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {DEVICE}")

    # ===== 2. CHUẨN BỊ DỮ LIỆU =====
    SAVE_DIR = create_run_dir()
    train_loader, val_loader, class_to_idx = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    class_names = list(class_to_idx.keys())
    print(f"[*] Số lớp: {len(class_names)} | Classes: {class_names}")

    # ===== 3. KHỞI TẠO MÔ HÌNH & OPTIMIZER =====
    model = Simple_GLKA(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), 
                          lr=LEARNING_RATE, 
                          momentum=MOMENTUM, 
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ===== 4. KHỞI TẠO HISTORY =====
    best_val_acc = 0.0
    best_val_loss = float('inf') # Khởi tạo giá trị vô cùng lớn cho loss
    history_train_loss, history_val_loss, history_val_acc = [], [], []
    best_epoch_preds, best_epoch_labels = None, None
    
    # Tạo file lưu history từng epoch
    epoch_history_file = os.path.join(SAVE_DIR, 'epoch_reports.txt')
    with open(epoch_history_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Image Size: {IMG_SIZE}\n")
        f.write(f"  Total Epochs: {EPOCHS}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"  Momentum: {MOMENTUM}\n")
        f.write(f"  Classes: {class_names}\n")
        f.write(f"\n{'='*80}\n")

    # ========== BẮT ĐẦU TRAINING LOOP ==========
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # ===== TRAINING PHASE =====
        print("[TRAIN] Đang huấn luyện...")
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            
            optimizer.zero_grad()
            outputs, _ = model(images)  # Output: (logits, features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history_train_loss.append(train_loss)

        # ===== VALIDATION PHASE =====
        print("[EVAL] Đang đánh giá mô hình...")
        model.eval()
        val_loss = 0.0
        corrects = 0
        current_preds, current_labels, current_features = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                outputs, features = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                corrects += torch.sum(preds == labels).item()
                
                current_preds.extend(preds.cpu().numpy())
                current_labels.extend(labels.cpu().numpy())
                current_features.extend(features.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects / len(val_loader.dataset)
        
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc * 100)
        scheduler.step()
        
        # ===== IN KẾT QUẢ MỖI EPOCH =====
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nResults: LR={current_lr:.6f} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.4f}%")
        
        # ===== LƯU ĐÁNH GIÁ CỦA MỖI EPOCH =====
        save_classification_report(current_preds, current_labels, class_names, SAVE_DIR, 
                                  epoch=epoch, train_loss=train_loss, val_loss=val_loss)

       # =# ===== LƯU MÔ HÌNH VÀ SINH HEATMAP CHIẾN THUẬT =====
        should_generate_gradcam = False

        # --- Kiểm tra Best Accuracy ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_acc.pth'))
            print(f"  [ACC] --> Cập nhật Best Acc: {val_acc*100:.2f}%")
            
            # Cập nhật các phân tích phụ trợ
            plot_confusion_matrix(current_preds, current_labels, class_names, SAVE_DIR)
            save_classification_report(current_preds, current_labels, class_names, SAVE_DIR)
            plot_tsne_features(current_features, current_labels, class_names, SAVE_DIR, epoch=epoch)
            
            should_generate_gradcam = True

        # --- Kiểm tra Best Loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_loss.pth'))
            print(f"  [LOSS] --> Cập nhật Best Loss: {val_loss:.4f}")
            
            should_generate_gradcam = True

        # --- CHỈ SINH GRAD-CAM KHI ĐẠT BEST (ACC HOẶC LOSS) ---
        if should_generate_gradcam:
            print(f"  [*] Đang sinh Heatmap cho điểm hội tụ tốt nhất tại epoch {epoch+1}...")
            generate_gradcam_heatmap(model, val_loader, DEVICE, SAVE_DIR, epoch, class_names)
            # (Tùy chọn) Bạn có thể lưu thêm report riêng cho bản best_loss nếu muốn
            # with open(os.path.join(SAVE_DIR, 'best_loss_info.txt'), 'w') as f:
            #     f.write(f"Epoch: {epoch+1}\nLoss: {val_loss:.4f}\nAcc: {val_acc*100:.2f}%")
            
            # Chỉ cập nhật confusion matrix khi có epoch tốt nhất
            plot_confusion_matrix(current_preds, current_labels, class_names, SAVE_DIR)
            
            # Chỉ lưu classification report chi tiết của best epoch
            save_classification_report(current_preds, current_labels, class_names, SAVE_DIR)
            
            # Chỉ cập nhật ảnh hội tụ đặc trưng khi có best epoch
            plot_tsne_features(current_features, current_labels, class_names, SAVE_DIR, epoch=epoch)
        
        # ===== VẼ BIỂU ĐỒ TRAINING =====
        plot_training_curves(history_train_loss, history_val_loss, history_val_acc, SAVE_DIR)
    

    # ========== KẾT THÚC TRAINING ==========
    print(f"\n{'='*60}")
    print(f"[✓] Hoàn thành training! Kết quả lưu tại: {SAVE_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()