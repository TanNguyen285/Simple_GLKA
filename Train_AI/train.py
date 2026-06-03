import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset_custom import get_data_loaders
from model.Simple_Anphax import Simple_GLKA

# ĐỒNG BỘ HÓA CẤU HÌNH TỪ CONFIG.PY
from config import config

# >>> ĐÃ XÓA HÀM create_run_dir VÌ LOGIC ĐÃ ĐƯỢC CHUYỂN SANG CONFIG.PY <<<

def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, digits=4)
    
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
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"\n{report}\n")


def plot_tsne_features(all_features, all_labels, class_names, save_dir, epoch=None):
    try:
        if len(all_features) == 0:
            return
        
        print("[*] Đang tính toán t-SNE...")
        feat_arr = np.array(all_features)
        label_arr = np.array(all_labels)
        
        if len(feat_arr) > 500:
            idx = np.random.choice(len(feat_arr), 500, replace=False)
            feat_arr = feat_arr[idx]
            label_arr = label_arr[idx]
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                    init='pca', learning_rate='auto')
        embedded = tsne.fit_transform(feat_arr)
        
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
    try:
        model.eval()
        class_images = {i: [] for i in range(len(class_names))}
        
        with torch.no_grad():
            for images, labels in val_loader:
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    class_images[label_idx].append(images[i:i+1])
        
        fixed_indices = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
        selected_images_and_labels = []
        
        for idx in fixed_indices:
            pair = []
            for class_idx in range(len(class_names)):
                if idx < len(class_images[class_idx]):
                    pair.append((class_images[class_idx][idx], class_idx))
                else:
                    if class_images[class_idx]:
                        pair.append((class_images[class_idx][-1], class_idx))
            
            if len(pair) == len(class_names):
                selected_images_and_labels.append(pair)
        
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super(ModelWrapper, self).__init__()
                self.original_model = original_model
            
            def forward(self, x):
                logits, _ = self.original_model(x)  
                return logits
        
        wrapper_model = ModelWrapper(model)
        target_layers = [model.blocks[-1]]  
        cam = GradCAM(model=wrapper_model, target_layers=target_layers)
        
        fig, axes = plt.subplots(len(class_names), len(selected_images_and_labels), figsize=(24, 8))
        
        for col, pair in enumerate(selected_images_and_labels):
            for row, (input_image, label_idx) in enumerate(pair):
                input_image = input_image.to(device)
                grayscale_cam = cam(input_tensor=input_image, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                input_image_np = input_image[0].cpu().permute(1, 2, 0).numpy()
                input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min() + 1e-8)
                
                visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
                
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
    # ĐỒNG BỘ THÔNG TIN ĐƯỜNG DẪN LƯU KẾT QUẢ TỪ CONFIG
    SAVE_DIR = config.SAVE_DIR
    DATA_DIR = config.DATA_DIR
    
    BATCH_SIZE = config.BATCH_SIZE        
    IMG_SIZE = config.IMG_SIZE          
    EPOCHS = config.EPOCHS             
    LEARNING_RATE = config.LEARNING_RATE   
    WEIGHT_DECAY = 7.e-4  
    MOMENTUM = 0.9       
    DEVICE = torch.device(config.DEVICE)
    print(f"[*] Device: {DEVICE}")

    # ===== CHUẨN BỊ DỮ LIỆU =====
    train_loader, val_loader, class_to_idx = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    class_names = list(class_to_idx.keys())
    print(f"[*] Số lớp: {len(class_names)} | Classes: {class_names}")

    # ===== KHỞI TẠO MÔ HÌNH & OPTIMIZER =====
    model = Simple_GLKA(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), 
                          lr=LEARNING_RATE, 
                          momentum=MOMENTUM, 
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ===== KHỞI TẠO HISTORY =====
    best_val_acc = 0.0
    best_val_loss = float('inf') 
    history_train_loss, history_val_loss, history_val_acc = [], [], []
    best_epoch_preds, best_epoch_labels = None, None
    
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
            outputs, _ = model(images)  
            
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nResults: LR={current_lr:.6f} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.4f}%")
        
        save_classification_report(current_preds, current_labels, class_names, SAVE_DIR, 
                                  epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        should_generate_gradcam = False

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_acc.pth'))
            print(f"  [ACC] --> Cập nhật Best Acc: {val_acc*100:.2f}%")
            
            plot_confusion_matrix(current_preds, current_labels, class_names, SAVE_DIR)
            save_classification_report(current_preds, current_labels, class_names, SAVE_DIR)
            plot_tsne_features(current_features, current_labels, class_names, SAVE_DIR, epoch=epoch)
            
            should_generate_gradcam = True

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_loss.pth'))
            print(f"  [LOSS] --> Cập nhật Best Loss: {val_loss:.4f}")
            
            should_generate_gradcam = True

        if should_generate_gradcam:
            print(f"  [*] Đang sinh Heatmap cho điểm hội tụ tốt nhất tại epoch {epoch+1}...")
            generate_gradcam_heatmap(model, val_loader, DEVICE, SAVE_DIR, epoch, class_names)
            plot_confusion_matrix(current_preds, current_labels, class_names, SAVE_DIR)
            save_classification_report(current_preds, current_labels, class_names, SAVE_DIR)
            plot_tsne_features(current_features, current_labels, class_names, SAVE_DIR, epoch=epoch)
        
        plot_training_curves(history_train_loss, history_val_loss, history_val_acc, SAVE_DIR)
    

    print(f"\n{'='*60}")
    print(f"[✓] Hoàn thành training! Kết quả lưu tại: {SAVE_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()