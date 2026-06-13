"""
train.py — GLKAFlow v2
Input per sample: (frame, diff, label)
Model: GLKAFlow_v2(frame, diff) → (logits, features)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import config, create_save_dir
from data_loader import get_data_loaders

# ---- import model ----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
model_dir   = os.path.join(parent_dir, "model")
if model_dir not in sys.path:
    sys.path.append(model_dir)
from GLKAFlow_v2 import GLKAFlow_v2


# ============================================================
# PLOT / REPORT HELPERS
# ============================================================

def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val Loss",   linewidth=2)
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Acc", color="green", linewidth=2)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(preds, labels, class_names, save_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def save_classification_report(preds, labels, class_names, save_dir,
                                epoch=None, train_loss=None, val_loss=None):
    acc    = accuracy_score(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=class_names,
                                   zero_division=0, digits=4)
    if epoch is not None:
        fpath = os.path.join(save_dir, "epoch_reports.txt")
        with open(fpath, "a") as f:
            f.write(f"\n{'='*80}\nEPOCH {epoch+1}\n{'='*80}\n")
            if train_loss is not None: f.write(f"Train Loss : {train_loss:.4f}\n")
            if val_loss  is not None: f.write(f"Val Loss   : {val_loss:.4f}\n")
            f.write(f"Accuracy   : {acc:.4f}\n\n{report}\n")
    else:
        with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n{report}\n")


def plot_tsne(features, labels, class_names, save_dir, epoch=None):
    try:
        arr = np.array(features)
        lbl = np.array(labels)
        if len(arr) > 500:
            idx = np.random.choice(len(arr), 500, replace=False)
            arr, lbl = arr[idx], lbl[idx]

        embedded = TSNE(n_components=2, perplexity=30, random_state=42,
                        init="pca", learning_rate="auto").fit_transform(arr)

        plt.figure(figsize=(10, 8))
        for i, name in enumerate(class_names):
            mask = lbl == i
            plt.scatter(embedded[mask, 0], embedded[mask, 1], label=name, alpha=0.6, s=50)
        ep = epoch + 1 if epoch is not None else "?"
        plt.title(f"t-SNE Epoch {ep}"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"tsne_epoch_{epoch+1:02d}.png" if epoch is not None else "tsne_final.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[!] t-SNE lỗi: {e}")


# ============================================================
# GRAD-CAM  (target: backbone cuối — frame branch)
# ============================================================

def generate_gradcam(model, val_loader, device, save_dir, epoch, class_names):
    """
    Wrap model để GradCAM nhận đúng input (frame, diff).
    Target layer: backbone.blocks[-1].project (EfficientBlock cuối của backbone).
    """
    try:
        model.eval()

        # Buffer mỗi class tối đa 5 sample
        class_buf = {i: [] for i in range(len(class_names))}
        with torch.no_grad():
            for frames, diffs, labels in val_loader:
                for i, lbl in enumerate(labels):
                    l = lbl.item()
                    if len(class_buf[l]) < 5:
                        class_buf[l].append((frames[i:i+1], diffs[i:i+1]))

        # Wrapper — GradCAM cần model(x) → logits (single tensor)
        # Giữ diff cố định, grad qua frame
        diff_fixed = [None]

        class FrameWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, frame):
                logits, _ = self.m(frame, diff_fixed[0])
                return logits

        wrapper      = FrameWrapper(model)
        target_layer = [model.backbone.blocks[-1].project[0]]   # Conv2d sau project
        cam          = GradCAM(model=wrapper, target_layers=target_layer)

        n_cols   = max(len(v) for v in class_buf.values())
        n_rows   = len(class_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1: axes = [axes]
        if n_cols == 1: axes = [[ax] for ax in axes]

        for row, cls_idx in enumerate(range(n_rows)):
            for col, (frame_t, diff_t) in enumerate(class_buf[cls_idx]):
                frame_t = frame_t.to(device)
                diff_t  = diff_t.to(device)
                diff_fixed[0] = diff_t

                grayscale_cam = cam(input_tensor=frame_t, targets=None)[0]
                inp_np = frame_t[0].cpu().permute(1, 2, 0).numpy()
                inp_np = (inp_np - inp_np.min()) / (inp_np.max() - inp_np.min() + 1e-8)
                viz    = show_cam_on_image(inp_np, grayscale_cam, use_rgb=True)

                axes[row][col].imshow(viz)
                axes[row][col].set_title(f"{class_names[cls_idx]} #{col}", fontsize=9)
                axes[row][col].axis("off")

            # Ẩn axes trống
            for col in range(len(class_buf[cls_idx]), n_cols):
                axes[row][col].axis("off")

        plt.suptitle(f"Grad-CAM (frame branch) — Epoch {epoch+1}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, f"gradcam_epoch_{epoch+1:02d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  [Grad-CAM] → {path}")

    except Exception as e:
        print(f"[!] Grad-CAM lỗi: {e}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_model():
    create_save_dir()
    SAVE_DIR = config.SAVE_DIR
    DEVICE   = torch.device(config.DEVICE)

    print(f"[*] Device : {DEVICE}")
    print(f"[*] Epochs : {config.EPOCHS}")

    # ----- Data -----
    train_loader, val_loader, class_to_idx = get_data_loaders(handle_imbalance=False)
    class_names = list(class_to_idx.keys())

    # ----- Model -----
    model     = GLKAFlow_v2(num_classes=config.NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ----- Optimizer -----
    if config.OPTIMIZER_TYPE == "SGD":
        optimizer = optim.SGD(model.parameters(), **config.OPTIMIZER_CONFIG)
    else:
        optimizer = optim.AdamW(model.parameters(), **config.OPTIMIZER_CONFIG)

    # ----- Scheduler -----
    if config.SCHEDULER_TYPE == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config.SCHEDULER_CONFIG)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **config.SCHEDULER_CONFIG)

    # ----- Tracking -----
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    hist_train, hist_val_loss, hist_val_acc = [], [], []

    # Log header
    with open(os.path.join(SAVE_DIR, "epoch_reports.txt"), "w") as f:
        f.write("Training Config:\n")
        f.write(f"  Model       : GLKAFlow_v2\n")
        f.write(f"  Classes     : {class_names}\n")
        f.write(f"  Batch size  : {config.BATCH_SIZE}\n")
        f.write(f"  Epochs      : {config.EPOCHS}\n")
        f.write(f"  Optimizer   : {config.OPTIMIZER_TYPE}\n")
        f.write(f"  Scheduler   : {config.SCHEDULER_TYPE}\n")
        f.write(f"  Diff mode   : {config.DIFF_MODE}\n")
        f.write(f"  FPS sample  : {config.FPS_SAMPLE}\n\n")

    # ========== LOOP ==========
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"{'='*60}")

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        for frames, diffs, labels in train_loader:
            frames = frames.to(DEVICE)
            diffs  = diffs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            logits, _ = model(frames, diffs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.size(0)

        train_loss /= len(train_loader.dataset)
        hist_train.append(train_loss)

        # ---- EVAL ----
        model.eval()
        val_loss  = 0.0
        corrects  = 0
        all_preds, all_labels, all_features = [], [], []

        with torch.no_grad():
            for frames, diffs, labels in val_loader:
                frames = frames.to(DEVICE)
                diffs  = diffs.to(DEVICE)
                labels = labels.to(DEVICE).long()

                logits, features = model(frames, diffs)
                loss      = criterion(logits, labels)
                val_loss += loss.item() * frames.size(0)
                preds     = torch.argmax(logits, dim=1)
                corrects += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.extend(features.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc   = corrects / len(val_loader.dataset)

        hist_val_loss.append(val_loss)
        hist_val_acc.append(val_acc * 100)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"  LR={lr:.6f} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.4f}%")

        save_classification_report(all_preds, all_labels, class_names, SAVE_DIR,
                                   epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        should_gradcam = False

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_acc.pth"))
            print(f"  [ACC] ★ Best Acc: {val_acc*100:.2f}%")
            plot_confusion_matrix(all_preds, all_labels, class_names, SAVE_DIR)
            save_classification_report(all_preds, all_labels, class_names, SAVE_DIR)
            plot_tsne(all_features, all_labels, class_names, SAVE_DIR, epoch=epoch)
            should_gradcam = True

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_loss.pth"))
            print(f"  [LOSS] ★ Best Loss: {val_loss:.4f}")
            should_gradcam = True

        if should_gradcam:
            generate_gradcam(model, val_loader, DEVICE, SAVE_DIR, epoch, class_names)

        plot_training_curves(hist_train, hist_val_loss, hist_val_acc, SAVE_DIR)

    print(f"\n[✓] Done! Kết quả: {SAVE_DIR}")


if __name__ == "__main__":
    train_model()
