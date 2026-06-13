from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ===== CONFIG =====
VAL_DIR     = "dataset/val"
SAVE_DIR    = "gradcam_results"
IMGSZ       = 224
CLASS_NAMES = ["0_khongket", "1_ketxe"]

fixed_indices = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
os.makedirs(SAVE_DIR, exist_ok=True)

state = {"best_acc": 0.0}  # track qua callback


def generate_gradcam_heatmap(model, device, save_dir, epoch, class_names):
    try:
        model.eval()

        class_images = {i: [] for i in range(len(class_names))}
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(VAL_DIR, class_name)
            for f in sorted(os.listdir(class_dir)):
                img_bgr = cv2.imread(os.path.join(class_dir, f))
                if img_bgr is None:
                    continue
                img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_res   = cv2.resize(img_rgb, (IMGSZ, IMGSZ))
                img_float = img_res.astype(np.float32) / 255.0
                tensor    = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
                class_images[class_idx].append((tensor, img_float))

        selected_images_and_labels = []
        for idx in fixed_indices:
            pair = []
            for class_idx in range(len(class_names)):
                imgs = class_images[class_idx]
                if idx < len(imgs):
                    pair.append((imgs[idx][0], imgs[idx][1], class_idx))
                elif imgs:
                    pair.append((imgs[-1][0], imgs[-1][1], class_idx))
            if len(pair) == len(class_names):
                selected_images_and_labels.append(pair)

        class ModelWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, x):
                out = self.model(x)
                return out[0] if isinstance(out, tuple) else out

        wrapper       = ModelWrapper(model)
        target_layers = [model.model[-2]]
        cam           = GradCAM(model=wrapper, target_layers=target_layers)

        fig, axes = plt.subplots(
            len(class_names),
            len(selected_images_and_labels),
            figsize=(24, 8)
        )

        for col, pair in enumerate(selected_images_and_labels):
            for row, (input_tensor, img_float, label_idx) in enumerate(pair):
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
                img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
                vis = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
                axes[row, col].imshow(vis)
                axes[row, col].set_title(
                    f'{class_names[label_idx]} (idx={fixed_indices[col]})', fontsize=9
                )
                axes[row, col].axis('off')

        plt.suptitle(f'Grad-CAM: 10 cặp ảnh cố định - Epoch {epoch + 1}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'gradcam_epoch_{epoch + 1:02d}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"  [GradCAM] Saved: {out_path}")

    except Exception as e:
        print(f"[!] Lỗi khi sinh Grad-CAM: {e}")
    finally:
        model.train()


def on_fit_epoch_end(trainer):
    # Lấy top1 accuracy từ metrics
    acc = trainer.metrics.get("metrics/accuracy_top1", 0.0)
    epoch = trainer.epoch
    device = next(trainer.model.parameters()).device

    if acc > state["best_acc"]:
        state["best_acc"] = acc
        print(f"  [ACC] Epoch {epoch + 1} → New best acc: {acc:.4f} → Sinh GradCAM...")
        generate_gradcam_heatmap(trainer.model, device, SAVE_DIR, epoch, CLASS_NAMES)


def main():
    model = YOLO("yolo26-cls.yaml")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data="dataset/",
        epochs=25,
        batch=32,
        imgsz=IMGSZ,
        device="0",
        save=True,
        name="traffic_cls",
        exist_ok=True,
        dropout=0.3,
        weight_decay=0.001,
        augment=True,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    freeze_support()
    main()