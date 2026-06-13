import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_Anphax import Simple_GLKA

DEFAULT_MODEL  = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\runs\Anphax\best_acc.pth"
DEFAULT_INPUT  = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\dataset\Class_pro\test"
DEFAULT_OUTPUT = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\dataset\Class_pro\gradcam_out"

LABELS   = ["thoang", "ket"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Grad-CAM ──────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._feat = None
        self._grad = None
        self._fh = target_layer.register_forward_hook(self._save_feat)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_feat(self, _, __, out):  self._feat = out.detach()
    def _save_grad(self, _, __, g):    self._grad = g[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        logits, _ = self.model(x)
        cls_idx   = logits.argmax(dim=1).item()
        logits[0, cls_idx].backward()

        w   = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._feat).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy(), cls_idx

    def remove(self):
        self._fh.remove(); self._bh.remove()


def make_overlay(img_path, cam):
    img     = cv2.imread(str(img_path))
    img     = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (224, 224))), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)


def load_model(pth_path):
    sd    = torch.load(pth_path, map_location=DEVICE, weights_only=True)
    model = Simple_GLKA(num_classes=2).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    # eval() nhưng bật grad để backward được
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def main(model_path, input_dir, output_dir):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}\n")

    model        = load_model(model_path)
    target_layer = model.blocks[-1].project[0]   # Conv2d 128→256, layer cuối trước pool
    gradcam      = GradCAM(model, target_layer)

    imgs = [p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Tìm thấy {len(imgs)} ảnh\n")

    for img_path in tqdm(imgs, desc="Grad-CAM"):
        try:
            x               = TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            cam, cls_idx    = gradcam(x)
            label           = LABELS[cls_idx]
            overlay         = make_overlay(img_path, cam)
            cv2.putText(overlay, label, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            rel  = img_path.relative_to(input_path)
            flat = "_".join(rel.parts)
            cv2.imwrite(str(output_path / flat), overlay)
        except Exception as e:
            tqdm.write(f"  ✗ {img_path.name}: {e}")

    gradcam.remove()
    print(f"\nXong — lưu tại: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.model, args.input, args.output)