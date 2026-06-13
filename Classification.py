import argparse, shutil, sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_Anphax import Simple_GLKA, GLKA

DEFAULT_MODEL  = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\runs\Anphax\best_acc.pth"
DEFAULT_INPUT  = r"C:\Users\ThisPC\Desktop\Datatrain_flow\video_da_lam_sang_no1"
DEFAULT_OUTPUT = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\dataset\Class_pro"

LABELS   = ["thoang", "ket"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(pth_path):
    sd    = torch.load(pth_path, map_location=DEVICE, weights_only=True)
    model = Simple_GLKA(num_classes=2).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    for m in model.modules():
        if isinstance(m, GLKA):
            m.switch_to_deploy()
    model.eval()
    return model

@torch.no_grad()
def classify(model, img_path):
    x = TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    logits, _ = model(x)
    return LABELS[torch.softmax(logits, 1)[0].argmax().item()]

def main(model_path, input_dir, output_dir):
    input_path = Path(input_dir)
    out_root   = Path(output_dir)
    for lb in LABELS:
        (out_root / lb).mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Output : {out_root}\n")

    model = load_model(model_path)
    imgs  = [p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Tìm thấy {len(imgs)} ảnh\n")

    counters = {lb: 0 for lb in LABELS}
    seen     = {}

    for img_path in tqdm(imgs, desc="Phân loại"):
        label = classify(model, img_path)
        rel   = img_path.relative_to(input_path)
        flat  = "_".join(rel.parts)
        dst   = out_root / label / flat
        key   = (label, flat)
        if key in seen:
            seen[key] += 1
            stem, ext = Path(flat).stem, Path(flat).suffix
            dst = out_root / label / f"{stem}_{seen[key]}{ext}"
        else:
            seen[key] = 0
        shutil.copy2(img_path, dst)
        counters[label] += 1

    print(f"\n  thoang : {counters['thoang']} ảnh")
    print(f"  ket    : {counters['ket']} ảnh")
    print(f"  Tổng   : {sum(counters.values())} ảnh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.model, args.input, args.output)