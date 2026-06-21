"""
4v1_preprocess_to_256.py
Resize generated realistic overlap dataset to 256x256.

Supports extended masks:
- masks_A, masks_B, masks_C
- visible_A, visible_B
- gap_A, gap_B
Copies order_labels.csv to processed_data_256.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "generated_data"
OUTPUT_ROOT = ROOT / "processed_data_256"
TARGET_SIZE = 256
CLEAR_OLD_OUTPUT = True

IMAGE_SUBDIR = "images"
MASK_SUBDIRS = ["masks_A", "masks_B", "masks_C", "visible_A", "visible_B", "gap_A", "gap_B"]

if CLEAR_OLD_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

(OUTPUT_ROOT / IMAGE_SUBDIR).mkdir(parents=True, exist_ok=True)
for sub in MASK_SUBDIRS:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def resize_with_padding(img, target_size=256, is_mask=False):
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if is_mask:
        img = img.resize((new_w, new_h), Image.NEAREST)
        canvas = Image.new("L", (target_size, target_size), 0)
    else:
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (target_size, target_size), 255)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas


def binarize_mask(mask_img):
    arr = np.array(mask_img)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")

image_paths = sorted((INPUT_ROOT / IMAGE_SUBDIR).glob("*.png"))
print(f"Found {len(image_paths)} generated images.")
if not image_paths:
    raise FileNotFoundError("No generated images found. Run 3v1_generate_synthetic_masks.py first.")

for idx, img_path in enumerate(image_paths, start=1):
    name = img_path.name

    missing = [sub for sub in MASK_SUBDIRS if not (INPUT_ROOT / sub / name).exists()]
    if missing:
        print(f"[SKIP] Missing {missing} for {name}")
        continue

    img = Image.open(img_path).convert("L")
    img = resize_with_padding(img, target_size=TARGET_SIZE, is_mask=False)
    img.save(OUTPUT_ROOT / IMAGE_SUBDIR / name)

    for sub in MASK_SUBDIRS:
        mask = Image.open(INPUT_ROOT / sub / name).convert("L")
        mask = binarize_mask(mask)
        mask = resize_with_padding(mask, target_size=TARGET_SIZE, is_mask=True)
        mask = binarize_mask(mask)
        mask.save(OUTPUT_ROOT / sub / name)

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(image_paths)}")

label_csv = INPUT_ROOT / "order_labels.csv"
if label_csv.exists():
    shutil.copy2(label_csv, OUTPUT_ROOT / "order_labels.csv")
    print(f"Copied labels: {OUTPUT_ROOT / 'order_labels.csv'}")

print("Done preprocessing to 256x256.")
