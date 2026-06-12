"""
3v1_generate_synthetic_masks.py
Generate 5,000 realistic chromosome overlap samples for missing-part prediction.

Fixes in this version:
- No thick contour/border is drawn into the training image or masks.
- The overlap region C is rendered more realistically: whichever chromosome is on top
  is blended with 50% opacity only inside C, with a soft edge like a Photoshop layer/filter.
- Saves full masks A/B/C, visible masks, missing-gap masks, and top-order labels.

Output generated_data/
    images/       : realistic overlap image
    masks_A/      : full A mask
    masks_B/      : full B mask
    masks_C/      : overlap C = A & B
    visible_A/    : visible part of A in final image
    visible_B/    : visible part of B in final image
    gap_A/        : missing/hidden part of A, only nonzero when B_ON_TOP
    gap_B/        : missing/hidden part of B, only nonzero when A_ON_TOP
    previews/     : colored QC preview only, not used for train
    order_labels.csv : A_ON_TOP/B_ON_TOP metadata
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import random
import cv2
import shutil
import csv

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "prepared_single_chromosomes" / "images_rgba"

OUT_ROOT = ROOT / "generated_data"
OUT_IMAGE_DIR = OUT_ROOT / "images"
OUT_MASK_A_DIR = OUT_ROOT / "masks_A"
OUT_MASK_B_DIR = OUT_ROOT / "masks_B"
OUT_MASK_C_DIR = OUT_ROOT / "masks_C"
OUT_VISIBLE_A_DIR = OUT_ROOT / "visible_A"
OUT_VISIBLE_B_DIR = OUT_ROOT / "visible_B"
OUT_GAP_A_DIR = OUT_ROOT / "gap_A"
OUT_GAP_B_DIR = OUT_ROOT / "gap_B"
OUT_PREVIEW_DIR = OUT_ROOT / "previews"
OUT_LABEL_CSV = OUT_ROOT / "order_labels.csv"

CANVAS_SIZE = 512
NUM_SAMPLES = 5000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BACKGROUND_DIFF_THRESHOLD = 22
MIN_OBJECT_AREA = 100
MIN_OVERLAP_PIXELS = 120
MAX_OVERLAP_RATIO = 0.55

TARGET_LONG_SIDE_MIN = 250
TARGET_LONG_SIDE_MAX = 380

# overlap realism: top object opacity inside C
TOP_OPACITY_IN_C = 0.50
SOFT_EDGE_BLUR_RADIUS = 3.0
NOISE_PROB = 0.30
NOISE_STD = 2.0

CLEAR_OLD_OUTPUT = True

# =========================
# INIT FOLDERS
# =========================
if CLEAR_OLD_OUTPUT and OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for folder in [
    OUT_IMAGE_DIR, OUT_MASK_A_DIR, OUT_MASK_B_DIR, OUT_MASK_C_DIR,
    OUT_VISIBLE_A_DIR, OUT_VISIBLE_B_DIR, OUT_GAP_A_DIR, OUT_GAP_B_DIR,
    OUT_PREVIEW_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return labels == largest_label


def extract_chromosome_object(image_path: Path):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    h, w = rgb.shape[:2]

    if np.min(alpha) < 250:
        mask = alpha > 10
    else:
        corner_size = max(5, min(h, w) // 12)
        corners = np.concatenate([
            rgb[:corner_size, :corner_size].reshape(-1, 3),
            rgb[:corner_size, w-corner_size:w].reshape(-1, 3),
            rgb[h-corner_size:h, :corner_size].reshape(-1, 3),
            rgb[h-corner_size:h, w-corner_size:w].reshape(-1, 3),
        ], axis=0)
        bg_color = np.median(corners, axis=0)
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_color.astype(np.float32), axis=2)
        gray = np.mean(rgb, axis=2)
        bg_gray = float(np.mean(bg_color))
        mask = (diff > BACKGROUND_DIFF_THRESHOLD) | (gray < bg_gray - 8)

    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = keep_largest_component(mask_uint8 > 0)

    if int(mask.sum()) < MIN_OBJECT_AREA:
        return None, None

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 4  # reduced pad: no thick visible border around object
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

    cropped_rgb = rgb[y1:y2+1, x1:x2+1]
    cropped_mask = mask[y1:y2+1, x1:x2+1]

    obj_rgba_arr = np.zeros((cropped_rgb.shape[0], cropped_rgb.shape[1], 4), dtype=np.uint8)
    obj_rgba_arr[:, :, :3] = cropped_rgb
    obj_rgba_arr[:, :, 3] = cropped_mask.astype(np.uint8) * 255

    return Image.fromarray(obj_rgba_arr, "RGBA"), Image.fromarray((cropped_mask.astype(np.uint8) * 255), "L")


def resize_keep_ratio(img: Image.Image, mask: Image.Image, target_long_side: int):
    w, h = img.size
    scale = target_long_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR), mask.resize((new_w, new_h), Image.NEAREST)


def rotate_pair(img: Image.Image, mask: Image.Image, angle: float):
    img_rot = img.rotate(angle, expand=True, resample=Image.BILINEAR, fillcolor=(255, 255, 255, 0))
    mask_rot = mask.rotate(angle, expand=True, resample=Image.NEAREST, fillcolor=0)
    return img_rot, mask_rot


def paste_to_rgba_canvas(obj_rgba: Image.Image, obj_mask: Image.Image, center_x: int, center_y: int):
    """Return a 512 RGBA layer and binary mask for one object."""
    layer = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 0))
    mask_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)

    w, h = obj_rgba.size
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)

    layer.alpha_composite(obj_rgba.convert("RGBA"), dest=(x, y))

    mask_arr = np.array(mask_canvas)
    obj_mask_arr = np.array(obj_mask.convert("L"))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(CANVAS_SIZE, x + w), min(CANVAS_SIZE, y + h)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if x1 < x2 and y1 < y2:
        region = mask_arr[y1:y2, x1:x2]
        obj_region = obj_mask_arr[oy1:oy2, ox1:ox2]
        region[obj_region > 0] = 255
        mask_arr[y1:y2, x1:x2] = region

    return layer, Image.fromarray(mask_arr, "L")


def alpha_over(base_rgb: np.ndarray, top_rgba: np.ndarray, alpha_override: np.ndarray | None = None) -> np.ndarray:
    """Alpha composite top over base. base RGB uint8, top RGBA uint8."""
    base = base_rgb.astype(np.float32)
    top_rgb = top_rgba[:, :, :3].astype(np.float32)
    alpha = top_rgba[:, :, 3].astype(np.float32) / 255.0
    if alpha_override is not None:
        alpha = alpha * np.clip(alpha_override.astype(np.float32), 0.0, 1.0)
    alpha = alpha[:, :, None]
    return np.clip(top_rgb * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def composite_realistic(layer_A: Image.Image, layer_B: Image.Image, mask_A: Image.Image, mask_B: Image.Image, top_label: str):
    """
    Composite using opacity 50% in C for the top layer.
    top_label: A_ON_TOP or B_ON_TOP.
    """
    A = np.array(mask_A) > 0
    B = np.array(mask_B) > 0
    C = A & B

    base = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    arr_A = np.array(layer_A.convert("RGBA"))
    arr_B = np.array(layer_B.convert("RGBA"))

    if top_label == "A_ON_TOP":
        lower_rgba, top_rgba = arr_B, arr_A
    else:
        lower_rgba, top_rgba = arr_A, arr_B

    # lower full opacity first
    out = alpha_over(base, lower_rgba)

    # Soft C map: only reduce opacity inside overlap, with blurred/soft edge.
    c_img = Image.fromarray((C.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(radius=SOFT_EDGE_BLUR_RADIUS))
    c_soft = np.array(c_img).astype(np.float32) / 255.0

    # Photoshop-like filter: top layer normal outside C, 50% opacity in C.
    # alpha_factor = 1 outside C, 0.5 inside C, soft transition near edge.
    alpha_factor = 1.0 - c_soft * (1.0 - TOP_OPACITY_IN_C)

    # Slight blur only for the top layer texture inside C to reduce fake hard overlap.
    top_img = Image.fromarray(top_rgba[:, :, :3], "RGB")
    top_blur = np.array(top_img.filter(ImageFilter.GaussianBlur(radius=1.15))).astype(np.uint8)
    c3 = c_soft[:, :, None]
    top_rgba_soft = top_rgba.copy()
    top_rgba_soft[:, :, :3] = np.clip(top_rgba[:, :, :3] * (1 - c3) + top_blur * c3, 0, 255).astype(np.uint8)

    out = alpha_over(out, top_rgba_soft, alpha_override=alpha_factor)

    if random.random() < NOISE_PROB:
        noise = np.random.normal(0, NOISE_STD, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(out, "RGB"), C


def make_preview(image: Image.Image, mask_A: Image.Image, mask_B: Image.Image, mask_C: Image.Image, top_label: str):
    base = np.array(image.convert("RGB")).astype(np.float32)
    A = np.array(mask_A) > 0
    B = np.array(mask_B) > 0
    C = np.array(mask_C) > 0
    overlay = base.copy()
    overlay[A] = overlay[A] * 0.70 + np.array([255, 0, 0]) * 0.30
    overlay[B] = overlay[B] * 0.70 + np.array([0, 255, 0]) * 0.30
    overlay[C] = overlay[C] * 0.50 + np.array([255, 255, 0]) * 0.50
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    # Add a small text header outside model training use only.
    return Image.fromarray(overlay, "RGB")


def make_sample(obj_A, mask_A, obj_B, mask_B):
    target_A = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    target_B = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    obj_A, mask_A = resize_keep_ratio(obj_A, mask_A, target_A)
    obj_B, mask_B = resize_keep_ratio(obj_B, mask_B, target_B)

    angle_A = 90 + random.uniform(-10, 10)
    angle_B = random.uniform(-10, 10)
    obj_A, mask_A = rotate_pair(obj_A, mask_A, angle_A)
    obj_B, mask_B = rotate_pair(obj_B, mask_B, angle_B)

    center_x = CANVAS_SIZE // 2 + random.randint(-16, 16)
    center_y = CANVAS_SIZE // 2 + random.randint(-16, 16)
    A_cx, A_cy = center_x + random.randint(-20, 20), center_y + random.randint(-12, 12)
    B_cx, B_cy = center_x + random.randint(-12, 12), center_y + random.randint(-20, 20)

    layer_A, mask_A_canvas = paste_to_rgba_canvas(obj_A, mask_A, A_cx, A_cy)
    layer_B, mask_B_canvas = paste_to_rgba_canvas(obj_B, mask_B, B_cx, B_cy)

    A_arr = np.array(mask_A_canvas) > 0
    B_arr = np.array(mask_B_canvas) > 0
    C_arr = A_arr & B_arr
    overlap_pixels = int(C_arr.sum())
    area_A, area_B = max(1, int(A_arr.sum())), max(1, int(B_arr.sum()))
    overlap_ratio = overlap_pixels / min(area_A, area_B)

    if overlap_pixels < MIN_OVERLAP_PIXELS or overlap_ratio > MAX_OVERLAP_RATIO:
        return None

    top_label = "A_ON_TOP" if random.random() < 0.5 else "B_ON_TOP"
    final_image, C_arr = composite_realistic(layer_A, layer_B, mask_A_canvas, mask_B_canvas, top_label)

    # Full masks
    mask_C_canvas = Image.fromarray((C_arr.astype(np.uint8) * 255), "L")

    # Visible masks and missing gaps.
    # If A is on top: A visible includes C; B is hidden at C.
    # If B is on top: B visible includes C; A is hidden at C.
    if top_label == "A_ON_TOP":
        visible_A = A_arr
        visible_B = B_arr & (~C_arr)
        gap_A = np.zeros_like(C_arr)
        gap_B = C_arr
    else:
        visible_A = A_arr & (~C_arr)
        visible_B = B_arr
        gap_A = C_arr
        gap_B = np.zeros_like(C_arr)

    return {
        "image": final_image,
        "mask_A": mask_A_canvas,
        "mask_B": mask_B_canvas,
        "mask_C": mask_C_canvas,
        "visible_A": Image.fromarray((visible_A.astype(np.uint8) * 255), "L"),
        "visible_B": Image.fromarray((visible_B.astype(np.uint8) * 255), "L"),
        "gap_A": Image.fromarray((gap_A.astype(np.uint8) * 255), "L"),
        "gap_B": Image.fromarray((gap_B.astype(np.uint8) * 255), "L"),
        "top_label": top_label,
        "top_class": 0 if top_label == "A_ON_TOP" else 1,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio": overlap_ratio,
    }

# =========================
# MAIN
# =========================
def main():
    single_paths = sorted(SOURCE_DIR.glob("*.png"))
    if len(single_paths) < 2:
        raise ValueError(f"Need at least 2 PNG images in {SOURCE_DIR}")

    print(f"Found {len(single_paths)} single chromosome PNG images.")
    print(f"Generating {NUM_SAMPLES} realistic overlap samples...")
    print("Top layer in C is blended at 50% opacity; no contour/border is drawn into train image.")

    rows = []
    created = 0
    attempts = 0
    max_attempts = NUM_SAMPLES * 80

    while created < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1
        path_A, path_B = random.sample(single_paths, 2)
        obj_A, mask_A = extract_chromosome_object(path_A)
        obj_B, mask_B = extract_chromosome_object(path_B)
        if obj_A is None or obj_B is None:
            continue

        sample = make_sample(obj_A, mask_A, obj_B, mask_B)
        if sample is None:
            continue

        created += 1
        name = f"img_{created:06d}.png"

        sample["image"].save(OUT_IMAGE_DIR / name)
        sample["mask_A"].save(OUT_MASK_A_DIR / name)
        sample["mask_B"].save(OUT_MASK_B_DIR / name)
        sample["mask_C"].save(OUT_MASK_C_DIR / name)
        sample["visible_A"].save(OUT_VISIBLE_A_DIR / name)
        sample["visible_B"].save(OUT_VISIBLE_B_DIR / name)
        sample["gap_A"].save(OUT_GAP_A_DIR / name)
        sample["gap_B"].save(OUT_GAP_B_DIR / name)

        preview = make_preview(sample["image"], sample["mask_A"], sample["mask_B"], sample["mask_C"], sample["top_label"])
        preview.save(OUT_PREVIEW_DIR / name)

        rows.append({
            "filename": name,
            "source_A": path_A.name,
            "source_B": path_B.name,
            "top_label": sample["top_label"],
            "top_class": sample["top_class"],
            "overlap_pixels": sample["overlap_pixels"],
            "overlap_ratio": round(float(sample["overlap_ratio"]), 6),
        })

        if created % 100 == 0:
            print(f"Created {created}/{NUM_SAMPLES}")

    with open(OUT_LABEL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_A", "source_B", "top_label", "top_class", "overlap_pixels", "overlap_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Created: {created}")
    print(f"Attempts: {attempts}")
    print(f"Labels : {OUT_LABEL_CSV}")
    if created < NUM_SAMPLES:
        print("Warning: Could not create enough samples. Try lowering MIN_OVERLAP_PIXELS.")


if __name__ == "__main__":
    main()
