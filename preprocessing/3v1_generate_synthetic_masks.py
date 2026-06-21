"""
3v1_generate_synthetic_masks.py
Fast parallel version for generating realistic chromosome overlap samples.

What changed vs the slow version:
- Uses ThreadPoolExecutor (--workers) to generate/save samples concurrently.
- Uses low PNG compression by default (--png-compress-level 1) to avoid wasting CPU time.
- Skips preview images by default because previews are not used for training.
  Use --preview-mode all or --preview-mode first if you need QC previews.

Final logic is unchanged:
- No thick border/contour is drawn into train images or masks.
- C = A & B.
- In C, the top chromosome is blended at 50% opacity with soft edge/blur.
- Saves full masks A/B/C, visible A/B, missing gaps A/B, and top-order labels.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

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
BACKGROUND_DIFF_THRESHOLD = 22
MIN_OBJECT_AREA = 100
MIN_OVERLAP_PIXELS = 120
MAX_OVERLAP_RATIO = 0.55
TARGET_LONG_SIDE_MIN = 250
TARGET_LONG_SIDE_MAX = 380
TOP_OPACITY_IN_C = 0.50
SOFT_EDGE_BLUR_RADIUS = 3.0
NOISE_PROB = 0.30
NOISE_STD = 2.0
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def reset_output(clear_old: bool) -> None:
    if clear_old and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for folder in [
        OUT_IMAGE_DIR, OUT_MASK_A_DIR, OUT_MASK_B_DIR, OUT_MASK_C_DIR,
        OUT_VISIBLE_A_DIR, OUT_VISIBLE_B_DIR, OUT_GAP_A_DIR, OUT_GAP_B_DIR,
        OUT_PREVIEW_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


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

    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = keep_largest_component(mask_uint8 > 0)

    if int(mask.sum()) < MIN_OBJECT_AREA:
        return None

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 1
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

    cropped_rgb = rgb[y1:y2+1, x1:x2+1]
    cropped_mask = mask[y1:y2+1, x1:x2+1]

    obj_rgba_arr = np.zeros((cropped_rgb.shape[0], cropped_rgb.shape[1], 4), dtype=np.uint8)
    obj_rgba_arr[:, :, :3] = cropped_rgb
    obj_rgba_arr[:, :, 3] = cropped_mask.astype(np.uint8) * 255

    return (
        image_path,
        Image.fromarray(obj_rgba_arr, "RGBA"),
        Image.fromarray(cropped_mask.astype(np.uint8) * 255, "L"),
    )


def load_source_objects(progress_every: int = 500):
    paths = sorted([p for p in SOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 chromosome images in {SOURCE_DIR}")
    print(f"[3v1] Found {len(paths)} prepared single chromosome images.")
    print("[3v1] Caching extracted chromosome objects into RAM for faster generation...")
    objects = []
    skipped = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        try:
            item = extract_chromosome_object(p)
            if item is None:
                skipped += 1
            else:
                objects.append(item)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"[SKIP] {p.name}: {e}")
        if i % progress_every == 0 or i == len(paths):
            print(f"[3v1][cache] {i}/{len(paths)} | ok={len(objects)} | skipped={skipped} | elapsed={time.time()-t0:.1f}s", flush=True)
    if len(objects) < 2:
        raise RuntimeError("Not enough valid chromosome objects after extraction.")
    return objects


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
    base = base_rgb.astype(np.float32)
    top_rgb = top_rgba[:, :, :3].astype(np.float32)
    alpha = top_rgba[:, :, 3].astype(np.float32) / 255.0
    if alpha_override is not None:
        alpha = alpha * np.clip(alpha_override.astype(np.float32), 0.0, 1.0)
    alpha = alpha[:, :, None]
    return np.clip(top_rgb * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def composite_realistic(layer_A: Image.Image, layer_B: Image.Image, mask_A: Image.Image, mask_B: Image.Image,
                        top_label: str, rng: random.Random, np_rng: np.random.Generator):
    A = np.array(mask_A) > 0
    B = np.array(mask_B) > 0
    C = A & B
    base = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    arr_A = np.array(layer_A.convert("RGBA"))
    arr_B = np.array(layer_B.convert("RGBA"))
    lower_rgba, top_rgba = (arr_B, arr_A) if top_label == "A_ON_TOP" else (arr_A, arr_B)

    out = alpha_over(base, lower_rgba)

    c_soft = np.array(
        Image.fromarray(C.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(radius=SOFT_EDGE_BLUR_RADIUS))
    ).astype(np.float32) / 255.0
    alpha_factor = 1.0 - c_soft * (1.0 - TOP_OPACITY_IN_C)

    top_blur = np.array(Image.fromarray(top_rgba[:, :, :3], "RGB").filter(ImageFilter.GaussianBlur(radius=1.15))).astype(np.uint8)
    c3 = c_soft[:, :, None]
    top_rgba_soft = top_rgba.copy()
    top_rgba_soft[:, :, :3] = np.clip(top_rgba[:, :, :3] * (1 - c3) + top_blur * c3, 0, 255).astype(np.uint8)
    out = alpha_over(out, top_rgba_soft, alpha_override=alpha_factor)

    if rng.random() < NOISE_PROB:
        out = np.clip(out.astype(np.float32) + np_rng.normal(0, NOISE_STD, out.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), C


def make_preview(image: Image.Image, mask_A: Image.Image, mask_B: Image.Image, mask_C: Image.Image):
    base = np.array(image.convert("RGB")).astype(np.float32)
    A = np.array(mask_A) > 0
    B = np.array(mask_B) > 0
    C = np.array(mask_C) > 0
    overlay = base.copy()
    overlay[A] = overlay[A] * 0.70 + np.array([255, 0, 0]) * 0.30
    overlay[B] = overlay[B] * 0.70 + np.array([0, 255, 0]) * 0.30
    overlay[C] = overlay[C] * 0.50 + np.array([255, 255, 0]) * 0.50
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")


def make_sample(item_A, item_B, rng: random.Random, np_rng: np.random.Generator):
    path_A, obj_A, mask_A = item_A
    path_B, obj_B, mask_B = item_B
    target_A = rng.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    target_B = rng.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    obj_A, mask_A = resize_keep_ratio(obj_A, mask_A, target_A)
    obj_B, mask_B = resize_keep_ratio(obj_B, mask_B, target_B)
    obj_A, mask_A = rotate_pair(obj_A, mask_A, 90 + rng.uniform(-10, 10))
    obj_B, mask_B = rotate_pair(obj_B, mask_B, rng.uniform(-10, 10))

    center_x = CANVAS_SIZE // 2 + rng.randint(-16, 16)
    center_y = CANVAS_SIZE // 2 + rng.randint(-16, 16)
    A_cx, A_cy = center_x + rng.randint(-20, 20), center_y + rng.randint(-12, 12)
    B_cx, B_cy = center_x + rng.randint(-12, 12), center_y + rng.randint(-20, 20)
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

    top_label = "A_ON_TOP" if rng.random() < 0.5 else "B_ON_TOP"
    final_image, C_arr = composite_realistic(layer_A, layer_B, mask_A_canvas, mask_B_canvas, top_label, rng, np_rng)
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
        "mask_C": Image.fromarray(C_arr.astype(np.uint8) * 255, "L"),
        "visible_A": Image.fromarray(visible_A.astype(np.uint8) * 255, "L"),
        "visible_B": Image.fromarray(visible_B.astype(np.uint8) * 255, "L"),
        "gap_A": Image.fromarray(gap_A.astype(np.uint8) * 255, "L"),
        "gap_B": Image.fromarray(gap_B.astype(np.uint8) * 255, "L"),
        "source_A": path_A.name,
        "source_B": path_B.name,
        "top_label": top_label,
        "top_class": 0 if top_label == "A_ON_TOP" else 1,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio": overlap_ratio,
    }


def save_png(img: Image.Image, path: Path, compress_level: int) -> None:
    # Lower compression is much faster. Training does not care about PNG file size.
    img.save(path, format="PNG", compress_level=int(compress_level), optimize=False)


def should_save_preview(index: int, preview_mode: str, preview_first: int) -> bool:
    if preview_mode == "all":
        return True
    if preview_mode == "first" and index <= preview_first:
        return True
    return False


def worker_make_and_save(index: int, seed: int, objects, max_attempts_per_sample: int,
                         preview_mode: str, preview_first: int, png_compress_level: int):
    # Different deterministic RNG per sample index, safe for parallel execution.
    local_seed = int(seed) + int(index) * 1_000_003
    rng = random.Random(local_seed)
    np_rng = np.random.default_rng(local_seed)

    attempts = 0
    for attempts in range(1, max_attempts_per_sample + 1):
        i_a, i_b = rng.sample(range(len(objects)), 2)
        sample = make_sample(objects[i_a], objects[i_b], rng, np_rng)
        if sample is None:
            continue

        name = f"img_{index:06d}.png"
        save_png(sample["image"], OUT_IMAGE_DIR / name, png_compress_level)
        save_png(sample["mask_A"], OUT_MASK_A_DIR / name, png_compress_level)
        save_png(sample["mask_B"], OUT_MASK_B_DIR / name, png_compress_level)
        save_png(sample["mask_C"], OUT_MASK_C_DIR / name, png_compress_level)
        save_png(sample["visible_A"], OUT_VISIBLE_A_DIR / name, png_compress_level)
        save_png(sample["visible_B"], OUT_VISIBLE_B_DIR / name, png_compress_level)
        save_png(sample["gap_A"], OUT_GAP_A_DIR / name, png_compress_level)
        save_png(sample["gap_B"], OUT_GAP_B_DIR / name, png_compress_level)

        if should_save_preview(index, preview_mode, preview_first):
            preview = make_preview(sample["image"], sample["mask_A"], sample["mask_B"], sample["mask_C"])
            save_png(preview, OUT_PREVIEW_DIR / name, png_compress_level)

        row = {
            "filename": name,
            "source_A": sample["source_A"],
            "source_B": sample["source_B"],
            "top_label": sample["top_label"],
            "top_class": sample["top_class"],
            "overlap_pixels": sample["overlap_pixels"],
            "overlap_ratio": round(float(sample["overlap_ratio"]), 6),
        }
        return {"ok": True, "index": index, "attempts": attempts, "row": row, "error": ""}

    return {"ok": False, "index": index, "attempts": attempts, "row": None, "error": "max attempts reached"}


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return ivalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=positive_int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear-old", action="store_true", default=True)
    ap.add_argument("--no-clear-old", dest="clear_old", action="store_false")
    ap.add_argument("--progress-every", type=positive_int, default=100)
    ap.add_argument("--max-attempts-per-sample", type=positive_int, default=80)
    ap.add_argument("--workers", type=positive_int, default=max(2, min(8, (os.cpu_count() or 2))))
    ap.add_argument("--png-compress-level", type=int, default=1, choices=list(range(10)))
    ap.add_argument("--preview-mode", choices=["none", "first", "all"], default="none")
    ap.add_argument("--preview-first", type=positive_int, default=60)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    reset_output(args.clear_old)

    print("=" * 80)
    print("3v1 GENERATE REALISTIC SYNTHETIC OVERLAP DATASET - PARALLEL FAST")
    print("=" * 80)
    print(f"Samples target    : {args.num_samples}")
    print(f"Output folder     : {OUT_ROOT}")
    print(f"Workers           : {args.workers}")
    print(f"PNG compression   : {args.png_compress_level}  (lower = faster, bigger files)")
    print(f"Preview mode      : {args.preview_mode}")
    print("Logic             : no contour; C region uses 50% opacity top-layer blend")

    objects = load_source_objects(progress_every=max(100, args.progress_every))

    rows = []
    created = 0
    failed = 0
    total_attempts = 0
    t0 = time.time()

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx in range(1, args.num_samples + 1):
            futures.append(executor.submit(
                worker_make_and_save,
                idx,
                args.seed,
                objects,
                args.max_attempts_per_sample,
                args.preview_mode,
                args.preview_first,
                args.png_compress_level,
            ))

        for fut in as_completed(futures):
            result = fut.result()
            total_attempts += int(result["attempts"])
            if result["ok"]:
                created += 1
                rows.append(result["row"])
            else:
                failed += 1
                if failed <= 20:
                    print(f"[3v1][FAIL] index={result['index']:06d}: {result['error']}", flush=True)

            done = created + failed
            if created % args.progress_every == 0 or done == args.num_samples:
                elapsed = time.time() - t0
                rate = created / max(elapsed, 1e-6)
                print(
                    f"[3v1] created={created}/{args.num_samples} failed={failed} "
                    f"attempts={total_attempts} rate={rate:.2f} img/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    rows = sorted(rows, key=lambda r: r["filename"])
    with open(OUT_LABEL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "source_A", "source_B", "top_label", "top_class", "overlap_pixels", "overlap_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("3v1 DONE")
    print("=" * 80)
    print(f"Created : {created}")
    print(f"Failed  : {failed}")
    print(f"Attempts: {total_attempts}")
    print(f"Labels  : {OUT_LABEL_CSV}")

    if created < args.num_samples:
        raise RuntimeError(
            f"Only created {created}/{args.num_samples}. "
            f"Try increasing --max-attempts-per-sample or lowering MIN_OVERLAP_PIXELS."
        )


if __name__ == "__main__":
    main()
