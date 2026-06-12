"""
10v2_evaluate_full_missing_pipeline.py
Low-memory / streaming evaluator for the complete NST missing-completion pipeline.

Why this version exists:
- The old evaluator loaded all 1,000 real_test samples and all predictions into RAM at once.
  Colab could be killed with exit=137 even when --batch-size 1.
- This version runs in streaming passes and loads only one model at a time:
    Pass 1: 6v1 visible/order model -> cache visible_A/B/C + predicted order.
    Pass 2: missing Teacher -> cache teacher gap_A/gap_B.
    Pass 3: missing Student -> compute final metrics + save outputs.
- Batch 40 is intended to work normally because RAM/VRAM is bounded by one model + one batch.

Output design:
- predicted_masks/* stores binary black/white masks, useful as supplementary mask evidence.
- predicted_original_canvas/* stores A/B/C/full A/full B cut from the original overlap image,
  so the visible result is image-like, not only white masks on black background.
- predicted_original_crop/* stores tight crops of the same original-image regions.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

# Keep TensorFlow quieter and avoid unnecessary global XLA memory pressure.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

IMG_SIZE = 256
EPS = 1e-7


def read_labels(split_dir: Path) -> dict[str, int]:
    label_csv = split_dir / "order_labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing {label_csv}")
    labels: dict[str, int] = {}
    with open(label_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = int(row["top_class"])
    return labels


def list_real_test_names(dataset_dir: Path) -> tuple[Path, list[str], dict[str, int]]:
    split_dir = dataset_dir / "real_test"
    labels = read_labels(split_dir)
    image_paths = sorted((split_dir / "images").glob("*.png"))
    names = [p.name for p in image_paths if p.name in labels]
    if not names:
        raise FileNotFoundError(f"No real_test samples in {split_dir}")
    return split_dir, names, labels


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def np_gray(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return arr[..., None]


def np_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    return (arr > 127).astype(np.float32)


def load_x(split_dir: Path, names: list[str]) -> np.ndarray:
    return np.stack([np_gray(split_dir / "images" / n) for n in names]).astype(np.float32)


def load_gt(split_dir: Path, names: list[str], labels: dict[str, int]):
    y_visible = []
    y_full = []
    y_gap = []
    y_order = []
    for n in names:
        va = np_mask(split_dir / "visible_A" / n)
        vb = np_mask(split_dir / "visible_B" / n)
        c = np_mask(split_dir / "masks_C" / n)
        ma = np_mask(split_dir / "masks_A" / n)
        mb = np_mask(split_dir / "masks_B" / n)
        ga = np_mask(split_dir / "gap_A" / n)
        gb = np_mask(split_dir / "gap_B" / n)
        y_visible.append(np.stack([va, vb, c], axis=-1))
        y_full.append(np.stack([ma, mb], axis=-1))
        y_gap.append(np.stack([ga, gb], axis=-1))
        y_order.append(labels[n])
    return (
        np.stack(y_visible).astype(np.float32),
        np.stack(y_full).astype(np.float32),
        np.stack(y_gap).astype(np.float32),
        np.array(y_order, dtype=np.int32),
    )


def load_visible_cache(cache_visible: Path, names: list[str]) -> np.ndarray:
    out = []
    for n in names:
        out.append(np.stack([
            np_mask(cache_visible / "visible_A" / n),
            np_mask(cache_visible / "visible_B" / n),
            np_mask(cache_visible / "C_overlap" / n),
        ], axis=-1))
    return np.stack(out).astype(np.float32)


def load_teacher_gap_cache(cache_teacher: Path, names: list[str]) -> np.ndarray:
    out = []
    for n in names:
        out.append(np.stack([
            np_mask(cache_teacher / "gap_A" / n),
            np_mask(cache_teacher / "gap_B" / n),
        ], axis=-1))
    return np.stack(out).astype(np.float32)


def save_mask(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr > 0.5).astype(np.uint8) * 255).save(path)


def save_gray01(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def original_region(gray_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Return original-intensity object on black background, not a binary mask."""
    return gray_hw.astype(np.float32) * (mask_hw > 0.5).astype(np.float32)


def save_original_region(gray_hw: np.ndarray, mask_hw: np.ndarray, canvas_path: Path, crop_path: Path | None = None, pad: int = 3) -> None:
    region = original_region(gray_hw, mask_hw)
    save_gray01(region, canvas_path)

    if crop_path is None:
        return
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    ys, xs = np.where(mask_hw > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        crop = np.zeros((16, 16), dtype=np.float32)
    else:
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, region.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, region.shape[1])
        crop = region[y0:y1, x0:x1]
    save_gray01(crop, crop_path)


def unpack_visible_order_prediction(pred):
    """Return (seg_prob, order_prob) for Keras dict/list outputs."""
    if isinstance(pred, dict):
        return pred["seg"], pred["order"]
    if isinstance(pred, (list, tuple)):
        seg = None
        order = None
        for item in pred:
            arr = np.asarray(item)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                seg = item
            elif arr.ndim == 2 and arr.shape[-1] == 2:
                order = item
        if seg is not None and order is not None:
            return seg, order
    raise ValueError(f"Cannot unpack visible/order prediction outputs: {type(pred)}")


def build_missing_inputs(x_gray: np.ndarray, pred_visible: np.ndarray, pred_order: np.ndarray):
    top_a = (pred_order == 0).astype(np.float32)[:, None, None, None]
    top_b = (pred_order == 1).astype(np.float32)[:, None, None, None]
    top_a_map = np.ones_like(x_gray) * top_a
    top_b_map = np.ones_like(x_gray) * top_b
    x_teacher = np.concatenate([x_gray, pred_visible, top_a_map, top_b_map], axis=-1).astype(np.float32)
    x_student = np.concatenate([x_gray, pred_visible], axis=-1).astype(np.float32)
    return x_teacher, x_student


def reconstruct_full(pred_visible: np.ndarray, pred_gap: np.ndarray, pred_order: np.ndarray) -> np.ndarray:
    va = pred_visible[..., 0]
    vb = pred_visible[..., 1]
    c = pred_visible[..., 2]
    ga = pred_gap[..., 0]
    gb = pred_gap[..., 1]

    rec_a = np.maximum(va, ga)
    rec_b = np.maximum(vb, gb)

    for i, order in enumerate(pred_order):
        if int(order) == 0:  # A_ON_TOP
            rec_a[i] = np.maximum(rec_a[i], c[i])
            rec_b[i] = np.maximum(rec_b[i], gb[i])
        else:  # B_ON_TOP
            rec_b[i] = np.maximum(rec_b[i], c[i])
            rec_a[i] = np.maximum(rec_a[i], ga[i])

    return np.stack([rec_a, rec_b], axis=-1).astype(np.float32)


def make_c_rule_candidates(pred_visible: np.ndarray, pred_gap: np.ndarray):
    va = pred_visible[..., 0]
    vb = pred_visible[..., 1]
    c = pred_visible[..., 2]
    ga = pred_gap[..., 0]
    gb = pred_gap[..., 1]

    a_top_A = np.maximum(va, c)
    a_top_B = np.maximum(vb, gb)
    b_top_A = np.maximum(va, ga)
    b_top_B = np.maximum(vb, c)

    cand_a_top = np.stack([a_top_A, a_top_B], axis=-1).astype(np.float32)
    cand_b_top = np.stack([b_top_A, b_top_B], axis=-1).astype(np.float32)
    return cand_a_top, cand_b_top


class DiceAccumulator:
    def __init__(self, channels: int):
        self.inter = np.zeros(channels, dtype=np.float64)
        self.den = np.zeros(channels, dtype=np.float64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.inter += np.sum(y_true * y_pred, axis=(0, 1, 2))
        self.den += np.sum(y_true + y_pred, axis=(0, 1, 2))

    def dice(self) -> np.ndarray:
        return (2 * self.inter + EPS) / (self.den + EPS)


def plot_confusion_from_matrix(conf: np.ndarray, out_path: Path) -> None:
    conf_norm = conf.astype(np.float64) / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(5, 4))
    plt.imshow(conf_norm * 100)
    plt.xticks([0, 1], ["A_TOP", "B_TOP"])
    plt.yticks([0, 1], ["A_TOP", "B_TOP"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("6v1 order prediction confusion (%)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{conf_norm[i, j] * 100:.1f}%", ha="center", va="center")
    plt.colorbar(label="Percent")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_metric_bar(metrics: dict, out_path: Path) -> None:
    labels = ["Gap mean", "Full A/B mean"]
    teacher = [metrics["teacher_mean_gap_dice_percent"], metrics["teacher_mean_full_AB_dice_percent"]]
    student = [metrics["student_mean_gap_dice_percent"], metrics["student_mean_full_AB_dice_percent"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, teacher, width, label="Teacher")
    plt.bar(x + width / 2, student, width, label="Student")
    plt.xticks(x, labels)
    plt.ylabel("Dice (%)")
    plt.title("Full pipeline: Teacher vs Student on real_test")
    plt.legend()
    for i, v in enumerate(teacher):
        plt.text(i - width / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(student):
        plt.text(i + width / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_showcase(entries: list[dict], best_name: str, out_path: Path) -> None:
    if not entries:
        return
    n = len(entries)
    cols = 8
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.8, max(8, n * 1.15)))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    titles = [
        "Overlap",
        "Pred A original",
        "Pred B original",
        "Pred C original/order",
        "GT missing mask",
        "Teacher missing mask",
        "Student missing mask",
        "Best full A|B original",
    ]
    for i, e in enumerate(entries):
        best_full = e["teacher_full_orig"] if best_name == "teacher" else e["student_full_orig"]
        imgs = [
            e["gray"],
            e["pred_A_orig"],
            e["pred_B_orig"],
            e["pred_C_orig"],
            e["gt_gap_union"],
            e["teacher_gap_union"],
            e["student_gap_union"],
            best_full,
        ]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=7)
        axes[i, 3].text(2, 12, e["order_txt"], color="yellow", fontsize=6,
                        bbox=dict(facecolor="black", alpha=0.45, pad=1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def cache_complete_visible(cache_visible: Path, names: list[str]) -> bool:
    if not (cache_visible / "predicted_order.csv").exists():
        return False
    for n in names:
        if not (cache_visible / "visible_A" / n).exists():
            return False
        if not (cache_visible / "visible_B" / n).exists():
            return False
        if not (cache_visible / "C_overlap" / n).exists():
            return False
    return True


def cache_complete_teacher(cache_teacher: Path, names: list[str]) -> bool:
    for n in names:
        if not (cache_teacher / "gap_A" / n).exists():
            return False
        if not (cache_teacher / "gap_B" / n).exists():
            return False
    return True


def pass1_visible(split_dir: Path, names: list[str], model_path: Path, cache_visible: Path, batch_size: int, force: bool) -> dict[str, dict]:
    if (not force) and cache_complete_visible(cache_visible, names):
        print("Pass 1/3: visible/order cache already complete, skipping 6v1 prediction.")
        return read_pred_order_csv(cache_visible / "predicted_order.csv")

    print("Pass 1/3: Predict visible A/B/C and order with 6v1, cache to disk...")
    for sub in ["visible_A", "visible_B", "C_overlap"]:
        (cache_visible / sub).mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    order_rows = []

    for start, batch_names in batched(names, batch_size):
        x = load_x(split_dir, batch_names)
        pred = model.predict(x, batch_size=batch_size, verbose=0)
        seg_prob, order_prob = unpack_visible_order_prediction(pred)
        pred_visible = (seg_prob >= 0.5).astype(np.float32)
        pred_order = np.argmax(order_prob, axis=1).astype(np.int32)

        for i, n in enumerate(batch_names):
            save_mask(pred_visible[i, ..., 0], cache_visible / "visible_A" / n)
            save_mask(pred_visible[i, ..., 1], cache_visible / "visible_B" / n)
            save_mask(pred_visible[i, ..., 2], cache_visible / "C_overlap" / n)
            order_rows.append({
                "filename": n,
                "pred_order_class": int(pred_order[i]),
                "pred_order": "A_ON_TOP" if int(pred_order[i]) == 0 else "B_ON_TOP",
                "prob_A_ON_TOP": float(order_prob[i, 0]),
                "prob_B_ON_TOP": float(order_prob[i, 1]),
            })
        print(f"  visible cached {min(start + len(batch_names), len(names))}/{len(names)}")
        del x, pred, seg_prob, order_prob, pred_visible, pred_order
        gc.collect()

    with open(cache_visible / "predicted_order.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
        writer.writeheader()
        writer.writerows(order_rows)

    del model
    keras.backend.clear_session()
    gc.collect()
    return {r["filename"]: r for r in order_rows}


def read_pred_order_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["pred_order_class"] = int(row["pred_order_class"])
            row["prob_A_ON_TOP"] = float(row["prob_A_ON_TOP"])
            row["prob_B_ON_TOP"] = float(row["prob_B_ON_TOP"])
            rows[row["filename"]] = row
    return rows


def pass2_teacher(split_dir: Path, names: list[str], teacher_path: Path, cache_visible: Path, order_info: dict[str, dict], cache_teacher: Path, batch_size: int, force: bool) -> None:
    if (not force) and cache_complete_teacher(cache_teacher, names):
        print("Pass 2/3: teacher gap cache already complete, skipping Teacher prediction.")
        return

    print("Pass 2/3: Predict missing gaps with Teacher only, cache to disk...")
    for sub in ["gap_A", "gap_B"]:
        (cache_teacher / sub).mkdir(parents=True, exist_ok=True)

    teacher = keras.models.load_model(teacher_path, compile=False, safe_mode=False)

    for start, batch_names in batched(names, batch_size):
        x = load_x(split_dir, batch_names)
        pred_visible = load_visible_cache(cache_visible, batch_names)
        pred_order = np.array([order_info[n]["pred_order_class"] for n in batch_names], dtype=np.int32)
        x_teacher, _ = build_missing_inputs(x, pred_visible, pred_order)

        pred_gap = (teacher.predict(x_teacher, batch_size=batch_size, verbose=0) >= 0.5).astype(np.float32)
        for i, n in enumerate(batch_names):
            save_mask(pred_gap[i, ..., 0], cache_teacher / "gap_A" / n)
            save_mask(pred_gap[i, ..., 1], cache_teacher / "gap_B" / n)
        print(f"  teacher gap cached {min(start + len(batch_names), len(names))}/{len(names)}")
        del x, pred_visible, pred_order, x_teacher, pred_gap
        gc.collect()

    del teacher
    keras.backend.clear_session()
    gc.collect()


def prepare_output_dirs(pred_root: Path, save_candidates: bool) -> None:
    mask_dirs = [
        "visible_A", "visible_B", "C_overlap",
        "teacher_gap_A", "teacher_gap_B", "teacher_full_A", "teacher_full_B",
        "student_gap_A", "student_gap_B", "student_full_A", "student_full_B",
    ]
    if save_candidates:
        mask_dirs += [
            "candidate_teacher_A_ON_TOP_full_A", "candidate_teacher_A_ON_TOP_full_B",
            "candidate_teacher_B_ON_TOP_full_A", "candidate_teacher_B_ON_TOP_full_B",
            "candidate_student_A_ON_TOP_full_A", "candidate_student_A_ON_TOP_full_B",
            "candidate_student_B_ON_TOP_full_A", "candidate_student_B_ON_TOP_full_B",
        ]
    for s in mask_dirs:
        (pred_root / "predicted_masks" / s).mkdir(parents=True, exist_ok=True)

    original_dirs = [
        "visible_A", "visible_B", "C_overlap",
        "teacher_gap_A", "teacher_gap_B", "teacher_full_A", "teacher_full_B",
        "student_gap_A", "student_gap_B", "student_full_A", "student_full_B",
    ]
    for root_name in ["predicted_original_canvas", "predicted_original_crop"]:
        for s in original_dirs:
            (pred_root / root_name / s).mkdir(parents=True, exist_ok=True)


def pass3_student_metrics_outputs(
    split_dir: Path,
    names: list[str],
    labels: dict[str, int],
    student_path: Path,
    cache_visible: Path,
    cache_teacher: Path,
    order_info: dict[str, dict],
    out_dir: Path,
    batch_size: int,
    max_showcase: int,
    save_candidates: bool,
) -> dict:
    print("Pass 3/3: Predict Student, compute metrics, save original-image outputs...")
    student = keras.models.load_model(student_path, compile=False, safe_mode=False)

    pred_root = out_dir / "predicted_results"
    prepare_output_dirs(pred_root, save_candidates=save_candidates)

    acc_visible = DiceAccumulator(3)
    acc_gap_t = DiceAccumulator(2)
    acc_gap_s = DiceAccumulator(2)
    acc_full_t = DiceAccumulator(2)
    acc_full_s = DiceAccumulator(2)
    visible_correct = 0
    visible_total = 0
    conf = np.zeros((2, 2), dtype=np.int64)
    order_rows = []
    showcase_entries: list[dict] = []

    for start, batch_names in batched(names, batch_size):
        x = load_x(split_dir, batch_names)
        y_visible, y_full, y_gap, y_order = load_gt(split_dir, batch_names, labels)
        pred_visible = load_visible_cache(cache_visible, batch_names)
        pred_order = np.array([order_info[n]["pred_order_class"] for n in batch_names], dtype=np.int32)
        pred_gap_teacher = load_teacher_gap_cache(cache_teacher, batch_names)

        _, x_student = build_missing_inputs(x, pred_visible, pred_order)
        pred_gap_student = (student.predict(x_student, batch_size=batch_size, verbose=0) >= 0.5).astype(np.float32)

        rec_teacher = reconstruct_full(pred_visible, pred_gap_teacher, pred_order)
        rec_student = reconstruct_full(pred_visible, pred_gap_student, pred_order)

        acc_visible.update(y_visible, pred_visible)
        acc_gap_t.update(y_gap, pred_gap_teacher)
        acc_gap_s.update(y_gap, pred_gap_student)
        acc_full_t.update(y_full, rec_teacher)
        acc_full_s.update(y_full, rec_student)
        visible_correct += int(np.sum(y_visible == pred_visible))
        visible_total += int(y_visible.size)
        for t, p in zip(y_order, pred_order):
            conf[int(t), int(p)] += 1

        if save_candidates:
            cand_teacher_A_top, cand_teacher_B_top = make_c_rule_candidates(pred_visible, pred_gap_teacher)
            cand_student_A_top, cand_student_B_top = make_c_rule_candidates(pred_visible, pred_gap_student)
        else:
            cand_teacher_A_top = cand_teacher_B_top = cand_student_A_top = cand_student_B_top = None

        for i, n in enumerate(batch_names):
            gray = x[i, ..., 0]
            stem = Path(n).stem + ".png"
            masks_root = pred_root / "predicted_masks"
            canvas_root = pred_root / "predicted_original_canvas"
            crop_root = pred_root / "predicted_original_crop"

            # Binary masks: supplemental evidence only.
            save_mask(pred_visible[i, ..., 0], masks_root / "visible_A" / stem)
            save_mask(pred_visible[i, ..., 1], masks_root / "visible_B" / stem)
            save_mask(pred_visible[i, ..., 2], masks_root / "C_overlap" / stem)
            save_mask(pred_gap_teacher[i, ..., 0], masks_root / "teacher_gap_A" / stem)
            save_mask(pred_gap_teacher[i, ..., 1], masks_root / "teacher_gap_B" / stem)
            save_mask(rec_teacher[i, ..., 0], masks_root / "teacher_full_A" / stem)
            save_mask(rec_teacher[i, ..., 1], masks_root / "teacher_full_B" / stem)
            save_mask(pred_gap_student[i, ..., 0], masks_root / "student_gap_A" / stem)
            save_mask(pred_gap_student[i, ..., 1], masks_root / "student_gap_B" / stem)
            save_mask(rec_student[i, ..., 0], masks_root / "student_full_A" / stem)
            save_mask(rec_student[i, ..., 1], masks_root / "student_full_B" / stem)

            # Original-image regions: main qualitative result.
            # A/B/C/full outputs are cut from the original overlap image, not plain white masks.
            region_items = {
                "visible_A": pred_visible[i, ..., 0],
                "visible_B": pred_visible[i, ..., 1],
                "C_overlap": pred_visible[i, ..., 2],
                "teacher_gap_A": pred_gap_teacher[i, ..., 0],
                "teacher_gap_B": pred_gap_teacher[i, ..., 1],
                "teacher_full_A": rec_teacher[i, ..., 0],
                "teacher_full_B": rec_teacher[i, ..., 1],
                "student_gap_A": pred_gap_student[i, ..., 0],
                "student_gap_B": pred_gap_student[i, ..., 1],
                "student_full_A": rec_student[i, ..., 0],
                "student_full_B": rec_student[i, ..., 1],
            }
            for folder, mask in region_items.items():
                save_original_region(gray, mask, canvas_root / folder / stem, crop_root / folder / stem)

            if save_candidates:
                save_mask(cand_teacher_A_top[i, ..., 0], masks_root / "candidate_teacher_A_ON_TOP_full_A" / stem)
                save_mask(cand_teacher_A_top[i, ..., 1], masks_root / "candidate_teacher_A_ON_TOP_full_B" / stem)
                save_mask(cand_teacher_B_top[i, ..., 0], masks_root / "candidate_teacher_B_ON_TOP_full_A" / stem)
                save_mask(cand_teacher_B_top[i, ..., 1], masks_root / "candidate_teacher_B_ON_TOP_full_B" / stem)
                save_mask(cand_student_A_top[i, ..., 0], masks_root / "candidate_student_A_ON_TOP_full_A" / stem)
                save_mask(cand_student_A_top[i, ..., 1], masks_root / "candidate_student_A_ON_TOP_full_B" / stem)
                save_mask(cand_student_B_top[i, ..., 0], masks_root / "candidate_student_B_ON_TOP_full_A" / stem)
                save_mask(cand_student_B_top[i, ..., 1], masks_root / "candidate_student_B_ON_TOP_full_B" / stem)

            order_rows.append({
                "filename": n,
                "gt_order": "A_ON_TOP" if int(y_order[i]) == 0 else "B_ON_TOP",
                "pred_order": "A_ON_TOP" if int(pred_order[i]) == 0 else "B_ON_TOP",
                "prob_A_ON_TOP": float(order_info[n]["prob_A_ON_TOP"]),
                "prob_B_ON_TOP": float(order_info[n]["prob_B_ON_TOP"]),
            })

            if len(showcase_entries) < max_showcase:
                teacher_full_orig = np.concatenate([
                    original_region(gray, rec_teacher[i, ..., 0]),
                    original_region(gray, rec_teacher[i, ..., 1]),
                ], axis=1)
                student_full_orig = np.concatenate([
                    original_region(gray, rec_student[i, ..., 0]),
                    original_region(gray, rec_student[i, ..., 1]),
                ], axis=1)
                showcase_entries.append({
                    "gray": gray,
                    "pred_A_orig": original_region(gray, pred_visible[i, ..., 0]),
                    "pred_B_orig": original_region(gray, pred_visible[i, ..., 1]),
                    "pred_C_orig": original_region(gray, pred_visible[i, ..., 2]),
                    "gt_gap_union": np.maximum(y_gap[i, ..., 0], y_gap[i, ..., 1]),
                    "teacher_gap_union": np.maximum(pred_gap_teacher[i, ..., 0], pred_gap_teacher[i, ..., 1]),
                    "student_gap_union": np.maximum(pred_gap_student[i, ..., 0], pred_gap_student[i, ..., 1]),
                    "teacher_full_orig": teacher_full_orig,
                    "student_full_orig": student_full_orig,
                    "order_txt": "A_TOP" if int(pred_order[i]) == 0 else "B_TOP",
                })

        print(f"  final evaluated {min(start + len(batch_names), len(names))}/{len(names)}")
        del x, y_visible, y_full, y_gap, y_order, pred_visible, pred_order, pred_gap_teacher
        del x_student, pred_gap_student, rec_teacher, rec_student
        gc.collect()

    with open(pred_root / "predicted_order.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
        writer.writeheader()
        writer.writerows(order_rows)

    del student
    keras.backend.clear_session()
    gc.collect()

    d_vis = acc_visible.dice()
    d_gap_t = acc_gap_t.dice()
    d_gap_s = acc_gap_s.dice()
    d_full_t = acc_full_t.dice()
    d_full_s = acc_full_s.dice()

    order_acc = float(np.trace(conf) / max(np.sum(conf), 1))
    visible_pixel_accuracy = float(visible_correct / max(visible_total, 1))
    mean_gap_t = float(np.mean(d_gap_t))
    mean_gap_s = float(np.mean(d_gap_s))
    mean_full_t = float(np.mean(d_full_t))
    mean_full_s = float(np.mean(d_full_s))
    best = "teacher" if mean_full_t >= mean_full_s else "student"

    metrics = {
        "real_test_count": int(len(names)),
        "batch_size": int(batch_size),
        "evaluation_mode": "low_memory_streaming_three_passes",
        "visible_order_model": {
            "visible_A_dice_percent": float(d_vis[0] * 100),
            "visible_B_dice_percent": float(d_vis[1] * 100),
            "C_overlap_dice_percent": float(d_vis[2] * 100),
            "mean_visible_ABC_dice_percent": float(np.mean(d_vis) * 100),
            "order_accuracy_percent": float(order_acc * 100),
            "visible_pixel_accuracy_percent": float(visible_pixel_accuracy * 100),
            "order_confusion_matrix": conf.tolist(),
        },
        "teacher_from_6v1_outputs": {
            "gap_A_dice_percent": float(d_gap_t[0] * 100),
            "gap_B_dice_percent": float(d_gap_t[1] * 100),
            "mean_gap_dice_percent": float(mean_gap_t * 100),
            "full_A_dice_percent": float(d_full_t[0] * 100),
            "full_B_dice_percent": float(d_full_t[1] * 100),
            "mean_full_AB_dice_percent": float(mean_full_t * 100),
        },
        "student_from_6v1_outputs": {
            "gap_A_dice_percent": float(d_gap_s[0] * 100),
            "gap_B_dice_percent": float(d_gap_s[1] * 100),
            "mean_gap_dice_percent": float(mean_gap_s * 100),
            "full_A_dice_percent": float(d_full_s[0] * 100),
            "full_B_dice_percent": float(d_full_s[1] * 100),
            "mean_full_AB_dice_percent": float(mean_full_s * 100),
        },
        "teacher_mean_gap_dice_percent": float(mean_gap_t * 100),
        "student_mean_gap_dice_percent": float(mean_gap_s * 100),
        "teacher_mean_full_AB_dice_percent": float(mean_full_t * 100),
        "student_mean_full_AB_dice_percent": float(mean_full_s * 100),
        "best_model_by_full_AB_dice": best,
        "output_note": (
            "predicted_original_canvas and predicted_original_crop are cut from original overlap images; "
            "predicted_masks are binary masks for supplementary checking."
        ),
    }

    with open(out_dir / "metrics_full_pipeline_real_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics_full_pipeline_real_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])

        def flatten(prefix: str, obj):
            for k, v in obj.items():
                if isinstance(v, dict):
                    yield from flatten(prefix + k + ".", v)
                else:
                    yield prefix + k, v

        for k, v in flatten("", metrics):
            w.writerow([k, v])

    plot_confusion_from_matrix(conf, out_dir / "order_confusion_from_6v1_heatmap.png")
    make_metric_bar(metrics, out_dir / "teacher_student_full_pipeline_compare.png")
    make_showcase(showcase_entries, best, out_dir / "full_pipeline_showcase_original_regions.png")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-showcase", type=int, default=60)
    ap.add_argument("--force-recompute-cache", action="store_true", help="Ignore cached visible/teacher predictions and recompute.")
    ap.add_argument("--save-candidates", action="store_true", help="Also save explicit A_ON_TOP/B_ON_TOP candidate masks. Disabled by default to reduce files.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)

    visible_model_path = results_dir / "visible_order" / "best_visible_order_teacher.keras"
    teacher_path = results_dir / "missing_completion" / "best_missing_teacher.keras"
    student_path = results_dir / "missing_completion" / "best_missing_student.keras"

    for p in [visible_model_path, teacher_path, student_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing trained model: {p}")

    out_dir = results_dir / "full_pipeline_real_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_stream_cache"
    cache_visible = cache_dir / "visible_order"
    cache_teacher = cache_dir / "teacher_gap"
    cache_visible.mkdir(parents=True, exist_ok=True)
    cache_teacher.mkdir(parents=True, exist_ok=True)

    split_dir, names, labels = list_real_test_names(dataset_dir)
    print("Loading data list only, streaming arrays batch-by-batch...")
    print(f"real_test samples: {len(names)} | Batch={args.batch_size}")
    print("Model loading is one-at-a-time to avoid exit=137 OOM.")

    order_info = pass1_visible(split_dir, names, visible_model_path, cache_visible, args.batch_size, args.force_recompute_cache)
    if not order_info:
        order_info = read_pred_order_csv(cache_visible / "predicted_order.csv")

    pass2_teacher(split_dir, names, teacher_path, cache_visible, order_info, cache_teacher, args.batch_size, args.force_recompute_cache)

    metrics = pass3_student_metrics_outputs(
        split_dir=split_dir,
        names=names,
        labels=labels,
        student_path=student_path,
        cache_visible=cache_visible,
        cache_teacher=cache_teacher,
        order_info=order_info,
        out_dir=out_dir,
        batch_size=args.batch_size,
        max_showcase=args.max_showcase,
        save_candidates=args.save_candidates,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved full pipeline results to: {out_dir}")
    print("Main image-like outputs:")
    print(f"  {out_dir / 'predicted_results' / 'predicted_original_canvas'}")
    print(f"  {out_dir / 'predicted_results' / 'predicted_original_crop'}")
    print("Supplementary binary masks:")
    print(f"  {out_dir / 'predicted_results' / 'predicted_masks'}")


if __name__ == "__main__":
    main()
