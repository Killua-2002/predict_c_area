"""
10v2_evaluate_full_missing_pipeline.py
Run the complete missing-part prediction pipeline on the 1,000-image real_test split.

Pipeline:
1. Load 6v1 visible/order model.
2. Predict visible_A, visible_B, C, and top order for real_test images.
3. Feed those 6v1 predictions into the missing-completion Teacher and Student.
4. Compare Teacher vs Student on gap_A/gap_B and reconstructed full A/B.
5. Save statistics, heatmaps, predicted masks, and a 60-case showcase.

Expected trained files:
    results/visible_order/best_visible_order_teacher.keras
    results/missing_completion/best_missing_teacher.keras
    results/missing_completion/best_missing_student.keras
"""
from __future__ import annotations

import argparse, csv, json, os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

IMG_SIZE = 256
EPS = 1e-7


def read_labels(split_dir: Path):
    label_csv = split_dir / "order_labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing {label_csv}")
    labels = {}
    with open(label_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = int(row["top_class"])
    return labels


def np_gray(path: Path):
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return arr[..., None]


def np_mask(path: Path):
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    return (arr > 127).astype(np.float32)


def load_real_test(dataset_dir: Path):
    split_dir = dataset_dir / "real_test"
    labels = read_labels(split_dir)
    image_paths = sorted((split_dir / "images").glob("*.png"))
    names = [p.name for p in image_paths if p.name in labels]
    if not names:
        raise FileNotFoundError(f"No real_test samples in {split_dir}")

    X = []
    y_visible = []
    y_full = []
    y_gap = []
    y_order = []
    for n in names:
        X.append(np_gray(split_dir / "images" / n))
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
        np.stack(X).astype(np.float32),
        np.stack(y_visible).astype(np.float32),
        np.stack(y_full).astype(np.float32),
        np.stack(y_gap).astype(np.float32),
        np.array(y_order, dtype=np.int32),
        names,
    )



def unpack_visible_order_prediction(pred):
    """Return (seg_prob, order_prob) for Keras dict/list outputs."""
    if isinstance(pred, dict):
        return pred["seg"], pred["order"]
    if isinstance(pred, (list, tuple)):
        seg = None; order = None
        for item in pred:
            arr = np.asarray(item)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                seg = item
            elif arr.ndim == 2 and arr.shape[-1] == 2:
                order = item
        if seg is not None and order is not None:
            return seg, order
    raise ValueError(f"Cannot unpack visible/order prediction outputs: {type(pred)}")


def make_c_rule_candidates(pred_visible, pred_gap):
    """
    Build two explicit A/B reconstruction hypotheses from predicted C.
    candidate_A_ON_TOP: A owns C, B receives missing C/gap.
    candidate_B_ON_TOP: B owns C, A receives missing C/gap.
    """
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


def dice_per_channel(y_true, y_pred):
    inter = np.sum(y_true * y_pred, axis=(0, 1, 2))
    den = np.sum(y_true + y_pred, axis=(0, 1, 2))
    return (2 * inter + EPS) / (den + EPS)


def dice_per_image_channel(y_true, y_pred):
    inter = np.sum(y_true * y_pred, axis=(1, 2))
    den = np.sum(y_true + y_pred, axis=(1, 2))
    return (2 * inter + EPS) / (den + EPS)


def pixel_acc(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def build_missing_inputs(X_gray, pred_visible, pred_order):
    top_a = (pred_order == 0).astype(np.float32)[:, None, None, None]
    top_b = (pred_order == 1).astype(np.float32)[:, None, None, None]
    top_a_map = np.ones_like(X_gray) * top_a
    top_b_map = np.ones_like(X_gray) * top_b
    X_teacher = np.concatenate([X_gray, pred_visible, top_a_map, top_b_map], axis=-1).astype(np.float32)
    X_student = np.concatenate([X_gray, pred_visible], axis=-1).astype(np.float32)
    return X_teacher, X_student


def reconstruct_full(pred_visible, pred_gap, pred_order):
    """
    pred_visible: [N,H,W,3] = visible_A, visible_B, C
    pred_gap    : [N,H,W,2] = gap_A, gap_B
    pred_order  : 0 A_ON_TOP, 1 B_ON_TOP
    Returns predicted full A/B masks using both C decision and gap completion.
    """
    va = pred_visible[..., 0]
    vb = pred_visible[..., 1]
    c = pred_visible[..., 2]
    ga = pred_gap[..., 0]
    gb = pred_gap[..., 1]

    rec_a = np.maximum(va, ga)
    rec_b = np.maximum(vb, gb)

    # Enforce the C-to-top rule from the order classifier.
    for i, order in enumerate(pred_order):
        if int(order) == 0:  # A_ON_TOP => A owns C visually, B has missing C
            rec_a[i] = np.maximum(rec_a[i], c[i])
            rec_b[i] = np.maximum(rec_b[i], gb[i])
        else:                # B_ON_TOP => B owns C visually, A has missing C
            rec_b[i] = np.maximum(rec_b[i], c[i])
            rec_a[i] = np.maximum(rec_a[i], ga[i])

    return np.stack([rec_a, rec_b], axis=-1).astype(np.float32)


def save_mask(arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr > 0.5).astype(np.uint8) * 255).save(path)


def plot_confusion(y_true, y_pred, out_path: Path):
    conf = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    conf_norm = conf.astype(np.float64) / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(5, 4))
    plt.imshow(conf_norm * 100)
    plt.xticks([0, 1], ["A_TOP", "B_TOP"])
    plt.yticks([0, 1], ["A_TOP", "B_TOP"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("6v1 Order prediction confusion (%)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{conf_norm[i, j] * 100:.1f}%", ha="center", va="center")
    plt.colorbar(label="Percent")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return conf


def make_metric_bar(metrics, out_path: Path):
    labels = ["Gap mean", "Full A/B mean"]
    teacher = [metrics["teacher_mean_gap_dice_percent"], metrics["teacher_mean_full_AB_dice_percent"]]
    student = [metrics["student_mean_gap_dice_percent"], metrics["student_mean_full_AB_dice_percent"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, teacher, width, label="Teacher")
    plt.bar(x + width/2, student, width, label="Student")
    plt.xticks(x, labels)
    plt.ylabel("Dice (%)")
    plt.title("Full pipeline: Teacher vs Student on real_test")
    plt.legend()
    for i, v in enumerate(teacher):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(student):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_showcase(X, y_full, y_gap, pred_visible, pred_order, p_gap_t, p_gap_s, rec_t, rec_s, best_name, names, out_path: Path, max_show=60):
    n = min(max_show, len(names))
    cols = 8
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.75, max(10, n * 1.10)))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    best_rec = rec_t if best_name == "teacher" else rec_s
    best_gap = p_gap_t if best_name == "teacher" else p_gap_s

    titles = ["Overlap", "Pred A vis", "Pred B vis", "Pred C/order", "GT missing", "Teacher missing", "Student missing", "Best full A|B"]
    for i in range(n):
        gray = X[i, ..., 0]
        pred_c = pred_visible[i, ..., 2]
        gt_gap_union = np.maximum(y_gap[i, ..., 0], y_gap[i, ..., 1])
        t_gap_union = np.maximum(p_gap_t[i, ..., 0], p_gap_t[i, ..., 1])
        s_gap_union = np.maximum(p_gap_s[i, ..., 0], p_gap_s[i, ..., 1])
        full_ab = np.concatenate([best_rec[i, ..., 0], best_rec[i, ..., 1]], axis=1)
        order_txt = "A_TOP" if pred_order[i] == 0 else "B_TOP"

        imgs = [gray, pred_visible[i, ..., 0], pred_visible[i, ..., 1], pred_c, gt_gap_union, t_gap_union, s_gap_union, full_ab]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=7)
        axes[i, 3].text(2, 12, order_txt, color="yellow", fontsize=6, bbox=dict(facecolor="black", alpha=0.4, pad=1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-showcase", type=int, default=60)
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

    print("Loading data...")
    X, y_visible, y_full, y_gap, y_order, names = load_real_test(dataset_dir)
    print(f"real_test samples: {len(names)} | Batch={args.batch_size}")

    print("Loading models from exact paths only, no recursive model search...")
    visible_model = keras.models.load_model(str(visible_model_path), compile=False, safe_mode=False)
    teacher = keras.models.load_model(str(teacher_path), compile=False, safe_mode=False)
    student = keras.models.load_model(str(student_path), compile=False, safe_mode=False)

    print("Step 1/3: Predict visible A/B/C and order with 6v1...")
    vo_pred = visible_model.predict(X, batch_size=args.batch_size, verbose=0)
    pred_visible_prob, pred_order_prob = unpack_visible_order_prediction(vo_pred)
    pred_visible = (pred_visible_prob >= 0.5).astype(np.float32)
    pred_order = np.argmax(pred_order_prob, axis=1).astype(np.int32)

    X_teacher, X_student = build_missing_inputs(X, pred_visible, pred_order)

    print("Step 2/3: Predict missing gaps with Teacher and Student from 6v1 outputs...")
    pred_gap_teacher = (teacher.predict(X_teacher, batch_size=args.batch_size, verbose=0) >= 0.5).astype(np.float32)
    pred_gap_student = (student.predict(X_student, batch_size=args.batch_size, verbose=0) >= 0.5).astype(np.float32)

    # Explicit C-rule hypotheses for report/debugging.
    # These represent the two possible interpretations of C before the order classifier chooses one.
    cand_teacher_A_top, cand_teacher_B_top = make_c_rule_candidates(pred_visible, pred_gap_teacher)
    cand_student_A_top, cand_student_B_top = make_c_rule_candidates(pred_visible, pred_gap_student)

    rec_teacher = reconstruct_full(pred_visible, pred_gap_teacher, pred_order)
    rec_student = reconstruct_full(pred_visible, pred_gap_student, pred_order)

    print("Step 3/3: Metrics and visualization...")
    d_vis = dice_per_channel(y_visible, pred_visible)
    d_gap_t = dice_per_channel(y_gap, pred_gap_teacher)
    d_gap_s = dice_per_channel(y_gap, pred_gap_student)
    d_full_t = dice_per_channel(y_full, rec_teacher)
    d_full_s = dice_per_channel(y_full, rec_student)
    order_acc = float(np.mean(pred_order == y_order))

    mean_gap_t = float(np.mean(d_gap_t))
    mean_gap_s = float(np.mean(d_gap_s))
    mean_full_t = float(np.mean(d_full_t))
    mean_full_s = float(np.mean(d_full_s))
    best = "teacher" if mean_full_t >= mean_full_s else "student"

    metrics = {
        "real_test_count": int(len(names)),
        "visible_order_model": {
            "visible_A_dice_percent": float(d_vis[0] * 100),
            "visible_B_dice_percent": float(d_vis[1] * 100),
            "C_overlap_dice_percent": float(d_vis[2] * 100),
            "mean_visible_ABC_dice_percent": float(np.mean(d_vis) * 100),
            "order_accuracy_percent": float(order_acc * 100),
            "visible_pixel_accuracy_percent": pixel_acc(y_visible, pred_visible) * 100,
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
    }

    with open(out_dir / "metrics_full_pipeline_real_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics_full_pipeline_real_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        def flatten(prefix, obj):
            for k, v in obj.items():
                if isinstance(v, dict):
                    yield from flatten(prefix + k + ".", v)
                else:
                    yield prefix + k, v
        for k, v in flatten("", metrics):
            w.writerow([k, v])

    conf = plot_confusion(y_order, pred_order, out_dir / "order_confusion_from_6v1_heatmap.png")
    metrics["visible_order_model"]["order_confusion_matrix"] = conf.tolist()
    make_metric_bar(metrics, out_dir / "teacher_student_full_pipeline_compare.png")
    make_showcase(X, y_full, y_gap, pred_visible, pred_order, pred_gap_teacher, pred_gap_student, rec_teacher, rec_student, best, names, out_dir / "full_pipeline_showcase_60.png", args.max_showcase)

    # Save predicted masks for all real_test samples.
    pred_root = out_dir / "predicted_masks"
    subdirs = [
        "visible_A", "visible_B", "C_overlap", "order_text",
        "teacher_gap_A", "teacher_gap_B", "teacher_full_A", "teacher_full_B",
        "student_gap_A", "student_gap_B", "student_full_A", "student_full_B",
        "candidate_teacher_A_ON_TOP_full_A", "candidate_teacher_A_ON_TOP_full_B",
        "candidate_teacher_B_ON_TOP_full_A", "candidate_teacher_B_ON_TOP_full_B",
        "candidate_student_A_ON_TOP_full_A", "candidate_student_A_ON_TOP_full_B",
        "candidate_student_B_ON_TOP_full_A", "candidate_student_B_ON_TOP_full_B",
    ]
    for s in subdirs:
        (pred_root / s).mkdir(parents=True, exist_ok=True)

    order_rows = []
    for i, n in enumerate(names):
        stem = Path(n).stem + ".png"
        save_mask(pred_visible[i, ..., 0], pred_root / "visible_A" / stem)
        save_mask(pred_visible[i, ..., 1], pred_root / "visible_B" / stem)
        save_mask(pred_visible[i, ..., 2], pred_root / "C_overlap" / stem)
        save_mask(pred_gap_teacher[i, ..., 0], pred_root / "teacher_gap_A" / stem)
        save_mask(pred_gap_teacher[i, ..., 1], pred_root / "teacher_gap_B" / stem)
        save_mask(rec_teacher[i, ..., 0], pred_root / "teacher_full_A" / stem)
        save_mask(rec_teacher[i, ..., 1], pred_root / "teacher_full_B" / stem)
        save_mask(pred_gap_student[i, ..., 0], pred_root / "student_gap_A" / stem)
        save_mask(pred_gap_student[i, ..., 1], pred_root / "student_gap_B" / stem)
        save_mask(rec_student[i, ..., 0], pred_root / "student_full_A" / stem)
        save_mask(rec_student[i, ..., 1], pred_root / "student_full_B" / stem)

        # Save explicit C-to-A/B hypotheses: A_ON_TOP and B_ON_TOP.
        save_mask(cand_teacher_A_top[i, ..., 0], pred_root / "candidate_teacher_A_ON_TOP_full_A" / stem)
        save_mask(cand_teacher_A_top[i, ..., 1], pred_root / "candidate_teacher_A_ON_TOP_full_B" / stem)
        save_mask(cand_teacher_B_top[i, ..., 0], pred_root / "candidate_teacher_B_ON_TOP_full_A" / stem)
        save_mask(cand_teacher_B_top[i, ..., 1], pred_root / "candidate_teacher_B_ON_TOP_full_B" / stem)
        save_mask(cand_student_A_top[i, ..., 0], pred_root / "candidate_student_A_ON_TOP_full_A" / stem)
        save_mask(cand_student_A_top[i, ..., 1], pred_root / "candidate_student_A_ON_TOP_full_B" / stem)
        save_mask(cand_student_B_top[i, ..., 0], pred_root / "candidate_student_B_ON_TOP_full_A" / stem)
        save_mask(cand_student_B_top[i, ..., 1], pred_root / "candidate_student_B_ON_TOP_full_B" / stem)

        order_rows.append({
            "filename": n,
            "gt_order": "A_ON_TOP" if int(y_order[i]) == 0 else "B_ON_TOP",
            "pred_order": "A_ON_TOP" if int(pred_order[i]) == 0 else "B_ON_TOP",
            "prob_A_ON_TOP": float(pred_order_prob[i, 0]),
            "prob_B_ON_TOP": float(pred_order_prob[i, 1]),
        })

    with open(pred_root / "predicted_order.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
        writer.writeheader(); writer.writerows(order_rows)

    # Update metrics JSON after adding confusion.
    with open(out_dir / "metrics_full_pipeline_real_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved full pipeline results to: {out_dir}")


if __name__ == "__main__":
    main()
