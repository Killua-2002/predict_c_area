"""
9v2_train_missing_completion_teacher_student.py
Model 2: missing-part/complement prediction.

Purpose:
- After 6v1 predicts visible A/B/C and top order, this stage learns the hidden part
  of the chromosome that is under the overlap.
- Teacher receives order maps as extra channels, so it is the stronger branch.
- Student receives no explicit order maps, so it must infer the missing branch from image + visible masks.
- Both predict 2 channels: gap_A and gap_B.
  gap_A = C only when A is under B, else empty.
  gap_B = C only when B is under A, else empty.

Dataset expected:
    dataset/train|val|test|real_test/images
    dataset/.../visible_A, visible_B, masks_C, gap_A, gap_B, order_labels.csv

Outputs:
    results/missing_completion/
        best_missing_teacher.keras
        best_missing_student.keras
        metrics_real_test_compare.json
        missing_compare_showcase_60.png
        history csv/json/curves
"""
from __future__ import annotations

import argparse, csv, json, os, random, math, time, re
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
from tensorflow.keras import layers

IMG_SIZE = 256
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def read_labels(split_dir: Path) -> Dict[str, int]:
    csv_path = split_dir / "order_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): labels[row["filename"]] = int(row["top_class"])
    return labels


def get_split_lists(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    labels = read_labels(split_dir)
    rows = []
    for p in sorted((split_dir / "images").glob("*.png")):
        n = p.name
        needed = ["visible_A", "visible_B", "masks_C", "gap_A", "gap_B"]
        if n in labels and all((split_dir / sub / n).exists() for sub in needed):
            rows.append((str(p), str(split_dir/"visible_A"/n), str(split_dir/"visible_B"/n), str(split_dir/"masks_C"/n), str(split_dir/"gap_A"/n), str(split_dir/"gap_B"/n), labels[n], n))
    if not rows: raise FileNotFoundError(f"No valid samples in {split_dir}")
    return rows


def read_gray(path):
    img = tf.io.read_file(path); img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 1]); return img


def read_mask(path):
    m = tf.io.read_file(path); m = tf.image.decode_png(m, channels=1)
    m = tf.image.resize(m, (IMG_SIZE, IMG_SIZE), method="nearest")
    m = tf.cast(m > 127, tf.float32); m.set_shape([IMG_SIZE, IMG_SIZE, 1]); return m


def load_paths(img_p, va_p, vb_p, c_p, ga_p, gb_p):
    gray, va, vb, c = read_gray(img_p), read_mask(va_p), read_mask(vb_p), read_mask(c_p)
    ga, gb = read_mask(ga_p), read_mask(gb_p)
    return gray, va, vb, c, ga, gb


def augment(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x); y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x); y = tf.image.flip_up_down(y)
    return x, y


def make_ds(dataset_dir: Path, split: str, role: str, batch: int, shuffle=False, do_aug=False):
    rows = get_split_lists(dataset_dir, split)
    if shuffle:
        import random
        random.shuffle(rows)
    cols = list(zip(*rows))
    
    paths_ds = tf.data.Dataset.from_tensor_slices(tuple(cols[:6]))
    paths_ds = paths_ds.map(load_paths, num_parallel_calls=tf.data.AUTOTUNE)
    meta_ds = tf.data.Dataset.from_tensor_slices((cols[6], cols[7]))
    
    ds = tf.data.Dataset.zip((paths_ds, meta_ds))

    if role == "teacher":
        def assemble_teacher(paths_out, meta_out):
            gray, va, vb, c, ga, gb = paths_out
            order, name = meta_out
            top_a = tf.ones_like(gray) * tf.cast(tf.equal(order, 0), tf.float32)
            top_b = tf.ones_like(gray) * tf.cast(tf.equal(order, 1), tf.float32)
            x = tf.concat([gray, va, vb, c, top_a, top_b], axis=-1)
            y = tf.concat([ga, gb], axis=-1)
            x.set_shape([IMG_SIZE, IMG_SIZE, 6]); y.set_shape([IMG_SIZE, IMG_SIZE, 2])
            return x, y
        ds = ds.map(assemble_teacher, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def assemble_student(paths_out, meta_out):
            gray, va, vb, c, ga, gb = paths_out
            x = tf.concat([gray, va, vb, c], axis=-1)
            y = tf.concat([ga, gb], axis=-1)
            x.set_shape([IMG_SIZE, IMG_SIZE, 4]); y.set_shape([IMG_SIZE, IMG_SIZE, 2])
            return x, y
        ds = ds.map(assemble_student, num_parallel_calls=tf.data.AUTOTUNE)

    if do_aug: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(1), len(rows)


def conv(x, f, drop=0):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    if drop: x = layers.SpatialDropout2D(drop)(x)
    return x


def build_unet(input_channels: int, base=32, name="missing_completion_unet"):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, input_channels))
    s1 = conv(inp, base); p1 = layers.MaxPooling2D()(s1)
    s2 = conv(p1, base*2); p2 = layers.MaxPooling2D()(s2)
    s3 = conv(p2, base*4, .05); p3 = layers.MaxPooling2D()(s3)
    s4 = conv(p3, base*8, .10); p4 = layers.MaxPooling2D()(s4)
    b = conv(p4, base*16, .15)
    x = layers.Conv2DTranspose(base*8, 2, 2, padding="same")(b); x = layers.Concatenate()([x,s4]); x = conv(x, base*8, .10)
    x = layers.Conv2DTranspose(base*4, 2, 2, padding="same")(x); x = layers.Concatenate()([x,s3]); x = conv(x, base*4, .05)
    x = layers.Conv2DTranspose(base*2, 2, 2, padding="same")(x); x = layers.Concatenate()([x,s2]); x = conv(x, base*2)
    x = layers.Conv2DTranspose(base, 2, 2, padding="same")(x); x = layers.Concatenate()([x,s1]); x = conv(x, base)
    out = layers.Conv2D(2, 1, activation="sigmoid", dtype="float32", name="gap_A_gap_B")(x)
    return keras.Model(inp, out, name=name)


def dice_loss(y_true, y_pred):
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    den = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    dice = (2*inter + 1e-6) / (den + 1e-6)
    return 1.0 - tf.reduce_mean(dice)


def gap_loss(y_true, y_pred):
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def gap_dice(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    den = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    return tf.reduce_mean((2*inter + 1e-6)/(den + 1e-6))


def save_history(hist, out_dir: Path, name: str):
    data = {k:[float(x) for x in v] for k,v in hist.history.items()}
    with open(out_dir / f"{name}_history.json", "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    keys = list(data.keys())
    with open(out_dir / f"{name}_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch"] + keys)
        for i in range(len(next(iter(data.values())))): w.writerow([i+1] + [data[k][i] for k in keys])
    for key, title in [("loss", "Loss"), ("gap_dice", "Gap Dice")]:
        plt.figure(figsize=(7,4))
        if key in data: plt.plot(data[key], label="train")
        if "val_"+key in data: plt.plot(data["val_"+key], label="val")
        plt.title(f"{name} {title}"); plt.xlabel("Epoch"); plt.ylabel(title); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{key}_curve.png", dpi=160); plt.close()


def np_gray(path):
    return (np.array(Image.open(path).convert("L").resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255.0)[...,None]

def np_mask(path):
    return (np.array(Image.open(path).convert("L").resize((IMG_SIZE,IMG_SIZE))).astype(np.float32) > 127).astype(np.float32)


def make_np_inputs(dataset_dir: Path, split="real_test"):
    rows = get_split_lists(dataset_dir, split)
    X_teacher, X_student, Y, names = [], [], [], []
    for img_p, va_p, vb_p, c_p, ga_p, gb_p, order, name in rows:
        gray, va, vb, c = np_gray(img_p), np_mask(va_p)[...,None], np_mask(vb_p)[...,None], np_mask(c_p)[...,None]
        top_a = np.ones_like(gray) * (1.0 if int(order) == 0 else 0.0)
        top_b = np.ones_like(gray) * (1.0 if int(order) == 1 else 0.0)
        X_teacher.append(np.concatenate([gray, va, vb, c, top_a, top_b], axis=-1))
        X_student.append(np.concatenate([gray, va, vb, c], axis=-1))
        Y.append(np.stack([np_mask(ga_p), np_mask(gb_p)], axis=-1))
        names.append(name)
    return np.stack(X_teacher).astype(np.float32), np.stack(X_student).astype(np.float32), np.stack(Y).astype(np.float32), names


def eval_compare(teacher, student, dataset_dir: Path, out_dir: Path, batch_size=16, max_show=60):
    Xt, Xs, Y, names = make_np_inputs(dataset_dir, "real_test")
    print(f"[9v2][eval] Predicting real_test: {len(names)} images ...")
    Pt = (teacher.predict(Xt, batch_size=batch_size, verbose=0) >= 0.5).astype(np.float32)
    Ps = (student.predict(Xs, batch_size=batch_size, verbose=0) >= 0.5).astype(np.float32)

    def dice_np(y, p):
        inter = np.sum(y*p, axis=(1,2))
        den = np.sum(y+p, axis=(1,2))
        d = (2*inter + 1e-7) / (den + 1e-7)
        return d
    dt = dice_np(Y, Pt); ds = dice_np(Y, Ps)
    mean_t_ch = dt.mean(axis=0); mean_s_ch = ds.mean(axis=0)
    mean_t, mean_s = float(dt.mean()), float(ds.mean())
    best = "teacher" if mean_t >= mean_s else "student"
    metrics = {
        "real_test_count": len(names),
        "teacher_mean_gap_dice_percent": mean_t*100,
        "student_mean_gap_dice_percent": mean_s*100,
        "teacher_gap_A_dice_percent": float(mean_t_ch[0]*100),
        "teacher_gap_B_dice_percent": float(mean_t_ch[1]*100),
        "student_gap_A_dice_percent": float(mean_s_ch[0]*100),
        "student_gap_B_dice_percent": float(mean_s_ch[1]*100),
        "best_model_on_real_test": best,
    }
    with open(out_dir / "metrics_real_test_compare.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)

    # bar chart
    plt.figure(figsize=(6,4))
    x = np.arange(2); width = 0.35
    plt.bar(x-width/2, mean_t_ch*100, width, label="Teacher")
    plt.bar(x+width/2, mean_s_ch*100, width, label="Student")
    plt.xticks(x, ["gap_A", "gap_B"]); plt.ylabel("Dice %"); plt.title("Real test missing-gap Dice")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "teacher_student_gap_dice_compare.png", dpi=180); plt.close()

    # showcase 60: overlap | GT gapA/gapB | teacher gap | student gap
    show = min(max_show, len(names)); cols=5
    fig, axes = plt.subplots(show, cols, figsize=(cols*2.2, max(9, show*1.25)))
    if show == 1: axes = np.expand_dims(axes,0)
    for i in range(show):
        gray = Xs[i,...,0]
        gt_union = np.maximum(Y[i,...,0], Y[i,...,1])
        t_union = np.maximum(Pt[i,...,0], Pt[i,...,1])
        s_union = np.maximum(Ps[i,...,0], Ps[i,...,1])
        err = np.abs(gt_union - (t_union if best == "teacher" else s_union))
        imgs = [gray, gt_union, t_union, s_union, err]
        titles = ["Overlap", "GT missing", "Teacher", "Student", "Best err"]
        for j in range(cols):
            axes[i,j].imshow(imgs[j], cmap="gray"); axes[i,j].axis("off")
            if i == 0: axes[i,j].set_title(titles[j], fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "missing_compare_showcase_60.png", dpi=180); plt.close()
    return metrics


def checkpoint_epoch(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.weights\.h5$", path.name)
    return int(m.group(1)) if m else -1


def latest_weight_checkpoint(ckpt_dir: Path):
    files = sorted(ckpt_dir.glob("epoch_*.weights.h5"), key=checkpoint_epoch)
    files = [p for p in files if checkpoint_epoch(p) > 0]
    if not files:
        return 0, None
    p = files[-1]
    return checkpoint_epoch(p), p


class KeepLastCheckpoints(keras.callbacks.Callback):
    def __init__(self, ckpt_dir: Path, keep: int = 3):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.keep = int(max(1, keep))

    def on_epoch_end(self, epoch, logs=None):
        files = sorted(self.ckpt_dir.glob("epoch_*.weights.h5"), key=checkpoint_epoch)
        for p in files[:-self.keep]:
            try:
                p.unlink()
            except Exception:
                pass


class CleanProgressCallback(keras.callbacks.Callback):
    def __init__(self, role: str, total_epochs: int, steps_per_epoch: int, mode: str = "line"):
        super().__init__()
        self.role = role
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = int(max(1, steps_per_epoch))
        self.mode = mode
        self.pbar = None
        self.t0 = None

    @staticmethod
    def _get(logs, key, default=None):
        if not logs:
            return default
        try:
            return float(logs.get(key, default))
        except Exception:
            return default

    @staticmethod
    def _fmt(v, percent=False):
        if v is None:
            return "-"
        return f"{v*100:.2f}%" if percent else f"{v:.4f}"

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        desc = f"[9v2][{self.role}] Epoch {epoch+1:03d}/{self.total_epochs:03d}"
        if self.mode == "tqdm" and tqdm is not None:
            self.pbar = tqdm(total=self.steps_per_epoch, desc=desc, unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5)
        else:
            print(desc)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"loss": self._fmt(self._get(logs, "loss")), "dice": self._fmt(self._get(logs, "gap_dice"), percent=True)})
        elif self.mode == "line":
            step = batch + 1
            if step == 1 or step == self.steps_per_epoch or step % max(1, self.steps_per_epoch // 5) == 0:
                pct = step / self.steps_per_epoch * 100
                print(f"  step {step:03d}/{self.steps_per_epoch:03d} ({pct:5.1f}%) | loss={self._fmt(self._get(logs,'loss'))} | dice={self._fmt(self._get(logs,'gap_dice'), percent=True)}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.pbar is not None:
            self.pbar.close(); self.pbar = None
        sec = time.time() - self.t0 if self.t0 else 0.0
        print(
            f"[9v2][{self.role}][{epoch+1:03d}/{self.total_epochs:03d}] {sec:6.1f}s | "
            f"loss={self._fmt(self._get(logs,'loss'))} | val_loss={self._fmt(self._get(logs,'val_loss'))} | "
            f"dice={self._fmt(self._get(logs,'gap_dice'), percent=True)} | val_dice={self._fmt(self._get(logs,'val_gap_dice'), percent=True)}",
            flush=True,
        )


def train_one(role, dataset_dir, out_dir, epochs, batch, base, lr, patience, resume=False, force_restart=False, progress_mode="line", max_keep_checkpoints=3):
    channels = 6 if role == "teacher" else 4
    model = build_unet(channels, base=base, name=f"missing_{role}")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=gap_loss, metrics=[gap_dice])
    train_ds, nt = make_ds(dataset_dir, "train", role, batch, True, True)
    val_ds, nv = make_ds(dataset_dir, "val", role, batch)
    steps_per_epoch = math.ceil(nt / batch)
    val_steps = math.ceil(nv / batch)
    print("="*80)
    print(f"9v2 MISSING COMPLETION {role.upper()} TRAINING")
    print("="*80)
    print(f"Dataset: train={nt}, val={nv}")
    print(f"Batch={batch} | steps/epoch={steps_per_epoch} | val_steps={val_steps}")
    print(f"Epochs={epochs} | patience={patience} | base_filters={base} | lr={lr}")

    ckpt = out_dir / f"best_missing_{role}.keras"
    final_path = out_dir / f"final_missing_{role}.keras"
    ckpt_dir = out_dir / f"checkpoints_{role}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    if force_restart:
        print(f"[9v2][{role}] force restart: ignore old checkpoints")
    elif resume:
        last_epoch, last_ckpt = latest_weight_checkpoint(ckpt_dir)
        if last_ckpt is not None:
            model.load_weights(str(last_ckpt))
            initial_epoch = last_epoch
            print(f"[9v2][{role}] Resumed from {last_ckpt} -> initial_epoch={initial_epoch}")
        elif final_path.exists():
            loaded = keras.models.load_model(str(final_path), compile=False, safe_mode=False)
            model.set_weights(loaded.get_weights())
            print(f"[9v2][{role}] Loaded final model weights: {final_path}")
        elif ckpt.exists():
            loaded = keras.models.load_model(str(ckpt), compile=False, safe_mode=False)
            model.set_weights(loaded.get_weights())
            print(f"[9v2][{role}] Loaded best model weights: {ckpt}")
        else:
            print(f"[9v2][{role}] No checkpoint found, train from scratch")

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_gap_dice", mode="max", save_best_only=True, verbose=0),
        keras.callbacks.ModelCheckpoint(str(ckpt_dir / "epoch_{epoch:03d}.weights.h5"), save_weights_only=True, save_freq="epoch", verbose=0),
        KeepLastCheckpoints(ckpt_dir, keep=max_keep_checkpoints),
        keras.callbacks.EarlyStopping(monitor="val_gap_dice", mode="max", patience=patience, restore_best_weights=True, verbose=0),
        keras.callbacks.CSVLogger(str(out_dir / f"{role}_epoch_log.csv"), append=bool(resume and initial_epoch > 0)),
    ]
    fit_verbose = 1 if progress_mode == "keras" else 0
    if progress_mode != "keras":
        callbacks.insert(0, CleanProgressCallback(role, epochs, steps_per_epoch, progress_mode))

    if initial_epoch >= epochs:
        print(f"[9v2][{role}] Already reached epoch {initial_epoch}/{epochs}; skip training.")
        hist = None
    else:
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=fit_verbose,
        )
        if hist is not None and hist.history:
            save_history(hist, out_dir, role)
    model.save(str(final_path))
    if not ckpt.exists():
        model.save(str(ckpt))
    print(f"[9v2][{role}] Saved best : {ckpt}")
    print(f"[9v2][{role}] Saved final: {final_path}")
    return keras.models.load_model(str(ckpt), compile=False, safe_mode=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--base-filters", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--skip-student", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--progress-mode", choices=["tqdm", "line", "keras"], default="line")
    ap.add_argument("--max-keep-checkpoints", type=int, default=3)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.results_dir) / "missing_completion"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_teacher:
        teacher = train_one("teacher", dataset_dir, out_dir, args.epochs, args.batch_size, args.base_filters, args.lr, args.patience, args.resume, args.force_restart, args.progress_mode, args.max_keep_checkpoints)
    else:
        teacher = keras.models.load_model(str(out_dir / "best_missing_teacher.keras"), compile=False, safe_mode=False)

    if not args.skip_student:
        student = train_one("student", dataset_dir, out_dir, args.epochs, args.batch_size, max(16, args.base_filters//2), args.lr, args.patience, args.resume, args.force_restart, args.progress_mode, args.max_keep_checkpoints)
    else:
        student = keras.models.load_model(str(out_dir / "best_missing_student.keras"), compile=False, safe_mode=False)

    metrics = eval_compare(teacher, student, dataset_dir, out_dir, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
