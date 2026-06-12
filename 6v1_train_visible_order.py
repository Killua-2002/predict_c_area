"""
6v1_train_visible_order.py
Model 1: predict visible A/B/C masks and classify which chromosome is on top.

Dataset expected:
    dataset/train|val|test|real_test/
        images/
        visible_A/
        visible_B/
        masks_C/
        order_labels.csv   with top_class: 0=A_ON_TOP, 1=B_ON_TOP

Output:
    results/visible_order/
        best_visible_order_teacher.keras
        history_visible_order.csv/json
        metrics_visible_order_real_test.json
        confusion_order_heatmap.png
        visible_order_showcase.png
        checkpoints/epoch_XXX.weights.h5 for resume
"""
from __future__ import annotations

import argparse, csv, json, os, random, shutil, math, time, re
from pathlib import Path
from typing import Dict, List, Tuple

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
        for row in csv.DictReader(f):
            labels[row["filename"]] = int(row["top_class"])
    return labels


def get_split_lists(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    labels = read_labels(split_dir)
    image_paths, va_paths, vb_paths, c_paths, y_order = [], [], [], [], []
    for p in sorted((split_dir / "images").glob("*.png")):
        name = p.name
        va = split_dir / "visible_A" / name
        vb = split_dir / "visible_B" / name
        mc = split_dir / "masks_C" / name
        if name in labels and va.exists() and vb.exists() and mc.exists():
            image_paths.append(str(p)); va_paths.append(str(va)); vb_paths.append(str(vb)); c_paths.append(str(mc)); y_order.append(labels[name])
    if not image_paths:
        raise FileNotFoundError(f"No valid samples in {split_dir}")
    return image_paths, va_paths, vb_paths, c_paths, np.array(y_order, dtype=np.int32)


def read_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img


def read_mask(path):
    m = tf.io.read_file(path)
    m = tf.image.decode_png(m, channels=1)
    m = tf.image.resize(m, (IMG_SIZE, IMG_SIZE), method="nearest")
    m = tf.cast(m > 127, tf.float32)
    m.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return m


def load_sample(img_p, va_p, vb_p, c_p, top_class):
    x = read_image(img_p)
    va, vb, c = read_mask(va_p), read_mask(vb_p), read_mask(c_p)
    y_seg = tf.concat([va, vb, c], axis=-1)
    y_seg.set_shape([IMG_SIZE, IMG_SIZE, 3])
    y_order = tf.one_hot(tf.cast(top_class, tf.int32), 2)
    return x, {"seg": y_seg, "order": y_order}


def augment(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y["seg"] = tf.image.flip_left_right(y["seg"])
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y["seg"] = tf.image.flip_up_down(y["seg"])
    if tf.random.uniform(()) > 0.7:
        x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), 0, 0.015), 0, 1)
    return x, y


def make_ds(dataset_dir: Path, split: str, batch: int, shuffle=False, do_aug=False):
    img, va, vb, c, order = get_split_lists(dataset_dir, split)
    ds = tf.data.Dataset.from_tensor_slices((img, va, vb, c, order))
    if shuffle:
        ds = ds.shuffle(min(len(img), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    if do_aug:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(1), len(img)


def conv_block(x, f, drop=0.0):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    if drop: x = layers.SpatialDropout2D(drop)(x)
    return x


def build_model(base=32):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 1), name="image")
    s1 = conv_block(inp, base); p1 = layers.MaxPooling2D()(s1)
    s2 = conv_block(p1, base*2); p2 = layers.MaxPooling2D()(s2)
    s3 = conv_block(p2, base*4, 0.05); p3 = layers.MaxPooling2D()(s3)
    s4 = conv_block(p3, base*8, 0.10); p4 = layers.MaxPooling2D()(s4)
    b = conv_block(p4, base*16, 0.15)

    # order classifier from bottleneck
    o = layers.GlobalAveragePooling2D()(b)
    o = layers.Dense(128, activation="relu")(o)
    o = layers.Dropout(0.25)(o)
    order = layers.Dense(2, activation="softmax", name="order")(o)

    x = layers.Conv2DTranspose(base*8, 2, strides=2, padding="same")(b); x = layers.Concatenate()([x, s4]); x = conv_block(x, base*8, 0.10)
    x = layers.Conv2DTranspose(base*4, 2, strides=2, padding="same")(x); x = layers.Concatenate()([x, s3]); x = conv_block(x, base*4, 0.05)
    x = layers.Conv2DTranspose(base*2, 2, strides=2, padding="same")(x); x = layers.Concatenate()([x, s2]); x = conv_block(x, base*2)
    x = layers.Conv2DTranspose(base, 2, strides=2, padding="same")(x); x = layers.Concatenate()([x, s1]); x = conv_block(x, base)
    seg = layers.Conv2D(3, 1, activation="sigmoid", dtype="float32", name="seg")(x)
    return keras.Model(inp, {"seg": seg, "order": order}, name="Visible_AB_C_Order_Teacher")


def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    den = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    return tf.reduce_mean((2*inter + 1e-6) / (den + 1e-6))


def dice_loss(y_true, y_pred):
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    den = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    dice = (2*inter + 1e-6) / (den + 1e-6)
    return 1.0 - tf.reduce_mean(dice)


def seg_loss(y_true, y_pred):
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def save_history(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(out_dir / "history_visible_order.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    keys = list(hist.keys())
    with open(out_dir / "history_visible_order.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch"] + keys)
        for i in range(len(next(iter(hist.values())))):
            w.writerow([i+1] + [hist[k][i] for k in keys])

    for metric_name, title in [("loss", "Total loss"), ("seg_dice_metric", "Segmentation Dice"), ("order_accuracy", "Order accuracy")]:
        plt.figure(figsize=(7,4))
        if metric_name in hist: plt.plot(hist[metric_name], label="train")
        if "val_" + metric_name in hist: plt.plot(hist["val_" + metric_name], label="val")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(title); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"curve_{metric_name}.png", dpi=160); plt.close()


def np_load_gray(path):
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return arr[..., None]

def np_load_mask(path):
    arr = np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    return (arr > 127).astype(np.float32)


def evaluate_real_test(model, dataset_dir: Path, out_dir: Path, batch_size: int = 16, max_showcase: int = 60):
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = dataset_dir / "real_test"
    labels = read_labels(split_dir)
    names = [p.name for p in sorted((split_dir / "images").glob("*.png")) if p.name in labels]
    X, Y_seg, Y_order = [], [], []
    for n in names:
        X.append(np_load_gray(split_dir / "images" / n))
        va = np_load_mask(split_dir / "visible_A" / n)
        vb = np_load_mask(split_dir / "visible_B" / n)
        c = np_load_mask(split_dir / "masks_C" / n)
        Y_seg.append(np.stack([va, vb, c], axis=-1))
        Y_order.append(labels[n])
    X = np.stack(X).astype(np.float32)
    Y_seg = np.stack(Y_seg).astype(np.float32)
    Y_order = np.array(Y_order, dtype=np.int32)

    print(f"[6v1][eval] Predicting real_test: {len(names)} images ...")
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    pred_seg_prob, pred_order_prob = unpack_visible_order_prediction(pred)
    pred_seg = (pred_seg_prob >= 0.5).astype(np.float32)
    pred_order = np.argmax(pred_order_prob, axis=1)

    dice_per_ch = []
    for c in range(3):
        inter = np.sum(Y_seg[...,c] * pred_seg[...,c])
        den = np.sum(Y_seg[...,c] + pred_seg[...,c])
        dice_per_ch.append(float((2*inter + 1e-7) / (den + 1e-7)))
    order_acc = float(np.mean(pred_order == Y_order))
    mean_dice = float(np.mean(dice_per_ch))

    conf = np.zeros((2,2), dtype=int)
    for t, p in zip(Y_order, pred_order): conf[t,p] += 1
    conf_norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)

    metrics = {
        "real_test_count": int(len(names)),
        "mean_visible_seg_dice_percent": mean_dice*100,
        "dice_visible_A_percent": dice_per_ch[0]*100,
        "dice_visible_B_percent": dice_per_ch[1]*100,
        "dice_C_percent": dice_per_ch[2]*100,
        "order_accuracy_percent": order_acc*100,
        "confusion_matrix": conf.tolist(),
    }
    with open(out_dir / "metrics_visible_order_real_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(5,4))
    plt.imshow(conf_norm*100)
    plt.xticks([0,1], ["A_TOP", "B_TOP"]); plt.yticks([0,1], ["A_TOP", "B_TOP"])
    plt.xlabel("Predicted"); plt.ylabel("Ground truth"); plt.title("Order classifier confusion (%)")
    for i in range(2):
        for j in range(2): plt.text(j, i, f"{conf_norm[i,j]*100:.1f}%", ha="center", va="center")
    plt.colorbar(label="Percent"); plt.tight_layout(); plt.savefig(out_dir / "confusion_order_heatmap.png", dpi=180); plt.close()

    # showcase: 60 samples, each row: overlap | visible A pred | visible B pred | C pred | order
    show_dir = out_dir / "showcase"; show_dir.mkdir(exist_ok=True)
    n_show = min(max_showcase, len(names))
    cols = 5; rows = n_show
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, max(8, rows*1.35)))
    if rows == 1: axes = np.expand_dims(axes, 0)
    for i in range(n_show):
        gray = X[i,...,0]
        imgs = [gray, pred_seg[i,...,0], pred_seg[i,...,1], pred_seg[i,...,2]]
        titles = ["Overlap", "Pred A_visible", "Pred B_visible", "Pred C"]
        for j in range(4):
            axes[i,j].imshow(imgs[j], cmap="gray"); axes[i,j].axis("off")
            if i == 0: axes[i,j].set_title(titles[j], fontsize=8)
        txt = "A_ON_TOP" if pred_order[i] == 0 else "B_ON_TOP"
        gt = "A_ON_TOP" if Y_order[i] == 0 else "B_ON_TOP"
        axes[i,4].axis("off"); axes[i,4].text(0.0, 0.5, f"GT: {gt}\nPred: {txt}", fontsize=7)
    plt.tight_layout(); plt.savefig(show_dir / "visible_order_showcase_60.png", dpi=180); plt.close()

    return metrics



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
        extra = files[:-self.keep]
        for p in extra:
            try:
                p.unlink()
            except Exception:
                pass


class CleanProgressCallback(keras.callbacks.Callback):
    """Clean notebook-friendly progress for 6v1.

    - Hides Keras' noisy default progress output.
    - Shows one tqdm bar per epoch in Colab/Jupyter.
    - Prints one compact metric line after each epoch.
    """
    def __init__(self, total_epochs: int, steps_per_epoch: int, mode: str = "tqdm"):
        super().__init__()
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = int(max(1, steps_per_epoch))
        self.mode = mode
        self.pbar = None
        self.t0 = None

    @staticmethod
    def _get(logs, key, default=None):
        if not logs:
            return default
        v = logs.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _fmt(v, percent=False):
        if v is None:
            return "-"
        return f"{v*100:.2f}%" if percent else f"{v:.4f}"

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        desc = f"[6v1] Epoch {epoch+1:03d}/{self.total_epochs:03d}"
        if self.mode == "tqdm" and tqdm is not None:
            self.pbar = tqdm(
                total=self.steps_per_epoch,
                desc=desc,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        else:
            print(desc)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": self._fmt(self._get(logs, "loss")),
                "dice": self._fmt(self._get(logs, "seg_dice_metric"), percent=True),
                "ord": self._fmt(self._get(logs, "order_accuracy"), percent=True),
            })
        elif self.mode == "line":
            step = batch + 1
            if step == 1 or step == self.steps_per_epoch or step % max(1, self.steps_per_epoch // 5) == 0:
                pct = step / self.steps_per_epoch * 100
                print(
                    f"  step {step:03d}/{self.steps_per_epoch:03d} ({pct:5.1f}%) | "
                    f"loss={self._fmt(self._get(logs, 'loss'))} | "
                    f"dice={self._fmt(self._get(logs, 'seg_dice_metric'), percent=True)} | "
                    f"order={self._fmt(self._get(logs, 'order_accuracy'), percent=True)}"
                )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        sec = time.time() - self.t0 if self.t0 else 0.0
        msg = (
            f"[6v1][{epoch+1:03d}/{self.total_epochs:03d}] "
            f"{sec:6.1f}s | "
            f"loss={self._fmt(self._get(logs, 'loss'))} | "
            f"val_loss={self._fmt(self._get(logs, 'val_loss'))} | "
            f"dice={self._fmt(self._get(logs, 'seg_dice_metric'), percent=True)} | "
            f"val_dice={self._fmt(self._get(logs, 'val_seg_dice_metric'), percent=True)} | "
            f"order={self._fmt(self._get(logs, 'order_accuracy'), percent=True)} | "
            f"val_order={self._fmt(self._get(logs, 'val_order_accuracy'), percent=True)}"
        )
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--base-filters", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--progress-mode", choices=["tqdm", "line", "keras"], default="tqdm")
    ap.add_argument("--show-summary", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from exact epoch_XXX.weights.h5 checkpoint if available")
    ap.add_argument("--force-restart", action="store_true", help="Ignore checkpoints and train from scratch")
    ap.add_argument("--max-keep-checkpoints", type=int, default=3)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.results_dir) / "visible_order"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, n_train = make_ds(dataset_dir, "train", args.batch_size, shuffle=True, do_aug=True)
    val_ds, n_val = make_ds(dataset_dir, "val", args.batch_size)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    val_steps = math.ceil(n_val / args.batch_size)
    print("="*80)
    print("6v1 VISIBLE A/B/C + ORDER TRAINING")
    print("="*80)
    print(f"Dataset: train={n_train}, val={n_val}")
    print(f"Batch={args.batch_size} | steps/epoch={steps_per_epoch} | val_steps={val_steps}")
    print(f"Epochs={args.epochs} | patience={args.patience} | base_filters={args.base_filters} | lr={args.lr}")
    print(f"Progress mode: {args.progress_mode}")

    model = build_model(base=args.base_filters)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss={"seg": seg_loss, "order": "categorical_crossentropy"},
        loss_weights={"seg": 1.0, "order": 0.35},
        metrics={"seg": [dice_metric], "order": ["accuracy"]},
    )
    if args.show_summary:
        model.summary()
    else:
        print(f"Model params: {model.count_params():,} (use --show-summary nếu muốn xem full summary)")

    ckpt = out_dir / "best_visible_order_teacher.keras"
    final_model_path = out_dir / "final_visible_order_teacher.keras"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    if args.force_restart:
        print("[6v1] force restart: ignore old checkpoints")
    elif args.resume:
        last_epoch, last_ckpt = latest_weight_checkpoint(ckpt_dir)
        if last_ckpt is not None:
            model.load_weights(str(last_ckpt))
            initial_epoch = last_epoch
            print(f"[6v1] Resumed from {last_ckpt} -> initial_epoch={initial_epoch}")
        elif final_model_path.exists():
            loaded = keras.models.load_model(str(final_model_path), compile=False, safe_mode=False)
            model.set_weights(loaded.get_weights())
            print(f"[6v1] Loaded final model weights: {final_model_path}")
        elif ckpt.exists():
            loaded = keras.models.load_model(str(ckpt), compile=False, safe_mode=False)
            model.set_weights(loaded.get_weights())
            print(f"[6v1] Loaded best model weights: {ckpt}")
        else:
            print("[6v1] No checkpoint found, train from scratch")

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_seg_dice_metric", mode="max", save_best_only=True, verbose=0),
        keras.callbacks.ModelCheckpoint(str(ckpt_dir / "epoch_{epoch:03d}.weights.h5"), save_weights_only=True, save_freq="epoch", verbose=0),
        KeepLastCheckpoints(ckpt_dir, keep=args.max_keep_checkpoints),
        keras.callbacks.EarlyStopping(monitor="val_seg_dice_metric", mode="max", patience=args.patience, restore_best_weights=True, verbose=0),
        keras.callbacks.CSVLogger(str(out_dir / "train_visible_order_epoch_log.csv"), append=bool(args.resume and initial_epoch > 0)),
    ]
    fit_verbose = 1 if args.progress_mode == "keras" else 0
    if args.progress_mode != "keras":
        callbacks.insert(0, CleanProgressCallback(args.epochs, steps_per_epoch, args.progress_mode))

    if initial_epoch >= args.epochs:
        print(f"[6v1] Already reached epoch {initial_epoch}/{args.epochs}; skip training and evaluate best model.")
        hist = None
    else:
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=fit_verbose,
        )
        if hist is not None and hist.history:
            save_history(hist, out_dir)
    model.save(str(final_model_path))
    print(f"[6v1] Saved best model : {ckpt}")
    print(f"[6v1] Saved final model: {out_dir / 'final_visible_order_teacher.keras'}")
    print(f"[6v1] Saved history    : {out_dir / 'history_visible_order.csv'}")

    if not ckpt.exists():
        model.save(str(ckpt))
    best = keras.models.load_model(str(ckpt), compile=False, safe_mode=False)
    metrics = evaluate_real_test(best, dataset_dir, out_dir, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
