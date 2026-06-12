"""
5v1_split_data.py
Split 5,000 processed samples into:
- 4,000 model-development samples -> train/val/test
- 1,000 real_test samples kept untouched for final evaluation

Default 4,000 split:
- train = 2,800
- val   = 600
- test  = 600
- real_test = 1,000
"""
from pathlib import Path
import shutil
import random
import csv

CLEAR_OLD_DATASET = True
ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_data_256"
IMAGE_DIR = PROCESSED / "images"
DATASET_DIR = ROOT / "dataset"

MASK_SUBDIRS = ["masks_A", "masks_B", "masks_C", "visible_A", "visible_B", "gap_A", "gap_B"]

TRAIN_POOL_SIZE = 4000
REAL_TEST_SIZE = 1000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
random.seed(SEED)

if CLEAR_OLD_DATASET and DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

splits = ["train", "val", "test", "real_test"]
for split in splits:
    (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    for sub in MASK_SUBDIRS:
        (DATASET_DIR / split / sub).mkdir(parents=True, exist_ok=True)

image_paths = sorted(IMAGE_DIR.glob("*.png"))
names = [p.name for p in image_paths]
if not names:
    raise FileNotFoundError("No processed images found. Run 4v1_preprocess_to_256.py first.")

random.shuffle(names)

if len(names) < TRAIN_POOL_SIZE + REAL_TEST_SIZE:
    raise ValueError(f"Need at least {TRAIN_POOL_SIZE + REAL_TEST_SIZE} images, found {len(names)}. Run 3v1 with NUM_SAMPLES=5000.")

train_pool = names[:TRAIN_POOL_SIZE]
real_test_names = names[TRAIN_POOL_SIZE:TRAIN_POOL_SIZE + REAL_TEST_SIZE]

n_train = int(TRAIN_POOL_SIZE * TRAIN_RATIO)
n_val = int(TRAIN_POOL_SIZE * VAL_RATIO)
train_names = train_pool[:n_train]
val_names = train_pool[n_train:n_train + n_val]
test_names = train_pool[n_train + n_val:]

print(f"Total processed: {len(names)}")
print(f"Train pool     : {len(train_pool)}")
print(f"  train        : {len(train_names)}")
print(f"  val          : {len(val_names)}")
print(f"  test         : {len(test_names)}")
print(f"Real test hold : {len(real_test_names)}")

# Read labels metadata if exists
label_map = {}
label_csv = PROCESSED / "order_labels.csv"
if label_csv.exists():
    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["filename"]] = row


def copy_set(file_names, split_name):
    split_rows = []
    for name in file_names:
        shutil.copy2(PROCESSED / "images" / name, DATASET_DIR / split_name / "images" / name)
        for sub in MASK_SUBDIRS:
            src = PROCESSED / sub / name
            if not src.exists():
                raise FileNotFoundError(f"Missing {src}")
            shutil.copy2(src, DATASET_DIR / split_name / sub / name)
        if name in label_map:
            split_rows.append(label_map[name])

    if split_rows:
        out_csv = DATASET_DIR / split_name / "order_labels.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"Saved labels: {out_csv}")

copy_set(train_names, "train")
copy_set(val_names, "val")
copy_set(test_names, "test")
copy_set(real_test_names, "real_test")

# master split manifest
with open(DATASET_DIR / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "count"])
    writer.writerow(["train", len(train_names)])
    writer.writerow(["val", len(val_names)])
    writer.writerow(["test", len(test_names)])
    writer.writerow(["real_test", len(real_test_names)])

print("Done splitting dataset.")
