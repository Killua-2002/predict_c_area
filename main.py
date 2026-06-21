"""
Main runner for the NST predict C-area / missing-completion project.

Edit the CONFIG block first. Every notebook cell from the old runner is now
represented as a True/False stage here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# CONFIG - edit this block first
# =============================================================================
REPO_URL = "https://github.com/Killua-2002/predict_c_area.git"
PROJECT_ROOT = Path(__file__).resolve().parent

# Colab / Drive
USE_GOOGLE_DRIVE = False
DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/nst_tach_results/results")
BACKUP_RESULTS_TO_DRIVE = True
SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP = True
ZIP_FULL_PIPELINE_OUTPUTS = True

# Local project paths
SOURCE_SINGLE_CHROMOSOMES_DIR = PROJECT_ROOT / "source_data" / "single_chromosomes"
PREPARED_SINGLE_CHROMOSOMES_DIR = PROJECT_ROOT / "prepared_single_chromosomes" / "images_rgba"
GENERATED_DATA_DIR = PROJECT_ROOT / "generated_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data_256"
DATASET_DIR = PROJECT_ROOT / "dataset"
LOCAL_RESULTS_DIR = PROJECT_ROOT / "results"

# Pipeline toggles
RUN_ENV_CHECK = True
INSTALL_DEPS = False
CLEAN_LOCAL_DATA = False
USE_EXISTING_DATASET = True

RUN_PREPARE_SINGLE_CHROMOSOMES = False
RUN_GENERATE_SYNTHETIC_IMAGES = False
RUN_PREPROCESS_TO_SIZE = False
RUN_SPLIT_DATA = False

RUN_MODEL_VISIBLE_ORDER = True
RUN_MODEL_MISSING_COMPLETION = True
RUN_EVALUATE_FULL_PIPELINE = True

# Model branch toggles inside Model 2
RUN_MISSING_TEACHER = True
RUN_MISSING_STUDENT = True

# Dataset / preprocessing params
NUM_SAMPLES = 5000
TRAIN_POOL_SIZE = 4000
REAL_TEST_SIZE = 1000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_SIZE = 256
SEED = 42
GENERATION_WORKERS = max(2, min(8, (os.cpu_count() or 2)))
PREVIEW_MODE = "none"  # one of: none, first, all

# Train params
BATCH_SIZE = 40
VISIBLE_EPOCHS = 200
MISSING_EPOCHS = 200
PATIENCE = 10
VISIBLE_BASE_FILTERS = 32
MISSING_BASE_FILTERS = 32
LR = 1e-4
MAX_KEEP_CHECKPOINTS = 3
RESUME = True
FORCE_RESTART = False
PROGRESS_MODE = "line"  # one of: tqdm, line, keras

# Evaluation params
MAX_SHOWCASE = 60
FORCE_RECOMPUTE_CACHE = False
SAVE_CANDIDATES = False


# =============================================================================
# Environment overrides - useful for the one-cell notebook
# =============================================================================
def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else Path(raw)


USE_GOOGLE_DRIVE = env_bool("USE_GOOGLE_DRIVE", USE_GOOGLE_DRIVE)
INSTALL_DEPS = env_bool("INSTALL_DEPS", INSTALL_DEPS)
CLEAN_LOCAL_DATA = env_bool("CLEAN_LOCAL_DATA", CLEAN_LOCAL_DATA)
USE_EXISTING_DATASET = env_bool("USE_EXISTING_DATASET", env_bool("USE_DATASET_FROM_GIT", USE_EXISTING_DATASET))
RUN_ENV_CHECK = env_bool("RUN_ENV_CHECK", RUN_ENV_CHECK)
BACKUP_RESULTS_TO_DRIVE = env_bool("BACKUP_RESULTS_TO_DRIVE", BACKUP_RESULTS_TO_DRIVE)
SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP = env_bool("SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP", SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP)
ZIP_FULL_PIPELINE_OUTPUTS = env_bool("ZIP_FULL_PIPELINE_OUTPUTS", ZIP_FULL_PIPELINE_OUTPUTS)

RUN_PREPARE_SINGLE_CHROMOSOMES = env_bool("RUN_PREPARE_SINGLE_CHROMOSOMES", RUN_PREPARE_SINGLE_CHROMOSOMES)
RUN_GENERATE_SYNTHETIC_IMAGES = env_bool("RUN_GENERATE_SYNTHETIC_IMAGES", RUN_GENERATE_SYNTHETIC_IMAGES)
RUN_PREPROCESS_TO_SIZE = env_bool("RUN_PREPROCESS_TO_SIZE", RUN_PREPROCESS_TO_SIZE)
RUN_SPLIT_DATA = env_bool("RUN_SPLIT_DATA", RUN_SPLIT_DATA)
RUN_MODEL_VISIBLE_ORDER = env_bool("RUN_MODEL_VISIBLE_ORDER", RUN_MODEL_VISIBLE_ORDER)
RUN_MODEL_MISSING_COMPLETION = env_bool("RUN_MODEL_MISSING_COMPLETION", RUN_MODEL_MISSING_COMPLETION)
RUN_EVALUATE_FULL_PIPELINE = env_bool("RUN_EVALUATE_FULL_PIPELINE", RUN_EVALUATE_FULL_PIPELINE)
RUN_MISSING_TEACHER = env_bool("RUN_MISSING_TEACHER", RUN_MISSING_TEACHER)
RUN_MISSING_STUDENT = env_bool("RUN_MISSING_STUDENT", RUN_MISSING_STUDENT)

SOURCE_SINGLE_CHROMOSOMES_DIR = env_path("SOURCE_SINGLE_CHROMOSOMES_DIR", SOURCE_SINGLE_CHROMOSOMES_DIR)
PREPARED_SINGLE_CHROMOSOMES_DIR = env_path("PREPARED_SINGLE_CHROMOSOMES_DIR", PREPARED_SINGLE_CHROMOSOMES_DIR)
GENERATED_DATA_DIR = env_path("GENERATED_DATA_DIR", GENERATED_DATA_DIR)
PROCESSED_DATA_DIR = env_path("PROCESSED_DATA_DIR", PROCESSED_DATA_DIR)
DATASET_DIR = env_path("DATASET_DIR", DATASET_DIR)
DRIVE_RESULTS_DIR = env_path("DRIVE_RESULTS_DIR", DRIVE_RESULTS_DIR)
LOCAL_RESULTS_DIR = env_path("LOCAL_RESULTS_DIR", LOCAL_RESULTS_DIR)

NUM_SAMPLES = env_int("NUM_SAMPLES", NUM_SAMPLES)
TRAIN_POOL_SIZE = env_int("TRAIN_POOL_SIZE", TRAIN_POOL_SIZE)
REAL_TEST_SIZE = env_int("REAL_TEST_SIZE", REAL_TEST_SIZE)
IMAGE_SIZE = env_int("IMAGE_SIZE", IMAGE_SIZE)
SEED = env_int("SEED", SEED)
GENERATION_WORKERS = env_int("GENERATION_WORKERS", GENERATION_WORKERS)
TRAIN_RATIO = env_float("TRAIN_RATIO", TRAIN_RATIO)
VAL_RATIO = env_float("VAL_RATIO", VAL_RATIO)

BATCH_SIZE = env_int("BATCH_SIZE", BATCH_SIZE)
VISIBLE_EPOCHS = env_int("VISIBLE_EPOCHS", VISIBLE_EPOCHS)
MISSING_EPOCHS = env_int("MISSING_EPOCHS", MISSING_EPOCHS)
PATIENCE = env_int("PATIENCE", PATIENCE)
VISIBLE_BASE_FILTERS = env_int("VISIBLE_BASE_FILTERS", VISIBLE_BASE_FILTERS)
MISSING_BASE_FILTERS = env_int("MISSING_BASE_FILTERS", MISSING_BASE_FILTERS)
LR = env_float("LR", LR)
MAX_KEEP_CHECKPOINTS = env_int("MAX_KEEP_CHECKPOINTS", MAX_KEEP_CHECKPOINTS)
RESUME = env_bool("RESUME", RESUME)
FORCE_RESTART = env_bool("FORCE_RESTART", FORCE_RESTART)
MAX_SHOWCASE = env_int("MAX_SHOWCASE", MAX_SHOWCASE)
FORCE_RECOMPUTE_CACHE = env_bool("FORCE_RECOMPUTE_CACHE", FORCE_RECOMPUTE_CACHE)
SAVE_CANDIDATES = env_bool("SAVE_CANDIDATES", SAVE_CANDIDATES)
PREVIEW_MODE = os.environ.get("PREVIEW_MODE", PREVIEW_MODE)
PROGRESS_MODE = os.environ.get("PROGRESS_MODE", "keras")

RESULTS_DIR = LOCAL_RESULTS_DIR


def command_text(args: list[str | Path]) -> str:
    parts = []
    for arg in args:
        text = str(arg)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def run_cmd(args: list[str | Path], title: str, check: bool = True) -> int:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("$", command_text(args))
    sys.stdout.flush()
    t0 = time.time()
    proc = subprocess.Popen(
        [str(a) for a in args],
        cwd=PROJECT_ROOT,
    )
    code = proc.wait()
    print(f"\n[exit={code}] elapsed={time.time() - t0:.1f}s")
    if check and code != 0:
        raise RuntimeError(f"Command failed: {command_text(args)}")
    return code


def maybe_mount_drive() -> None:
    if not USE_GOOGLE_DRIVE:
        return
    # The Colab notebook already mounts the drive before running main.py.
    # We just ensure the results directory exists.
    DRIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Local train/eval results: {RESULTS_DIR}")
    print(f"Drive backup/resume path: {DRIVE_RESULTS_DIR}")


def copy_tree_filtered(src: Path, dst: Path, skip_predicted_masks: bool = True) -> int:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print("[SKIP] source not found:", src)
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        if skip_predicted_masks and "predicted_masks" in rel.parts:
            continue
        if "__pycache__" in rel.parts:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)
        copied += 1
    return copied


def restore_results_from_drive() -> None:
    if not (USE_GOOGLE_DRIVE and BACKUP_RESULTS_TO_DRIVE):
        return
    maybe_mount_drive()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Restore Drive -> local")
    print("  from:", DRIVE_RESULTS_DIR)
    print("  to  :", RESULTS_DIR)
    copied = copy_tree_filtered(
        DRIVE_RESULTS_DIR,
        RESULTS_DIR,
        skip_predicted_masks=SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP,
    )
    print("[RESTORE DONE]", copied, "files copied")


def backup_stage_to_drive(stage_name: str) -> None:
    if not (USE_GOOGLE_DRIVE and BACKUP_RESULTS_TO_DRIVE):
        return
    maybe_mount_drive()
    src = RESULTS_DIR / stage_name
    dst = DRIVE_RESULTS_DIR / stage_name
    copied = copy_tree_filtered(
        src,
        dst,
        skip_predicted_masks=SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP,
    )
    print(f"[BACKUP {stage_name} DONE]", copied, "files copied to", dst)


def backup_all_results_to_drive() -> None:
    if not (USE_GOOGLE_DRIVE and BACKUP_RESULTS_TO_DRIVE):
        return
    maybe_mount_drive()
    copied = copy_tree_filtered(
        RESULTS_DIR,
        DRIVE_RESULTS_DIR,
        skip_predicted_masks=SKIP_PREDICTED_MASKS_IN_DRIVE_BACKUP,
    )
    print("[BACKUP ALL DONE]", copied, "files copied to", DRIVE_RESULTS_DIR)


def zip_full_pipeline_outputs_to_drive() -> None:
    if not (USE_GOOGLE_DRIVE and BACKUP_RESULTS_TO_DRIVE and ZIP_FULL_PIPELINE_OUTPUTS):
        return
    maybe_mount_drive()
    src = RESULTS_DIR / "full_pipeline_real_test"
    if not src.exists():
        print("[ZIP SKIP] missing:", src)
        return
    zip_base = DRIVE_RESULTS_DIR / "full_pipeline_real_test_outputs"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=src)
    print("[ZIP DONE]", zip_path)


def env_check() -> None:
    print("Python:", sys.version)
    print("Project:", PROJECT_ROOT)
    print("Results:", RESULTS_DIR)
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    print(f"Disk free: {free / (1024 ** 3):.1f} GB / {total / (1024 ** 3):.1f} GB")
    if shutil.which("nvidia-smi"):
        run_cmd(["nvidia-smi"], "GPU check", check=False)
    else:
        print("GPU: nvidia-smi not found")


def install_deps() -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "pillow",
            "opencv-python",
            "matplotlib",
            "pandas",
            "scikit-learn",
            "tqdm",
        ],
        "Install light dependencies",
    )


def clean_local_outputs() -> None:
    folders = [
        RESULTS_DIR,
        PROJECT_ROOT / "result",
        PROJECT_ROOT / "checkpoint",
        PROJECT_ROOT / "checkpoints",
    ]
    if not USE_EXISTING_DATASET:
        folders += [GENERATED_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR]

    for folder in folders:
        if folder.exists():
            print(f"Deleting local old output: {folder}")
            shutil.rmtree(folder)


def print_dataset_contract() -> None:
    print("\nDataset contract")
    print(f"- Raw single chromosomes: {SOURCE_SINGLE_CHROMOSOMES_DIR}")
    print(f"- Prepared RGBA singles : {PREPARED_SINGLE_CHROMOSOMES_DIR}")
    print(f"- Generated samples     : {GENERATED_DATA_DIR}")
    print(f"- Processed {IMAGE_SIZE} data    : {PROCESSED_DATA_DIR}")
    print(f"- Final dataset         : {DATASET_DIR}")
    print(f"- Use existing dataset  : {USE_EXISTING_DATASET}")
    print("Expected final dataset folders per split:")
    print("  images, masks_A, masks_B, masks_C, visible_A, visible_B, gap_A, gap_B, order_labels.csv")


def py_script(path: str) -> list[str | Path]:
    return [sys.executable, "-u", PROJECT_ROOT / path]


def run_pipeline() -> None:
    maybe_mount_drive()
    if RUN_ENV_CHECK:
        env_check()
    if INSTALL_DEPS:
        install_deps()
    if CLEAN_LOCAL_DATA:
        clean_local_outputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_dataset_contract()

    if RUN_PREPARE_SINGLE_CHROMOSOMES:
        print("\n" + "#" * 90)
        print("### BƯỚC 1: TIỀN XỬ LÝ - CHUẨN BỊ SINGLE CHROMOSOMES ###")
        print("#" * 90)
        run_cmd(
            py_script("preprocessing/2v1_prepare_single_chromosomes.py")
            + [
                "--source-dir",
                SOURCE_SINGLE_CHROMOSOMES_DIR,
                "--out-dir",
                PREPARED_SINGLE_CHROMOSOMES_DIR,
                "--clear-old",
            ],
            "Prepare single chromosome images",
        )

    if RUN_GENERATE_SYNTHETIC_IMAGES:
        print("\n" + "#" * 90)
        print("### BƯỚC 2: TIỀN XỬ LÝ - TẠO DỮ LIỆU TỔNG HỢP (SYNTHETIC) ###")
        print("#" * 90)
        run_cmd(
            py_script("preprocessing/3v1_generate_synthetic_masks.py")
            + [
                "--source-dir",
                PREPARED_SINGLE_CHROMOSOMES_DIR,
                "--out-root",
                GENERATED_DATA_DIR,
                "--num-samples",
                str(NUM_SAMPLES),
                "--seed",
                str(SEED),
                "--progress-every",
                "100",
                "--workers",
                str(GENERATION_WORKERS),
                "--preview-mode",
                PREVIEW_MODE,
            ],
            "Generate synthetic overlap samples",
        )

    if RUN_PREPROCESS_TO_SIZE:
        print("\n" + "#" * 90)
        print(f"### BƯỚC 3: TIỀN XỬ LÝ - RESIZE ẢNH VỀ {IMAGE_SIZE}x{IMAGE_SIZE} ###")
        print("#" * 90)
        run_cmd(
            py_script("preprocessing/4v1_preprocess_to_256.py")
            + [
                "--input-root",
                GENERATED_DATA_DIR,
                "--output-root",
                PROCESSED_DATA_DIR,
                "--target-size",
                str(IMAGE_SIZE),
                "--clear-old",
            ],
            f"Preprocess generated samples to {IMAGE_SIZE}x{IMAGE_SIZE}",
        )

    if RUN_SPLIT_DATA:
        print("\n" + "#" * 90)
        print("### BƯỚC 4: TIỀN XỬ LÝ - CHIA TẬP TRAIN / VAL / TEST ###")
        print("#" * 90)
        run_cmd(
            py_script("preprocessing/5v1_split_data.py")
            + [
                "--processed-dir",
                PROCESSED_DATA_DIR,
                "--dataset-dir",
                DATASET_DIR,
                "--train-pool-size",
                str(TRAIN_POOL_SIZE),
                "--real-test-size",
                str(REAL_TEST_SIZE),
                "--train-ratio",
                str(TRAIN_RATIO),
                "--val-ratio",
                str(VAL_RATIO),
                "--seed",
                str(SEED),
                "--clear-old",
            ],
            "Split dataset into train/val/test/real_test",
        )

    if RUN_MODEL_VISIBLE_ORDER:
        print("\n" + "#" * 90)
        print("### BƯỚC 5: HUẤN LUYỆN MODEL 1 (VISIBLE MASKS & ORDER) ###")
        print("#" * 90)
        restore_results_from_drive()
        cmd = (
            py_script("models/6v1_train_visible_order.py")
            + [
                "--dataset-dir",
                DATASET_DIR,
                "--results-dir",
                RESULTS_DIR,
                "--epochs",
                str(VISIBLE_EPOCHS),
                "--batch-size",
                str(BATCH_SIZE),
                "--base-filters",
                str(VISIBLE_BASE_FILTERS),
                "--lr",
                str(LR),
                "--patience",
                str(PATIENCE),
                "--progress-mode",
                PROGRESS_MODE,
                "--max-keep-checkpoints",
                str(MAX_KEEP_CHECKPOINTS),
            ]
        )
        if RESUME:
            cmd.append("--resume")
        if FORCE_RESTART:
            cmd.append("--force-restart")
        run_cmd(cmd, "Train Model 1: visible A/B/C + order")
        backup_stage_to_drive("visible_order")

    if RUN_MODEL_MISSING_COMPLETION:
        print("\n" + "#" * 90)
        print("### BƯỚC 6: HUẤN LUYỆN MODEL 2 (MISSING COMPLETION - TEACHER & STUDENT) ###")
        print("#" * 90)
        restore_results_from_drive()
        cmd = (
            py_script("models/9v2_train_missing_completion_teacher_student.py")
            + [
                "--dataset-dir",
                DATASET_DIR,
                "--results-dir",
                RESULTS_DIR,
                "--epochs",
                str(MISSING_EPOCHS),
                "--batch-size",
                str(BATCH_SIZE),
                "--base-filters",
                str(MISSING_BASE_FILTERS),
                "--lr",
                str(LR),
                "--patience",
                str(PATIENCE),
                "--progress-mode",
                PROGRESS_MODE,
                "--max-keep-checkpoints",
                str(MAX_KEEP_CHECKPOINTS),
            ]
        )
        if not RUN_MISSING_TEACHER:
            cmd.append("--skip-teacher")
        if not RUN_MISSING_STUDENT:
            cmd.append("--skip-student")
        if RESUME:
            cmd.append("--resume")
        if FORCE_RESTART:
            cmd.append("--force-restart")
        run_cmd(cmd, "Train Model 2: missing completion teacher/student")
        backup_stage_to_drive("missing_completion")

    if RUN_EVALUATE_FULL_PIPELINE:
        print("\n" + "#" * 90)
        print("### BƯỚC 7: ĐÁNH GIÁ TỔNG THỂ (EVALUATE FULL PIPELINE) ###")
        print("#" * 90)
        backup_stage_to_drive("missing_completion")
        restore_results_from_drive()
        cmd = (
            py_script("evaluation/10v2_evaluate_full_missing_pipeline.py")
            + [
                "--dataset-dir",
                DATASET_DIR,
                "--results-dir",
                RESULTS_DIR,
                "--batch-size",
                str(BATCH_SIZE),
                "--max-showcase",
                str(MAX_SHOWCASE),
            ]
        )
        if FORCE_RECOMPUTE_CACHE:
            cmd.append("--force-recompute-cache")
        if SAVE_CANDIDATES:
            cmd.append("--save-candidates")
        run_cmd(cmd, "Evaluate full real_test pipeline")
        backup_stage_to_drive("full_pipeline_real_test")
        zip_full_pipeline_outputs_to_drive()

    print("\nPipeline finished.")
    print(f"Results folder: {RESULTS_DIR}")
    if USE_GOOGLE_DRIVE:
        print(f"Drive results folder: {DRIVE_RESULTS_DIR}")
    for metrics_path in [
        RESULTS_DIR / "visible_order" / "metrics_visible_order_real_test.json",
        RESULTS_DIR / "missing_completion" / "metrics_real_test_compare.json",
        RESULTS_DIR / "full_pipeline_real_test" / "metrics_full_pipeline_real_test.json",
    ]:
        if metrics_path.exists():
            print(f"\n{metrics_path}")
            try:
                print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2)[:4000])
            except Exception:
                print(metrics_path.read_text(encoding="utf-8")[:4000])


if __name__ == "__main__":
    run_pipeline()
