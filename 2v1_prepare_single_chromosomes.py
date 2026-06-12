"""
2v1_prepare_single_chromosomes.py
Flatten raw single chromosome images into prepared_single_chromosomes/images_rgba.

Input accepted:
    source_data/single_chromosomes/**.png|jpg|jpeg|bmp|tif|tiff

Output:
    prepared_single_chromosomes/images_rgba/*.png

This script is intentionally simple and robust: it converts every readable image to RGBA PNG.
The 3v1 generator will later extract chromosome masks from alpha if present, otherwise from background color.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source_data" / "single_chromosomes"
OUT_DIR = ROOT / "prepared_single_chromosomes" / "images_rgba"
CLEAR_OLD_OUTPUT = True
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source folder: {SOURCE_DIR}")

    if CLEAR_OLD_OUTPUT and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [p for p in SOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not paths:
        raise FileNotFoundError(f"No image files found under: {SOURCE_DIR}")

    ok = 0
    skipped = 0
    for idx, p in enumerate(sorted(paths), start=1):
        try:
            img = Image.open(p).convert("RGBA")
            out_name = f"single_{idx:06d}.png"
            img.save(OUT_DIR / out_name)
            ok += 1
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
            skipped += 1

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(paths)}")

    print("Done preparing single chromosomes.")
    print(f"Input images : {len(paths)}")
    print(f"Saved images : {ok}")
    print(f"Skipped      : {skipped}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
