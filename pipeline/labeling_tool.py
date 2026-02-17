"""Phase 1: CLI labeling tool for cropped field images."""

import csv
import os
import subprocess
import sys
from pathlib import Path

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "crops"
LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"


def load_existing_labels(csv_path: Path) -> set[str]:
    """Load already-labeled image paths from the CSV."""
    labeled = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["image_path"])
    return labeled


def open_image(image_path: Path):
    """Open an image with the system default viewer."""
    if sys.platform == "win32":
        os.startfile(str(image_path))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(image_path)])
    else:
        subprocess.run(["xdg-open", str(image_path)])


def run_labeling(crops_dir: Path = CROPS_DIR, labels_csv: Path = LABELS_CSV):
    """Interactive CLI labeling session.

    Displays each unlabeled crop image and prompts the user to enter the value.
    Supports resume — already-labeled images are skipped.
    """
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_files = sorted(crops_dir.glob("*.png"))
    if not crop_files:
        print(f"No crop images found in {crops_dir}")
        print("Run region_cropper.py first to generate crops.")
        return

    labeled = load_existing_labels(labels_csv)
    unlabeled = [f for f in crop_files if str(f) not in labeled]

    if not unlabeled:
        print(f"All {len(crop_files)} images already labeled.")
        return

    print(f"Found {len(unlabeled)} unlabeled images ({len(labeled)} already done)")
    print("Enter the value you see in each image. Commands:")
    print("  (empty) = skip")
    print("  q       = quit and save")
    print()

    # Ensure CSV exists with header
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_path", "field_name", "value"])

        for i, crop_path in enumerate(unlabeled, 1):
            # Extract field name from filename: {pdf_name}_{field_name}.png
            field_name = "_".join(crop_path.stem.split("_")[1:]) if "_" in crop_path.stem else crop_path.stem

            print(f"[{i}/{len(unlabeled)}] {crop_path.name}")
            print(f"  Field: {field_name}")

            try:
                open_image(crop_path)
            except Exception:
                print(f"  (Could not open image viewer, check: {crop_path})")

            value = input("  Value: ").strip()
            if value.lower() == "q":
                print("Quitting. Progress saved.")
                break
            if value == "":
                print("  Skipped.")
                continue

            writer.writerow([str(crop_path), field_name, value])
            f.flush()
            print(f"  Saved: {value}")

    total_labeled = len(load_existing_labels(labels_csv))
    print(f"\nTotal labeled: {total_labeled}/{len(crop_files)}")


if __name__ == "__main__":
    run_labeling()
