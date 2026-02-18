"""Phase 1: GUI labeling tool for cropped field images.

Opens each crop in an OpenCV window. User types the value they see,
then presses Enter to save. Saves labels to data/labels.csv.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "crops"
LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"


def load_existing_labels(csv_path: Path) -> dict[str, str]:
    """Load already-labeled image paths from the CSV."""
    labeled = {}
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled[row["image_path"]] = row["value"]
    return labeled


def make_display(crop: np.ndarray, field_name: str, idx: int, total: int,
                 typed: str, msg: str = "") -> np.ndarray:
    """Build the display image: zoomed crop + info bar + typed text."""
    # Scale crop to be visible (at least 400px wide)
    h, w = crop.shape[:2]
    scale = max(1, 400 // max(w, 1))
    zoomed = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Convert grayscale to BGR if needed
    if len(zoomed.shape) == 2:
        zoomed = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR)

    # Build info panel below the crop
    panel_h = 140
    panel_w = max(zoomed.shape[1], 600)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    cv2.putText(panel, f"[{idx}/{total}] {field_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(panel, f"Type value + ENTER | 's'=skip | 'q'=quit+save | Backspace=delete",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, f"Value: {typed}_", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if msg:
        cv2.putText(panel, msg, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    # Pad zoomed crop to match panel width
    if zoomed.shape[1] < panel_w:
        pad = np.zeros((zoomed.shape[0], panel_w - zoomed.shape[1], 3), dtype=np.uint8)
        zoomed = np.hstack([zoomed, pad])
    elif zoomed.shape[1] > panel_w:
        panel2 = np.zeros((panel_h, zoomed.shape[1], 3), dtype=np.uint8)
        panel2[:, :panel_w] = panel
        panel = panel2

    return np.vstack([zoomed, panel])


def run_labeling(crops_dir: Path = CROPS_DIR, labels_csv: Path = LABELS_CSV):
    """GUI labeling session using OpenCV.

    Shows each crop image in a window. User types the value and presses Enter.
    Supports resume -- already-labeled images are skipped.
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

    total = len(unlabeled)
    print(f"Found {total} unlabeled images ({len(labeled)} already done)")
    print("Label each crop in the GUI window. Press Enter to save, 's' to skip, 'q' to quit.\n")

    # Ensure CSV exists with header
    write_header = not labels_csv.exists()
    csvfile = open(labels_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["image_path", "field_name", "value"])

    cv2.namedWindow("Labeling Tool", cv2.WINDOW_AUTOSIZE)
    saved_count = 0

    for i, crop_path in enumerate(unlabeled, 1):
        # Extract field name: {pdf_name}_{field_name}.png -> field_name
        parts = crop_path.stem.split("_", 1)
        field_name = parts[1] if len(parts) > 1 else crop_path.stem

        crop = cv2.imread(str(crop_path))
        if crop is None:
            print(f"  Could not read {crop_path.name}, skipping")
            continue

        typed = ""
        msg = ""

        while True:
            display = make_display(crop, field_name, i, total, typed, msg)
            cv2.imshow("Labeling Tool", display)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC = quit
                print("Quitting. Progress saved.")
                csvfile.close()
                cv2.destroyAllWindows()
                return

            if key == ord("q") and typed == "":
                print("Quitting. Progress saved.")
                csvfile.close()
                cv2.destroyAllWindows()
                return

            if key == ord("s") and typed == "":
                print(f"  [{i}/{total}] {field_name}: SKIPPED")
                break

            if key == 13:  # Enter
                if typed == "":
                    msg = "Type a value first, or press 's' to skip"
                    continue
                writer.writerow([str(crop_path), field_name, typed])
                csvfile.flush()
                saved_count += 1
                print(f"  [{i}/{total}] {field_name}: {typed}")
                break

            if key == 8:  # Backspace
                typed = typed[:-1]
                msg = ""
                continue

            # Accept printable chars (digits, dot, minus, plus)
            ch = chr(key) if 32 <= key < 127 else None
            if ch is not None:
                typed += ch
                msg = ""

    cv2.destroyAllWindows()
    csvfile.close()

    total_labeled = len(load_existing_labels(labels_csv))
    print(f"\nLabeled this session: {saved_count}")
    print(f"Total labeled: {total_labeled}/{len(crop_files)}")


if __name__ == "__main__":
    run_labeling()
