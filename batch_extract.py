#!/usr/bin/env python3
"""Batch-extract all transformer test PDFs → Excel workbook.

Usage:
    python batch_extract.py                          # process all PDFs
    python batch_extract.py --pdf-dir "path/to/dir" # custom PDF folder
    python batch_extract.py --no-resume              # reprocess everything
    python batch_extract.py --to-excel               # rebuild Excel from saved progress only

Progress is saved after every PDF to output/progress.jsonl so the run can be
interrupted and resumed freely. The Excel file is written at the end.

Output:
    output/transformer_data.xlsx
        Sheet "Data"   — one row per PDF, one column per field (corrected value)
        Sheet "Errors" — PDFs that failed to process (if any)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.trocr_predict import load_trocr
from model.trocr_predict import predict_image as trocr_predict
from pipeline.pdf_to_images import convert_pdf
from pipeline.region_cropper import REF_PAGE0, REF_PAGE1, crop_fields, load_field_config
from postprocess.rules import validate_all

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_PDF_DIR = Path(
    r"C:\Users\eren_\OneDrive\Belgeler\onedrive\Masaüstü\test reports\raw"
)
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TMP_IMAGES = OUTPUT_DIR / "_tmp" / "images"
TMP_CROPS = OUTPUT_DIR / "_tmp" / "crops"
PROGRESS_FILE = OUTPUT_DIR / "progress.jsonl"
EXCEL_FILE = OUTPUT_DIR / "transformer_data.xlsx"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ref_images() -> dict:
    """Load alignment reference images if they exist."""
    ref_images = {}
    for page_num, ref_path in [(0, REF_PAGE0), (1, REF_PAGE1)]:
        if ref_path.exists():
            img = cv2.imread(str(ref_path))
            if img is not None:
                ref_images[page_num] = img
    return ref_images


def load_progress() -> dict:
    """Return {filename: row_dict} for all already-processed PDFs."""
    done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["filename"]] = row
    return done


def append_progress(row: dict):
    """Append one result row to the JSONL progress file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_to_excel(rows: list, field_names: list, output_path: Path):
    """Write collected rows to a two-sheet Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_rows = [r for r in rows if "_error" not in r]
    error_rows = [r for r in rows if "_error" in r]

    # Ordered columns: filename first, then every field, then summary columns
    ordered_cols = ["filename"] + field_names + ["avg_confidence", "flags"]

    df_data = pd.DataFrame(data_rows)
    # Keep only known columns in order; missing ones become NaN (→ empty cell)
    df_data = df_data.reindex(columns=[c for c in ordered_cols if c in df_data.columns or c == "filename"])

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="Data", index=False)
        if error_rows:
            df_err = pd.DataFrame(error_rows)[["filename", "_error"]]
            df_err.to_excel(writer, sheet_name="Errors", index=False)

    n = len(data_rows)
    e = len(error_rows)
    print(f"\nExcel saved: {output_path}  ({n} rows, {e} errors)")


# ── Core processing ───────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    processor,
    model,
    device,
    fields: dict,
    ref_images: dict,
) -> dict:
    """Run the full pipeline on one PDF; return a flat result dict."""
    pdf_name = pdf_path.stem

    # Step 1: PDF → images
    TMP_IMAGES.mkdir(parents=True, exist_ok=True)
    image_paths = convert_pdf(pdf_path, TMP_IMAGES)
    if not image_paths:
        return {"filename": pdf_path.name, "_error": "No images extracted from PDF"}

    # Step 2: Crop all fields
    TMP_CROPS.mkdir(parents=True, exist_ok=True)
    all_crops = []
    for img_path in image_paths:
        crops = crop_fields(
            img_path, fields, TMP_CROPS, preprocess=True, ref_images=ref_images
        )
        all_crops.extend(crops)

    if not all_crops:
        return {
            "filename": pdf_path.name,
            "_error": "No fields cropped — calibration coordinates may not match this form",
        }

    # Step 3: TrOCR inference on each crop
    predictions: dict[str, tuple[str, float]] = {}
    for crop_path in all_crops:
        field_name = crop_path.stem
        if field_name.startswith(pdf_name + "_"):
            field_name = field_name[len(pdf_name) + 1:]

        img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            predictions[field_name] = ("", 0.0)
        else:
            text, conf = trocr_predict(img, processor, model, device)
            predictions[field_name] = (text, conf)

    # Step 4: Validate and auto-correct
    raw_values = {name: text for name, (text, _) in predictions.items()}
    results = validate_all(raw_values)

    # Step 5: Build flat output row
    row = {"filename": pdf_path.name}
    flags = []
    confidences = []

    for field_name, result in results.items():
        row[field_name] = result.get("corrected", result.get("original", ""))
        _, conf = predictions.get(field_name, ("", 0.0))
        confidences.append(conf)

        if not result.get("in_range", True):
            flags.append(f"{field_name}: {result['validation_error']}")
        elif result.get("was_corrected"):
            flags.append(f"{field_name} auto-corrected ({result['correction_reason']})")

    row["avg_confidence"] = round(
        sum(confidences) / max(len(confidences), 1), 3
    )
    row["flags"] = " | ".join(flags)

    return row


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract transformer test PDFs to Excel"
    )
    parser.add_argument(
        "--pdf-dir", default=str(DEFAULT_PDF_DIR),
        help="Folder containing PDF files (default: OneDrive raw folder)"
    )
    parser.add_argument(
        "--output", default=str(EXCEL_FILE),
        help="Output Excel file path"
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Skip already-processed PDFs (default: on)"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Reprocess all PDFs from scratch"
    )
    parser.add_argument(
        "--to-excel", action="store_true",
        help="Only convert existing progress.jsonl to Excel, skip OCR"
    )
    args = parser.parse_args()

    excel_path = Path(args.output)
    fields = load_field_config()
    field_names = list(fields.keys())

    # ── Excel-only mode ───────────────────────────────────────────────────────
    if args.to_excel:
        rows = list(load_progress().values())
        if not rows:
            print(f"No progress found at {PROGRESS_FILE}. Run without --to-excel first.")
            return
        save_to_excel(rows, field_names, excel_path)
        return

    # ── Find PDFs ─────────────────────────────────────────────────────────────
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))   # case-insensitive on Windows
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    # ── Resume logic ──────────────────────────────────────────────────────────
    if not args.resume and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress file cleared (--no-resume).")

    done = load_progress()
    remaining = [p for p in pdfs if p.name not in done]

    print(f"PDFs found : {len(pdfs)}")
    print(f"Already done: {len(done)}")
    print(f"To process  : {len(remaining)}")

    if not remaining:
        print("\nAll PDFs already processed.")
        save_to_excel(list(done.values()), field_names, excel_path)
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading TrOCR model (microsoft/trocr-base-handwritten)...")
    print("Note: no fine-tuned weights found — using the base model.")
    processor, model, device = load_trocr()
    print(f"Model running on: {device}\n")

    # ── Load alignment references ─────────────────────────────────────────────
    ref_images = load_ref_images()
    if ref_images:
        print(f"Alignment references loaded for pages: {sorted(ref_images.keys())}")
    else:
        print("No reference images found — cropping at fixed coordinates (no alignment)")
    print()

    # ── Process ───────────────────────────────────────────────────────────────
    total = len(pdfs)
    try:
        for pdf_path in remaining:
            n_done = len(load_progress())
            print(f"[{n_done + 1}/{total}] {pdf_path.name} ...", end=" ", flush=True)
            try:
                row = process_pdf(pdf_path, processor, model, device, fields, ref_images)
                append_progress(row)
                if "_error" in row:
                    print(f"ERROR — {row['_error']}")
                else:
                    n_flags = row["flags"].count("|") + 1 if row["flags"] else 0
                    print(
                        f"done  conf={row.get('avg_confidence', '?'):.2f}  "
                        f"flags={n_flags}"
                    )
            except Exception as exc:
                err_row = {"filename": pdf_path.name, "_error": str(exc)}
                append_progress(err_row)
                print(f"FAILED — {exc}")
            finally:
                # Remove per-PDF temp files to keep disk usage low
                shutil.rmtree(TMP_IMAGES, ignore_errors=True)
                shutil.rmtree(TMP_CROPS, ignore_errors=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved. Re-run to continue.")

    # ── Write Excel ───────────────────────────────────────────────────────────
    all_rows = list(load_progress().values())
    save_to_excel(all_rows, field_names, excel_path)


if __name__ == "__main__":
    main()
