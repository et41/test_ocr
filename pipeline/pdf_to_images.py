"""Phase 1: Batch convert PDFs to 300 DPI images."""

import os
import sys
from pathlib import Path

from pdf2image import convert_from_path

# Reuse POPPLER_PATH from existing ocr_engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_engine import POPPLER_PATH

RAW_PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


def convert_pdf(pdf_path: Path, output_dir: Path = IMAGES_DIR, dpi: int = 300) -> list[Path]:
    """Convert a single PDF to PNG images at the given DPI.

    Returns list of output image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = pdf_path.stem

    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    saved_paths = []
    for i, image in enumerate(images):
        out_path = output_dir / f"{pdf_name}_page{i}.png"
        image.save(str(out_path), "PNG")
        saved_paths.append(out_path)
        print(f"  Saved: {out_path.name}")

    return saved_paths


def convert_all(pdf_dir: Path = RAW_PDF_DIR, output_dir: Path = IMAGES_DIR, dpi: int = 300) -> list[Path]:
    """Convert all PDFs in the directory to images.

    Returns list of all output image paths.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return []

    all_paths = []
    for pdf_path in pdfs:
        print(f"Converting: {pdf_path.name}")
        all_paths.extend(convert_pdf(pdf_path, output_dir, dpi))

    print(f"\nConverted {len(pdfs)} PDF(s) → {len(all_paths)} image(s)")
    return all_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to images for the recognition pipeline")
    parser.add_argument("--input", type=str, help="Single PDF file to convert (default: all in data/raw_pdfs/)")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    args = parser.parse_args()

    if args.input:
        convert_pdf(Path(args.input), dpi=args.dpi)
    else:
        convert_all(dpi=args.dpi)
