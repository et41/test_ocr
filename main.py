import argparse
import json
import os
import sys

from ocr_engine import (
    extract_handwritten_tables,
    extract_structured_data,
    extract_tables_from_pdf,
    extract_text_from_pdf,
)


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF files using Tesseract OCR.")
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "--format",
        choices=["text", "json", "tables", "handwriting", "both", "all"],
        default="both",
        help="Output format: text, json, tables, handwriting, both (text+json), all (text+json+tables+handwriting) (default: both)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    if args.format in ("text", "both", "all"):
        print("Extracting text...")
        text = extract_text_from_pdf(args.input)
        text_path = os.path.join(args.output_dir, f"{base_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {text_path}")

    if args.format in ("json", "both", "all"):
        print("Extracting structured data...")
        data = extract_structured_data(args.input)
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

    if args.format in ("tables", "all"):
        print("Extracting tables...")
        tables_data = extract_tables_from_pdf(args.input)
        tables_path = os.path.join(args.output_dir, f"{base_name}_tables.json")
        with open(tables_path, "w", encoding="utf-8") as f:
            json.dump(tables_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {tables_path}")

    if args.format in ("handwriting", "all"):
        print("Extracting handwritten text...")
        hw_data = extract_handwritten_tables(args.input)
        hw_path = os.path.join(args.output_dir, f"{base_name}_handwritten.json")
        with open(hw_path, "w", encoding="utf-8") as f:
            json.dump(hw_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {hw_path}")

    print("Done.")


if __name__ == "__main__":
    main()
