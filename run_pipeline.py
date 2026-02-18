"""Full pipeline entry point: PDF → Image → Crop → Predict → Validate → Output JSON.

Usage:
    python run_pipeline.py path/to/report.pdf
    python run_pipeline.py data/raw_pdfs/           # process all PDFs in directory
    python run_pipeline.py report.pdf --review       # interactive review of flagged fields
    python run_pipeline.py report.pdf --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from correction.review_tool import format_output, review_results, save_corrections_for_retraining
from model.predict import load_model, predict_image
from pipeline.pdf_to_images import convert_pdf
from pipeline.region_cropper import REF_PAGE0, REF_PAGE1, crop_fields, load_field_config
from postprocess.rules import cross_validate, load_validation_rules, validate_all

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "images"
CROPS_DIR = DATA_DIR / "crops"


def process_pdf(pdf_path: Path, model, device, fields: dict, review: bool = False) -> dict:
    """Process a single PDF through the full pipeline.

    Returns structured output dict with all field values and confidence scores.
    """
    pdf_name = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    # Step 1: PDF → Images
    print("\n[1/5] Converting PDF to images...")
    image_paths = convert_pdf(pdf_path, IMAGES_DIR)
    if not image_paths:
        print("Error: No images generated from PDF.")
        return {}

    # Step 2: Crop fields (with alignment to reference)
    print("\n[2/5] Cropping fields...")
    ref_images = {}
    for page_num, ref_path in [(0, REF_PAGE0), (1, REF_PAGE1)]:
        if ref_path.exists():
            ref = cv2.imread(str(ref_path))
            if ref is not None:
                ref_images[page_num] = ref

    all_crops = []
    for img_path in image_paths:
        crops = crop_fields(img_path, fields, CROPS_DIR, preprocess=True, ref_images=ref_images)
        all_crops.extend(crops)
    print(f"  Cropped {len(all_crops)} fields")

    if not all_crops:
        print("Warning: No fields cropped. Check that fields.yaml has calibrated coordinates.")
        return {}

    # Step 3: Predict
    print("\n[3/5] Running model predictions...")
    predictions: dict[str, tuple[str, float]] = {}
    for crop_path in all_crops:
        # Extract field name from filename: {pdf_name}_{field_name}.png
        field_name = crop_path.stem
        if field_name.startswith(pdf_name + "_"):
            field_name = field_name[len(pdf_name) + 1:]

        image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            predictions[field_name] = ("", 0.0)
            continue

        text, confidence = predict_image(image, model, device)
        predictions[field_name] = (text, confidence)
        print(f"  {field_name}: {text} ({confidence:.1%})")

    # Step 4: Validate
    print("\n[4/5] Validating results...")
    raw_values = {name: text for name, (text, _) in predictions.items()}
    results = validate_all(raw_values)

    rules = load_validation_rules()
    warnings = cross_validate(results, rules)

    corrections = sum(1 for r in results.values() if r.get("was_corrected"))
    out_of_range = sum(1 for r in results.values() if not r.get("in_range", True))
    if corrections:
        print(f"  Auto-corrected {corrections} field(s)")
    if out_of_range:
        print(f"  {out_of_range} field(s) out of range")
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    # Step 5: Review (optional)
    if review:
        print("\n[5/5] Interactive review...")
        results = review_results(results, predictions)
        save_corrections_for_retraining(results, CROPS_DIR, pdf_name)
    else:
        print("\n[5/5] Skipping review (use --review to enable)")

    return format_output(results, predictions, warnings)


def main():
    parser = argparse.ArgumentParser(
        description="Handwriting recognition pipeline for transformer test reports"
    )
    parser.add_argument("input", type=str, help="PDF file or directory of PDFs")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--review", action="store_true", help="Enable interactive review of flagged fields")
    parser.add_argument("--confidence-threshold", type=float, default=0.85,
                        help="Confidence threshold for flagging (default: 0.85)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Gather PDF paths
    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDF files found in {input_path}")
            sys.exit(1)
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_paths = [input_path]
    else:
        print(f"Error: {input_path} is not a PDF file or directory")
        sys.exit(1)

    # Load model once
    print("Loading CRNN model...")
    try:
        model, device = load_model()
        print(f"Model loaded on {device}")
    except FileNotFoundError:
        print("Error: No trained model found at model/checkpoints/best_model.pt")
        print("Run model/train.py first to train the model.")
        sys.exit(1)

    # Load field config once
    fields = load_field_config()
    calibrated = sum(1 for v in fields.values() if v.get("bbox") != [0, 0, 0, 0])
    print(f"Field config: {calibrated}/{len(fields)} fields calibrated")

    # Process each PDF
    all_results = {}
    for pdf_path in pdf_paths:
        result = process_pdf(pdf_path, model, device, fields, review=args.review)
        all_results[pdf_path.stem] = result

    # Output
    output_data = all_results if len(pdf_paths) > 1 else next(iter(all_results.values()), {})

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n" + json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
