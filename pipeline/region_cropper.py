"""Phase 2: Automatic field cropping based on fixed coordinates from fields.yaml."""

import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_engine import preprocess_image

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fields.yaml"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "crops"


def load_field_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load field definitions from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("fields", {})


def crop_fields(image_path: Path, fields: dict, output_dir: Path = CROPS_DIR,
                preprocess: bool = True) -> list[Path]:
    """Crop all defined fields from a single image.

    Args:
        image_path: Path to the full page image.
        fields: Field definitions dict from config.
        output_dir: Directory to save cropped images.
        preprocess: Whether to apply preprocessing to crops.

    Returns:
        List of saved crop file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load {image_path}")
        return []

    # Determine page number from filename (e.g., report_page0.png → page 0)
    stem = image_path.stem
    page_num = 0
    if "_page" in stem:
        try:
            page_num = int(stem.split("_page")[-1])
        except ValueError:
            page_num = 0

    pdf_name = stem.rsplit("_page", 1)[0] if "_page" in stem else stem
    saved = []

    for field_name, field_data in fields.items():
        if field_data.get("page", 0) != page_num:
            continue

        bbox = field_data.get("bbox", [0, 0, 0, 0])
        if bbox == [0, 0, 0, 0]:
            continue  # Uncalibrated field

        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"  Warning: Empty crop for {field_name} at {bbox}")
            continue

        if preprocess:
            crop = preprocess_image(crop, for_trocr=False)

        out_path = output_dir / f"{pdf_name}_{field_name}.png"
        cv2.imwrite(str(out_path), crop)
        saved.append(out_path)

    return saved


def crop_all(images_dir: Path = IMAGES_DIR, config_path: Path = CONFIG_PATH,
             output_dir: Path = CROPS_DIR) -> list[Path]:
    """Crop fields from all images in the images directory.

    Returns list of all saved crop paths.
    """
    fields = load_field_config(config_path)
    calibrated = {k: v for k, v in fields.items() if v.get("bbox") != [0, 0, 0, 0]}
    if not calibrated:
        print("No calibrated fields found. Run coordinate_picker.py first.")
        return []

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {images_dir}")
        return []

    all_crops = []
    for img_path in image_files:
        print(f"Cropping: {img_path.name}")
        crops = crop_fields(img_path, fields, output_dir)
        all_crops.extend(crops)
        print(f"  → {len(crops)} fields cropped")

    print(f"\nTotal: {len(all_crops)} crops from {len(image_files)} images")
    return all_crops


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop defined fields from page images")
    parser.add_argument("--image", type=str, help="Single image to crop (default: all in data/images/)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to fields.yaml")
    args = parser.parse_args()

    if args.image:
        fields = load_field_config(Path(args.config))
        crop_fields(Path(args.image), fields)
    else:
        crop_all(config_path=Path(args.config))
