"""TrOCR-based inference for handwritten numeric fields.

Uses microsoft/trocr-base-handwritten (pre-trained on large handwriting datasets).
If a fine-tuned checkpoint exists at model/checkpoints/trocr_finetuned/, it is
preferred over the base model.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
TROCR_FINETUNED = CHECKPOINT_DIR / "trocr_finetuned"
TROCR_BASE = "microsoft/trocr-base-handwritten"

# Characters we accept in the output; everything else is stripped
ALLOWED_CHARS = set("0123456789.,-+")


def load_trocr(model_path: str = None) -> tuple:
    """Load TrOCR processor and model.

    Args:
        model_path: HuggingFace model ID or local path. If None, uses the
                    fine-tuned checkpoint if available, else the base model.

    Returns:
        Tuple of (processor, model, device).
    """
    if model_path is None:
        model_path = str(TROCR_FINETUNED) if TROCR_FINETUNED.exists() else TROCR_BASE

    print(f"Loading TrOCR from: {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return processor, model, device


def predict_image(image: np.ndarray, processor, model, device) -> tuple[str, float]:
    """Run TrOCR inference on a single crop image.

    Args:
        image: Grayscale or BGR numpy array.
        processor: TrOCRProcessor instance.
        model: VisionEncoderDecoderModel instance.
        device: Torch device.

    Returns:
        Tuple of (decoded_text, confidence).
        Confidence is the mean max-softmax probability over generated tokens.
    """
    # Convert to RGB PIL (TrOCR ViT encoder expects 3-channel input)
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image_rgb)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Keep only numeric / decimal characters; normalise comma → dot
    text = "".join(c for c in text if c in ALLOWED_CHARS)
    text = text.replace(",", ".")

    # Confidence: mean of per-token max softmax probability
    if outputs.scores:
        token_probs = [
            torch.softmax(s, dim=-1).max(dim=-1).values.item()
            for s in outputs.scores
        ]
        confidence = float(np.mean(token_probs))
    else:
        confidence = 0.0

    return text, confidence


def predict_file(image_path: str, model_path: str = None) -> tuple[str, float]:
    """Convenience function: predict from an image file path.

    Returns:
        Tuple of (decoded_text, confidence).
    """
    processor, model, device = load_trocr(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return predict_image(image, processor, model, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TrOCR inference on a crop image")
    parser.add_argument("image", type=str, help="Path to crop image")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path or HuggingFace model ID (default: auto)")
    args = parser.parse_args()

    text, confidence = predict_file(args.image, args.model)
    print(f"Predicted: {text}")
    print(f"Confidence: {confidence:.2%}")
