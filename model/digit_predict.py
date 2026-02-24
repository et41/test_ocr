#!/usr/bin/env python3
"""
Inference: predict a handwritten number from a crop image using DigitCNN.

Drop-in replacement for trocr_predict.py.

Usage (CLI):
  python -m model.digit_predict data/crops/300049-01_hv_res_tap1_1w1u.png

Usage (library):
  from model.digit_predict import load_digit_cnn, predict_image
  model, device = load_digit_cnn()
  pred = predict_image("path/to/crop.png", model=model, device=device)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.char_segment import segment_crop
from model.digit_cnn import DigitCNN, IDX_TO_CHAR, NUM_CLASSES

MODEL_PATH = Path(__file__).resolve().parent / "digit_cnn.pth"

_model = None
_device = None


def load_digit_cnn(model_path: Path = MODEL_PATH):
    """Load and cache the DigitCNN model."""
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = DigitCNN(num_classes=NUM_CLASSES)
    _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
    _model.to(_device)
    _model.eval()
    return _model, _device


def predict_crop(img_gray: np.ndarray, model=None, device=None) -> str:
    """
    Predict the number string from a grayscale crop image.

    Args:
        img_gray: Grayscale numpy array (H x W, uint8).
        model, device: Pre-loaded model (uses cached if None).

    Returns:
        Predicted string, e.g. "0.09", "1,7645", "964".
        Returns "" if no characters are found.
    """
    if model is None or device is None:
        model, device = load_digit_cnn()

    chars = segment_crop(img_gray)   # no expected_count — best-effort
    if not chars:
        return ""

    x = torch.tensor(
        np.stack(chars)[:, np.newaxis], dtype=torch.float32
    ).to(device)                     # (N, 1, 32, 32)

    with torch.no_grad():
        indices = model(x).argmax(dim=1).cpu().tolist()

    return "".join(IDX_TO_CHAR[i] for i in indices)


def predict_image(img_path: str, model=None, device=None) -> str:
    """Load a crop from disk and predict its value."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    return predict_crop(img, model=model, device=device)


# ---------------------------------------------------------------------------
# CLI: quick test on one or more images
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m model.digit_predict <image> [image ...]")
        sys.exit(1)

    m, dev = load_digit_cnn()
    for path in sys.argv[1:]:
        pred = predict_image(path, model=m, device=dev)
        print(f"{Path(path).name:<50}  {pred}")
