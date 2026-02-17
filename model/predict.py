"""Phase 3: Inference with the trained CRNN model."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.crnn import CRNN
from pipeline.dataset import BLANK_IDX, IDX_TO_CHAR, IMG_HEIGHT, NUM_CLASSES, resize_to_fixed_height

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"


def load_model(checkpoint_path: Path = CHECKPOINT_PATH, device: torch.device = None) -> tuple[CRNN, torch.device]:
    """Load the trained CRNN model from checkpoint.

    Returns:
        Tuple of (model, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def predict_image(image: np.ndarray, model: CRNN, device: torch.device) -> tuple[str, float]:
    """Run inference on a single grayscale crop image.

    Args:
        image: Grayscale numpy array (the cropped field).
        model: Loaded CRNN model.
        device: Torch device.

    Returns:
        Tuple of (decoded_text, confidence_score).
        Confidence is the average probability of the predicted characters.
    """
    # Preprocess: resize to fixed height
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = resize_to_fixed_height(image, IMG_HEIGHT)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        log_probs = model(tensor)  # (T, 1, C)

    # Greedy decode with confidence
    probs = torch.exp(log_probs[:, 0, :])  # (T, C)
    argmax = probs.argmax(dim=1)  # (T,)
    max_probs = probs.max(dim=1).values  # (T,)

    # Collapse CTC: remove blanks and consecutive duplicates
    decoded_chars = []
    char_confidences = []
    prev = -1
    for t in range(argmax.shape[0]):
        idx = argmax[t].item()
        conf = max_probs[t].item()
        if idx == BLANK_IDX:
            prev = idx
            continue
        if idx != prev:
            if idx in IDX_TO_CHAR:
                decoded_chars.append(IDX_TO_CHAR[idx])
                char_confidences.append(conf)
        prev = idx

    text = "".join(decoded_chars)
    confidence = float(np.mean(char_confidences)) if char_confidences else 0.0

    return text, confidence


def predict_file(image_path: str, checkpoint_path: str = str(CHECKPOINT_PATH)) -> tuple[str, float]:
    """Convenience function: predict from an image file path.

    Returns:
        Tuple of (decoded_text, confidence_score).
    """
    model, device = load_model(Path(checkpoint_path))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return predict_image(image, model, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CRNN inference on a crop image")
    parser.add_argument("image", type=str, help="Path to crop image")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Model checkpoint path")
    args = parser.parse_args()

    text, confidence = predict_file(args.image, args.checkpoint)
    print(f"Predicted: {text}")
    print(f"Confidence: {confidence:.2%}")
