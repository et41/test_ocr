"""Phase 3: PyTorch Dataset for handwritten numeric field images."""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"

# Character set: digits, decimal point, minus sign, CTC blank
CHARS = "0123456789.-"
BLANK_IDX = len(CHARS)  # 12 = CTC blank token
NUM_CLASSES = len(CHARS) + 1  # 13 total

# Char ↔ index mappings
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

IMG_HEIGHT = 32  # Fixed height; width is variable


def encode_label(text: str) -> list[int]:
    """Convert text string to list of character indices."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_label(indices: list[int]) -> str:
    """Convert index sequence back to text, collapsing CTC blanks and repeats."""
    result = []
    prev = -1
    for idx in indices:
        if idx == BLANK_IDX:
            prev = idx
            continue
        if idx != prev:
            if idx in IDX_TO_CHAR:
                result.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(result)


def resize_to_fixed_height(image: np.ndarray, target_height: int = IMG_HEIGHT) -> np.ndarray:
    """Resize image to fixed height, preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, target_height), dtype=np.uint8)
    scale = target_height / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized


class FieldDataset(Dataset):
    """Dataset of cropped field images with text labels.

    Loads image paths and labels from labels.csv.
    Each image is resized to a fixed height (32px) with variable width.
    """

    def __init__(self, csv_path: Path = LABELS_CSV, transform=None):
        self.samples: list[tuple[str, str, str]] = []  # (image_path, field_name, value)
        self.transform = transform

        if not csv_path.exists():
            raise FileNotFoundError(f"Labels file not found: {csv_path}")

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include samples with valid encodable values
                value = row["value"].strip()
                if value and all(c in CHAR_TO_IDX for c in value):
                    self.samples.append((row["image_path"], row["field_name"], value))

        if not self.samples:
            raise ValueError("No valid labeled samples found in CSV")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, field_name, value = self.samples[idx]

        # Load grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Return a blank image if file is missing
            image = np.zeros((IMG_HEIGHT, IMG_HEIGHT), dtype=np.uint8)

        image = resize_to_fixed_height(image, IMG_HEIGHT)

        # Normalize to [0, 1] and add channel dimension
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        if self.transform:
            tensor = self.transform(tensor)

        label = encode_label(value)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": tensor,
            "label": label_tensor,
            "label_length": len(label),
            "text": value,
            "field_name": field_name,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function to handle variable-width images and labels.

    Pads images to the maximum width in the batch.
    """
    max_width = max(item["image"].shape[2] for item in batch)

    images = []
    labels = []
    label_lengths = []
    texts = []
    field_names = []

    for item in batch:
        img = item["image"]
        # Pad width to max_width
        pad_w = max_width - img.shape[2]
        if pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w), value=0.0)
        images.append(img)
        labels.append(item["label"])
        label_lengths.append(item["label_length"])
        texts.append(item["text"])
        field_names.append(item["field_name"])

    return {
        "images": torch.stack(images),
        "labels": torch.cat(labels),
        "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
        "texts": texts,
        "field_names": field_names,
    }
