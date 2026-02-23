"""Fine-tune TrOCR on labeled crop images.

Starts from microsoft/trocr-base-handwritten (pre-trained on IAM + SROIE)
and fine-tunes on our transformer test report crops.

Usage:
    python model/trocr_finetune.py
    python model/trocr_finetune.py --epochs 50 --lr 2e-5
"""

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dataset import augment_image

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
TROCR_FINETUNED = CHECKPOINT_DIR / "trocr_finetuned"
TROCR_BASE = "microsoft/trocr-base-handwritten"
LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"

ALLOWED_CHARS = set("0123456789.,-+")


class TrOCRDataset(Dataset):
    """Dataset of labeled crop images for TrOCR fine-tuning."""

    def __init__(self, samples: list[tuple[str, str]], processor: TrOCRProcessor, augment: bool = False):
        self.samples = samples
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, value = self.samples[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((32, 64), dtype=np.uint8)

        if self.augment:
            image = augment_image(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(image_rgb)
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize the label text
        labels = self.processor.tokenizer(
            text=value,
            return_tensors="pt",
            padding="max_length",
            max_length=20,
            truncation=True,
        ).input_ids.squeeze(0)

        # Replace padding tokens with -100 so they are ignored in the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels, "text": value}


def compute_accuracy(preds: list[str], targets: list[str]) -> float:
    """Exact-match accuracy after stripping to allowed chars."""
    correct = sum(
        "".join(c for c in p.strip() if c in ALLOWED_CHARS) == t
        for p, t in zip(preds, targets)
    )
    return correct / max(len(targets), 1)


def finetune(
    csv_path: str = str(LABELS_CSV),
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
    val_split: float = 0.2,
    patience: int = 8,
):
    """Fine-tune TrOCR on the labeled crop dataset.

    Args:
        csv_path:   Path to labels.csv.
        epochs:     Maximum training epochs.
        batch_size: Batch size.
        lr:         Learning rate.
        val_split:  Fraction of data held out for validation.
        patience:   Early stopping patience (epochs without val loss improvement).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all samples
    all_samples: list[tuple[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row["value"].strip()
            if value:
                all_samples.append((row["image_path"], value))

    if len(all_samples) < 2:
        raise ValueError(f"Need at least 2 labeled samples, found {len(all_samples)}")
    print(f"Total samples: {len(all_samples)}")

    # Load base model
    print(f"\nLoading base model: {TROCR_BASE}")
    processor = TrOCRProcessor.from_pretrained(TROCR_BASE)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_BASE)

    # Ensure decoder generation config is consistent with tokenizer
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model = model.to(device)

    # Train / val split
    random.shuffle(all_samples)
    val_size = max(1, int(len(all_samples) * val_split))
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    print(f"Train: {len(train_samples)}, Validation: {len(val_samples)}")

    train_set = TrOCRDataset(train_samples, processor, augment=True)
    val_set = TrOCRDataset(val_samples, processor, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print()
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds: list[str] = []
        all_targets: list[str] = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                generated_ids = model.generate(pixel_values, max_new_tokens=20)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                all_preds.extend(preds)
                all_targets.extend(batch["text"])

        val_loss /= len(val_loader)
        accuracy = compute_accuracy(all_preds, all_targets)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {accuracy:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_pretrained(str(TROCR_FINETUNED))
            processor.save_pretrained(str(TROCR_FINETUNED))
            print(f"  -> Saved best model to {TROCR_FINETUNED}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break

    print(f"\nFine-tuning complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {TROCR_FINETUNED}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on labeled crop images")
    parser.add_argument("--csv", type=str, default=str(LABELS_CSV), help="Path to labels.csv")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    args = parser.parse_args()

    finetune(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
