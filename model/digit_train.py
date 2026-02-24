#!/usr/bin/env python3
"""
Train DigitCNN from labeled crop images.

For each (crop image, string label) in labels.csv:
  1. Segment the crop into individual characters
  2. If segment count == label length, add each (char_img, char) pair to dataset
  3. Train CNN and save best model to model/digit_cnn.pth

Run:
  source .venv/Scripts/activate
  python -m model.digit_train
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.char_segment import CHAR_SIZE, segment_crop
from model.digit_cnn import CHAR_TO_IDX, CLASSES, NUM_CLASSES, DigitCNN

LABELS_CSV = Path(__file__).resolve().parent.parent / "data" / "labels.csv"
MODEL_PATH = Path(__file__).resolve().parent / "digit_cnn.pth"
SEED = 42


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_char_dataset(labels_csv: Path) -> list[tuple[np.ndarray, str]]:
    """
    Returns (char_img float32, char_label) pairs from all labeled crops.
    Crops whose segment count doesn't match label length are skipped.
    """
    samples: list[tuple[np.ndarray, str]] = []
    skipped_unreadable = skipped_mismatch = skipped_bad_chars = 0

    with open(labels_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        label = row["value"].strip()

        # Only keep characters the model knows
        if not label or not all(c in CHAR_TO_IDX for c in label):
            skipped_bad_chars += 1
            continue

        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped_unreadable += 1
            continue

        chars = segment_crop(img, expected_count=len(label))
        if chars is None:
            skipped_mismatch += 1
            continue

        for char_img, char_label in zip(chars, label):
            samples.append((char_img, char_label))

    total = len(rows)
    used = total - skipped_unreadable - skipped_mismatch - skipped_bad_chars
    print(f"Crops used: {used}/{total}  "
          f"(skipped: {skipped_mismatch} seg-mismatch, "
          f"{skipped_unreadable} unreadable, "
          f"{skipped_bad_chars} bad-chars)")
    print(f"Character samples: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, str]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        char_img, char_label = self.samples[idx]
        # char_img: (32, 32) float32, ink=1, bg=0
        x = torch.tensor(char_img[np.newaxis], dtype=torch.float32)  # (1, 32, 32)

        if self.augment:
            angle = random.uniform(-12, 12)
            x = TF.rotate(x, angle, fill=0.0)
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            x = TF.affine(x, angle=0, translate=(tx, ty), scale=1.0, shear=0, fill=0.0)
            x = (x + torch.randn_like(x) * 0.04).clamp(0.0, 1.0)

        return x, CHAR_TO_IDX[char_label]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int = 80, batch_size: int = 32, lr: float = 1e-3):
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples = extract_char_dataset(LABELS_CSV)
    if not samples:
        print("No training samples — check that labels.csv paths exist.")
        return

    # Class distribution
    dist = Counter(s[1] for s in samples)
    print("Class distribution:", {c: dist.get(c, 0) for c in CLASSES})

    # Classes with zero samples get a dummy count of 1 to avoid div-by-zero
    class_counts = [max(dist.get(c, 0), 1) for c in CLASSES]
    missing = [c for c in CLASSES if dist.get(c, 0) == 0]
    if missing:
        print(f"WARNING: no training samples for classes: {missing}")

    # Train / val split
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_ds = CharDataset(samples[:split], augment=True)
    val_ds   = CharDataset(samples[split:], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Inverse-frequency class weights
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    weights = weights / weights.sum() * NUM_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}  |  train={len(train_ds)}  val={len(val_ds)}\n")

    model     = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)
        scheduler.step()

        # --- validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total   += len(y)
        val_acc = correct / total if total else 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.1%}  — saved to {MODEL_PATH}")

    # --- per-class accuracy on full val set ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    per_class_correct = Counter()
    per_class_total   = Counter()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for p, t in zip(preds.tolist(), y.tolist()):
                per_class_total[CLASSES[t]] += 1
                if p == t:
                    per_class_correct[CLASSES[t]] += 1

    print("\nPer-class val accuracy:")
    for c in CLASSES:
        n = per_class_total.get(c, 0)
        ok = per_class_correct.get(c, 0)
        bar = "#" * int(ok / max(n, 1) * 20)
        print(f"  '{c}'  {ok:3d}/{n:3d}  {bar}")


if __name__ == "__main__":
    train()
