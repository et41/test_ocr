"""Phase 3: Training script for the CRNN model."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.crnn import CRNN
from pipeline.dataset import (
    BLANK_IDX,
    LABELS_CSV,
    NUM_CLASSES,
    FieldDataset,
    collate_fn,
    decode_label,
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def compute_cer(predicted: str, target: str) -> float:
    """Compute Character Error Rate using edit distance."""
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    # Simple Levenshtein distance
    m, n = len(predicted), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if predicted[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n] / n


def greedy_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """CTC greedy decoding on a batch.

    Args:
        log_probs: (T, B, C) log probabilities

    Returns:
        List of decoded index sequences (one per batch item)
    """
    # (T, B) — best class at each timestep
    argmax = log_probs.argmax(dim=2)  # (T, B)
    batch_size = argmax.shape[1]
    results = []
    for b in range(batch_size):
        indices = argmax[:, b].tolist()
        results.append(indices)
    return results


def train(
    csv_path: str = str(LABELS_CSV),
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 15,
):
    """Train the CRNN model.

    Args:
        csv_path: Path to labels CSV.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        val_split: Fraction of data for validation.
        patience: Early stopping patience (epochs without improvement).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset — separate train (with augmentation) and val (without)
    full_dataset = FieldDataset(Path(csv_path), augment=False)
    print(f"Total samples: {len(full_dataset)}")

    # Train/val split by indices
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    all_indices = list(range(len(full_dataset)))
    import random as rng
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # Create separate datasets with/without augmentation
    train_set = FieldDataset(Path(csv_path), augment=True)
    val_set = FieldDataset(Path(csv_path), augment=False)

    train_subset = torch.utils.data.Subset(train_set, train_indices)
    val_subset = torch.utils.data.Subset(val_set, val_indices)
    print(f"Train: {train_size}, Validation: {val_size}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            log_probs = model(images)  # (T, B, C)
            T = log_probs.shape[0]
            input_lengths = torch.full((images.shape[0],), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total_cer = 0.0
        num_val = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                labels = batch["labels"].to(device)
                label_lengths = batch["label_lengths"].to(device)

                log_probs = model(images)
                T = log_probs.shape[0]
                input_lengths = torch.full((images.shape[0],), T, dtype=torch.long, device=device)

                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                decoded_batch = greedy_decode(log_probs)
                for decoded_indices, target_text in zip(decoded_batch, batch["texts"]):
                    pred_text = decode_label(decoded_indices)
                    total_cer += compute_cer(pred_text, target_text)
                    if pred_text == target_text:
                        correct += 1
                    num_val += 1

        val_loss /= len(val_loader)
        avg_cer = total_cer / max(num_val, 1)
        accuracy = correct / max(num_val, 1)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"CER: {avg_cer:.4f} | "
            f"Acc: {accuracy:.2%} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "cer": avg_cer,
                "accuracy": accuracy,
            }, checkpoint_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CRNN model on labeled field crops")
    parser.add_argument("--csv", type=str, default=str(LABELS_CSV), help="Path to labels.csv")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    train(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
