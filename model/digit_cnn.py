"""
Lightweight CNN for single handwritten character classification.

Classes: 0-9, dot (.), comma (,)  — 12 total.
Input:   (1, 32, 32) float32 tensor, ink=1 background=0.
Output:  logits over 12 classes.
"""

import torch
import torch.nn as nn

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]
NUM_CLASSES = len(CLASSES)
CHAR_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CLASSES)}


class DigitCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
