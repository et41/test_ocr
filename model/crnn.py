"""Phase 3: CNN + BiLSTM + CTC model for handwritten numeric recognition."""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN model for sequence recognition of handwritten numeric fields.

    Architecture:
        Input: (batch, 1, 32, W) grayscale image
        → CNN feature extractor (Conv + BN + ReLU + MaxPool)
        → Reshape to sequence (W', features)
        → Bidirectional LSTM (2 layers, 256 hidden)
        → Linear projection → num_classes
    """

    def __init__(self, num_classes: int = 13, lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1: 1 → 64 channels
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16

            # Block 2: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8

            # Block 3: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 256 → 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8→4, width unchanged

            # Block 5: 256 → 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: 512 → 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4→2, width unchanged

            # Block 7: collapse height to 1
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Output: (batch, 512, 1, W')
        )

        # BiLSTM sequence model
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0,
        )

        # Output projection
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 32, W)

        Returns:
            Log probabilities of shape (T, batch, num_classes) for CTC loss
        """
        # CNN: (B, 1, 32, W) → (B, 512, 1, W')
        conv = self.cnn(x)

        # Reshape: (B, 512, 1, W') → (B, W', 512)
        b, c, h, w = conv.shape
        conv = conv.squeeze(2)  # (B, 512, W')
        conv = conv.permute(0, 2, 1)  # (B, W', 512)

        # LSTM: (B, W', 512) → (B, W', hidden*2)
        lstm_out, _ = self.lstm(conv)

        # Linear: (B, W', hidden*2) → (B, W', num_classes)
        output = self.fc(lstm_out)

        # CTC expects (T, B, C)
        output = output.permute(1, 0, 2)

        # Log softmax for CTC loss
        return torch.nn.functional.log_softmax(output, dim=2)
