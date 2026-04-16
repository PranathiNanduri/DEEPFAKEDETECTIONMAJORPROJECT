"""
Optional temporal branch skeleton for future video-based deepfake detection.
Not required for image-only execution.
"""

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
