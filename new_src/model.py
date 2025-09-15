import torch
import torch.nn as nn


class PatchSelectionCNN(nn.Module):
    """
    Lightweight CNN with up to 4 conv layers for multi-label prediction.
    Input: (B, 3, H, W)
    Output: (B, num_classes) with logits
    """

    def __init__(self, num_classes: int = 196, in_ch: int = 3) -> None:
        super().__init__()
        # 4 conv layers total (lightweight)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x  # logits for BCEWithLogitsLoss

