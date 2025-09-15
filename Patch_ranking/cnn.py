import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A compact 3-layer CNN suitable for CIFAR-sized images (32x32).
    Produces logits for multi-label prediction of patch indices.
    """

    def __init__(self, num_outputs: int, in_channels: int = 3) -> None:
        super().__init__()

        # 3 conv blocks (2–4 layers requested; we use 3)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 8x8 -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_outputs),  # logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

