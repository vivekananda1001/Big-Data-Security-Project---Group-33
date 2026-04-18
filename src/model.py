import torch
import torch.nn as nn
import torch.nn.functional as F


class FLCNN(nn.Module):
    """CNN matching Paper 2's architecture.
    4 conv layers (3x3 kernels), 2 FC layers, softmax output.
    Used for CIFAR-10, CIFAR-100, MNIST, FMNIST."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, img_size: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        flat_size = 128 * (img_size // 4) * (img_size // 4)
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Dataset -> model config mapping
MODEL_CONFIGS = {
    "mnist": {"in_channels": 1, "num_classes": 10, "img_size": 28},
    "fmnist": {"in_channels": 1, "num_classes": 10, "img_size": 28},
    "cifar10": {"in_channels": 3, "num_classes": 10, "img_size": 32},
    "cifar100": {"in_channels": 3, "num_classes": 100, "img_size": 32},
}


def get_model(dataset_name: str) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset_name]
    return FLCNN(**cfg)


def get_model_params_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)
