import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture == "resnet18":
        model = resnet18(weights=None)

        # CIFAR-10 images 
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    raise ValueError(f"Unsupported architecture: {architecture}")