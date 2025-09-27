# model.py
import torch
import torch.nn as nn

# Import specific model constructors and weight enums
from torchvision.models import (
    efficientnet_b4,
    efficientnet_b0,
    EfficientNet_B4_Weights,
    EfficientNet_B0_Weights,
)

class PestDetectionModel(nn.Module):
    def __init__(self, num_classes=102, backbone="b4", pretrained=True):
        """
        Args:
            num_classes (int): number of target classes
            backbone (str): "b4" (EfficientNet-B4) or "b0" (EfficientNet-B0)
            pretrained (bool): whether to load ImageNet pretrained weights
        """
        super(PestDetectionModel, self).__init__()

        self.backbone_name = backbone.lower()
        if self.backbone_name == "b4":
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
            # classifier in torchvision EfficientNet has structure: Sequential(Dropout, Linear)
            num_features = self.backbone.classifier[1].in_features
        elif self.backbone_name == "b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone. Choose 'b4' or 'b0'.")

        # Replace the classifier head with a custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(num_classes=102, device="cuda", backbone="b4", pretrained=True):
    """
    Helper to create & move model to device.

    Args:
      num_classes (int)
      device (str or torch.device)
      backbone (str): 'b4' or 'b0'
      pretrained (bool): load ImageNet weights via torchvision's weight enums
    """
    model = PestDetectionModel(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    return model.to(device)
