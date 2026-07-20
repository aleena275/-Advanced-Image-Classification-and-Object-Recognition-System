"""
model.py
---------
Defines two families of models used by the Image Classification &
Object Recognition system:

1. SimpleCNN      -> A LeNet-5 / AlexNet style CNN built from scratch.
2. TransferModel  -> A wrapper around torchvision pretrained models
                      (ResNet18 / VGG16 / EfficientNet-B0) that supports
                      both feature-extraction and fine-tuning modes.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------
# 1. CNN built from scratch (Phase 3 of the project workflow)
# ---------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A small, from-scratch CNN (AlexNet-style) suitable for datasets
    such as CIFAR-10 (10 classes, 32x32 images).
    Includes conv/pool blocks, batch-norm and dropout for regularization.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32 -> 16x16
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16 -> 8x8
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8x8 -> 4x4
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """Used by Grad-CAM to hook into the last convolutional layer."""
        return self.features[-4]  # last Conv2d before final pooling/dropout


# ---------------------------------------------------------------------
# 2. Transfer learning wrapper (Phase 4 of the project workflow)
# ---------------------------------------------------------------------
class TransferModel(nn.Module):
    """
    Wraps a torchvision pretrained backbone (resnet18 / vgg16 /
    efficientnet_b0) and replaces its classification head.

    mode="feature_extraction" -> freeze all backbone weights, train head only
    mode="fine_tune"          -> unfreeze the last few layers as well
    """

    SUPPORTED = ("resnet18", "vgg16", "efficientnet_b0")

    def __init__(self, backbone: str = "resnet18", num_classes: int = 10,
                 mode: str = "feature_extraction", pretrained: bool = True):
        super().__init__()
        if backbone not in self.SUPPORTED:
            raise ValueError(f"backbone must be one of {self.SUPPORTED}")

        self.backbone_name = backbone
        weights = "DEFAULT" if pretrained else None

        if backbone == "resnet18":
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
            self.backbone = net
            self.feature_layer = "layer4"

        elif backbone == "vgg16":
            net = models.vgg16(weights=weights)
            in_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(in_features, num_classes)
            self.backbone = net
            self.feature_layer = "features"

        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, num_classes)
            self.backbone = net
            self.feature_layer = "features"

        self._set_trainable_layers(mode)

    def _set_trainable_layers(self, mode: str):
        for param in self.backbone.parameters():
            param.requires_grad = False

        if mode == "feature_extraction":
            # Only the new classification head trains
            head = self._get_head()
            for param in head.parameters():
                param.requires_grad = True

        elif mode == "fine_tune":
            # Unfreeze head + last block for fine-tuning
            head = self._get_head()
            for param in head.parameters():
                param.requires_grad = True
            if self.backbone_name == "resnet18":
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            else:
                # unfreeze last few feature layers for vgg/efficientnet
                children = list(self.backbone.features.children())
                for layer in children[-3:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            raise ValueError("mode must be 'feature_extraction' or 'fine_tune'")

    def _get_head(self):
        if self.backbone_name == "resnet18":
            return self.backbone.fc
        elif self.backbone_name == "vgg16":
            return self.backbone.classifier[6]
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.classifier[1]

    def forward(self, x):
        return self.backbone(x)


def build_model(model_type: str = "transfer", **kwargs) -> nn.Module:
    """
    Factory function.
    model_type: "scratch" -> SimpleCNN
                "transfer" -> TransferModel
    """
    if model_type == "scratch":
        return SimpleCNN(**kwargs)
    elif model_type == "transfer":
        return TransferModel(**kwargs)
    raise ValueError("model_type must be 'scratch' or 'transfer'")
