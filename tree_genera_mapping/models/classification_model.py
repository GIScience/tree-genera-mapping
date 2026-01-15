"""
ResNet-based classifier supporting 3, 4 and 5-channel input images,
plus a dual-branch model that fuses RGB+NIR and Height channels.

Supports ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152 from Torchvision.
Automatically adapts the first convolutional layer for 3, 4 and 5-channel inputs.

Torchvision ResNet models:
https://pytorch.org/vision/stable/models.html#classification

Original ResNet paper:
https://arxiv.org/abs/1512.03385
"""
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights
)

MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
}


def _build_resnet_model(model_name: str, num_classes: int, in_channels: int) -> nn.Module:
    """ Example of an architecture:
            Input [5-ch: RGB+NIR+Height]
               ↓
        Modified Conv2d (5→64) ← pretrained RGB weights + heuristic init
               ↓
        BatchNorm + ReLU + MaxPool
               ↓
        ResNet Layer1 (256)
               ↓
        ResNet Layer2 (512)
               ↓
        ResNet Layer3 (1024)
               ↓
        ResNet Layer4 (2048)
               ↓
        AdaptiveAvgPool
               ↓
        Fully Connected (2048→num_classes)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    model_fn, weights = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)

    # Modify first conv layer
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    with torch.no_grad():
        if in_channels == 3:
            model.conv1.weight.copy_(old_conv.weight)
        elif in_channels == 4:
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3] = old_conv.weight[:, 0]  # Fill 4th with R
        elif in_channels == 5:
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3] = old_conv.weight[:, 0]  # 4th channel
            model.conv1.weight[:, 4] = old_conv.weight[:, 1]  # 5th channel

    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet3ch(model_name="resnet50", num_classes=10):
    """
    Returns a ResNet classifier for 3-channel RGB input.
    """
    return _build_resnet_model(model_name, num_classes, in_channels=3)


def get_resnet4ch(model_name="resnet50", num_classes=10):
    """
    Returns a ResNet classifier for 4-channel input (e.g. RGB + NIR).
    """
    return _build_resnet_model(model_name, num_classes, in_channels=4)


def get_resnet5ch(model_name="resnet50", num_classes=10):
    """
    Returns a ResNet classifier for 5-channel input (e.g. RGB + NIR + Height).
    """
    return _build_resnet_model(model_name, num_classes, in_channels=5)
# --------------------------
# RGBI + Height Fusion Model
# --------------------------

class RGBIHeightFusionNet(nn.Module):
    """
    Dual-branch model for 4-channel RGB+NIR and 1-channel Height input.
    Uses pretrained ResNet for RGB+NIR and a small CNN stem for height.
                 ┌──────────────┐          ┌─────────────┐
                 │ RGB + NIR    │          │ Height Map  │
                 │ (4-channel)  │          │ (1-channel) │
                 └─────┬────────┘          └─────┬───────┘
         Pretrained ResNet Conv1 (4→64)   Small conv stem (1→64)
                 ↓                                 ↓
         BN → ReLU → MaxPool             Conv → BN → ReLU → MaxPool
                 ↓                                 ↓
           ResNet Layer1                        Conv → BN → ReLU
                 ↓                                 ↓
                Feature maps: (B, 256, H/4, W/4)   (B, 64, H/4, W/4)
                             └──────┬──────┘
                                    ↓
                         Concatenate (→ B, 320, H/4, W/4)
                                    ↓
                     1×1 Conv (320 → 256) — Fusion layer
                                    ↓
                 Shared ResNet Layers 2–4 (Pretrained)
                                    ↓
                        GlobalAvgPool → FC (2048 → num_classes)
    """
    def __init__(self, model_name="resnet101", num_classes=10, pretrained=True):
        super().__init__()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")

        model_fn, weights_enum = MODEL_REGISTRY[model_name]
        weights = weights_enum if pretrained else None
        resnet = model_fn(weights=weights)

        # RGB+NIR stem — manually initialize from pretrained RGB weights
        self.rgbn_stem = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            self.rgbn_stem.weight[:, :3] = resnet.conv1.weight  # RGB
            if pretrained:
                self.rgbn_stem.weight[:, 3] = resnet.conv1.weight[:, 0]  # NIR = R
            else:
                nn.init.kaiming_uniform_(self.rgbn_stem.weight[:, 3:4], a=0, mode='fan_in', nonlinearity='relu')

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        # Height branch
        self.height_stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Fusion projection
        self.fusion_proj = nn.Conv2d(320, 256, kernel_size=1)

        # Shared ResNet backbone
        self.shared_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        rgbn = x[:, :4]
        height = x[:, 4:]

        # RGBN branch
        x_rgbn = self.rgbn_stem(rgbn)
        x_rgbn = self.bn1(x_rgbn)
        x_rgbn = self.relu(x_rgbn)
        x_rgbn = self.maxpool(x_rgbn)
        x_rgbn = self.layer1(x_rgbn)

        # Height branch
        x_h = self.height_stem(height)

        # Fusion
        x = torch.cat([x_rgbn, x_h], dim=1)
        x = self.fusion_proj(x)
        x = self.shared_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def get_rgbih_fusion(model_name="resnet101", num_classes=10, pretrained=True):
    return RGBIHeightFusionNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)


if __name__=='__main__':
    from tree_genera_mapping.models.model_summary import summary

    print("\n🌿 Summary for RGBIHeight model")
    model1 = get_resnet5ch(model_name="resnet101", num_classes=5)
    summary(model1, in_channels=5, img_height=128, img_width=128)

    print("\n🌿 Summary for dual-branch RGBI + Height model")
    model2 = get_rgbih_fusion(model_name="resnet101", num_classes=10,pretrained=True)
    summary(model2, in_channels=5, img_height=128, img_width=128)




