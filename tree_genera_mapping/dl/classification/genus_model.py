"""
genus_model.py
Model factory for ResNet-based tree genus classification.

Supports:
- ResNet with 3/4/5-channel input (RGB / RGB+NIR / RGB+NIR+Height)
- Fusion model (RGB+NIR branch + Height branch)
- Multimodal model (image + tabular features)

Torchvision ResNet models:
https://pytorch.org/vision/stable/models.html#classification

Original ResNet paper:
https://arxiv.org/abs/1512.03385
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
)

MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
}


def _get_resnet(model_name: str, pretrained: bool) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported ResNet: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    fn, weights = MODEL_REGISTRY[model_name]
    return fn(weights=weights if pretrained else None)


def _replace_conv1_for_in_channels(
    resnet: nn.Module,
    in_channels: int,
    pretrained: bool,
    extra_channel_init: str = "copy",  # "copy" or "mean"
) -> None:
    """
    In-place replacement of ResNet conv1 for in_channels ∈ {3,4,5}.
    Copies pretrained RGB weights if pretrained=True.
    For extra channels:
      - "copy": 4th <- R, 5th <- G
      - "mean": extra <- mean(R,G,B)
    """
    if in_channels not in (3, 4, 5):
        raise ValueError("in_channels must be 3, 4, or 5")

    old = resnet.conv1
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            # copy RGB
            new.weight[:, :3] = old.weight

            if in_channels > 3:
                if extra_channel_init == "mean":
                    extra_w = old.weight[:, :3].mean(dim=1, keepdim=True)  # (64,1,7,7)
                    for c in range(3, in_channels):
                        new.weight[:, c : c + 1] = extra_w
                elif extra_channel_init == "copy":
                    new.weight[:, 3] = old.weight[:, 0]  # 4th <- R (e.g., NIR)
                    if in_channels == 5:
                        new.weight[:, 4] = old.weight[:, 1]  # 5th <- G (e.g., Height)
                else:
                    raise ValueError("extra_channel_init must be 'copy' or 'mean'")
    else:
        # not pretrained: default init is fine (Kaiming uniform by PyTorch)
        pass

    resnet.conv1 = new


def build_resnet_classifier(
    model_name: str,
    num_classes: int,
    in_channels: int,
    pretrained: bool = True,
    extra_channel_init: str = "copy",
) -> nn.Module:
    """
    Build a ResNet classifier adapted for in_channels ∈ {3,4,5}.
    """
    base = _get_resnet(model_name, pretrained=pretrained)
    _replace_conv1_for_in_channels(
        base,
        in_channels=in_channels,
        pretrained=pretrained,
        extra_channel_init=extra_channel_init,
    )
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base


def get_resnet3ch(model_name: str = "resnet50", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    return build_resnet_classifier(model_name, num_classes, in_channels=3, pretrained=pretrained)


def get_resnet4ch(model_name: str = "resnet50", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    return build_resnet_classifier(model_name, num_classes, in_channels=4, pretrained=pretrained)


def get_resnet5ch(model_name: str = "resnet50", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    return build_resnet_classifier(model_name, num_classes, in_channels=5, pretrained=pretrained)


class RGBIHeightFusionNet(nn.Module):
    """
    Dual-branch model:
    - Branch A: RGB+NIR (4ch) -> ResNet stem+layer1
    - Branch B: Height (1ch)  -> small CNN stem
    - Fuse -> ResNet layers2-4 -> pool -> fc
    """

    def __init__(self, model_name: str = "resnet101", num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        resnet = _get_resnet(model_name, pretrained=pretrained)

        # RGBN stem (4ch)
        self.rgbn_stem = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.rgbn_stem.weight[:, :3] = resnet.conv1.weight  # RGB
                self.rgbn_stem.weight[:, 3] = resnet.conv1.weight[:, 0]  # NIR <- R

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        # Height branch (1ch)
        self.height_stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Fuse: layer1 output = 256, height stem output = 64
        self.fusion_proj = nn.Conv2d(256 + 64, 256, kernel_size=1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgbn = x[:, :4]      # (B,4,H,W)
        height = x[:, 4:5]   # (B,1,H,W)

        # RGBN path
        a = self.rgbn_stem(rgbn)
        a = self.bn1(a)
        a = self.relu(a)
        a = self.maxpool(a)
        a = self.layer1(a)  # (B,256,H/4,W/4)

        # Height path
        b = self.height_stem(height)  # (B,64,H/4,W/4)

        # Fuse
        z = torch.cat([a, b], dim=1)  # (B,320,...)
        z = self.fusion_proj(z)       # (B,256,...)

        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)

        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return self.fc(z)


def get_rgbih_fusion(model_name: str = "resnet101", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    return RGBIHeightFusionNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)


# ------------------------------
# Multimodal (image + tabular)
# ------------------------------
class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalResNet(nn.Module):
    """
    Image + tabular fusion model for tree genus classification.

    Image input:
        x_img: [B, 5, H, W]  (RGB + NIR + Height)
    Tabular input:
        x_tab: [B, T]       (NDVI stats, canopy metrics, etc.)
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        tabular_dim: int,
        tab_hidden: int = 256,
        fused_hidden: int = 512,
        p: float = 0.2,
        pretrained: bool = True,
        extra_channel_init: str = "copy",
    ):
        super().__init__()

        base = _get_resnet(backbone, pretrained=pretrained)
        _replace_conv1_for_in_channels(
            base,
            in_channels=5,
            pretrained=pretrained,
            extra_channel_init=extra_channel_init,
        )

        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        self.tabular = TabularMLP(in_dim=tabular_dim, hidden=tab_hidden, out_dim=256, p=p)

        self.head = nn.Sequential(
            nn.Linear(feature_dim + 256, fused_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(fused_hidden, num_classes),
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(x_img)
        tab_feat = self.tabular(x_tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(fused)
