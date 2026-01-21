"""
tree_model.py

Model factory for tree crown detection.

Supports:
- Faster R-CNN (torchvision) with ResNet-FPN backbones: resnet50, resnet101, resnet152
- DETR (HuggingFace transformers) with ResNet-50 backbone (optional)

Multi-channel input:
- Works with 3, 4, 5 (or N>=3) channel tensors
- Conv1 weights:
    - First 3 channels copy pretrained RGB weights
    - Extra channels are initialized as the mean of RGB weights (stable default)

Notes:
- Faster R-CNN uses GeneralizedRCNNTransform for normalization.
  You SHOULD pass channel mean/std that matches your dataset preprocessing.

Torchvision Models:
https://pytorch.org/vision/stable/models.html#object-detection

Original Papers:
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- DETR: https://arxiv.org/abs/2005.12872
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

try:
    from transformers import DetrForObjectDetection
except Exception:  # transformers optional
    DetrForObjectDetection = None


BackboneName = Literal["resnet50", "resnet101", "resnet152"]
DetectorName = Literal["fasterrcnn", "detr"]


@dataclass(frozen=True)
class NormalizeConfig:
    mean: List[float]
    std: List[float]
    min_size: int = 800
    max_size: int = 1333


# ---------------------------------------------------------------------
# Conv1 expansion (moved from tree_train.py)
# ---------------------------------------------------------------------
def modify_resnet_input_channels(model: FasterRCNN, in_channels: int = 5) -> FasterRCNN:
    """
    Backwards-compatible helper (used in your older training code):
    modifies torchvision FasterRCNN backbone conv1 to accept in_channels.

    Behavior:
    - copy pretrained RGB weights into first 3 channels
    - init extra channels with mean of RGB weights
    """
    if in_channels < 3:
        raise ValueError("in_channels must be >= 3")

    old_conv: nn.Conv2d = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight[:, :3]
        if in_channels > 3:
            mean_rgb = old_conv.weight[:, :3].mean(dim=1, keepdim=True)  # [out,1,k,k]
            new_conv.weight[:, 3:] = mean_rgb.repeat(1, in_channels - 3, 1, 1)

    model.backbone.body.conv1 = new_conv
    return model


def _expand_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Create a new conv with in_channels, copying pretrained weights:
    - RGB copied to first 3 channels
    - extra channels = mean over RGB weights
    """
    if in_channels < 3:
        raise ValueError("in_channels must be >= 3")

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight[:, :3]
        if in_channels > 3:
            mean_rgb = conv.weight[:, :3].mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean_rgb.repeat(1, in_channels - 3, 1, 1)

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


def _default_norm(in_channels: int) -> NormalizeConfig:
    # matches your tree_train.py defaults
    return NormalizeConfig(
        mean=[0.5] * in_channels,
        std=[0.25] * in_channels,
        min_size=800,
        max_size=1333,
    )

# ---------------------------------------------------------------------
# Faster R-CNN (canonical factory)
# ---------------------------------------------------------------------
def get_faster_rcnn(
    *,
    num_classes: int,
    in_channels: int = 5,
    backbone_name: BackboneName = "resnet50",
    pretrained_backbone: bool = True,
    norm: Optional[NormalizeConfig] = None,
) -> FasterRCNN:
    """
    Build Faster R-CNN with ResNet-FPN backbone.

    IMPORTANT: torchvision FasterRCNN expects num_classes INCLUDING background.
    Typical: num_classes=2 for {background, tree}.
    """
    backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained_backbone)

    # Expand conv1 on ResNet backbone (cleaner than patching after model creation)
    backbone.body.conv1 = _expand_conv_in_channels(backbone.body.conv1, in_channels)

    model = FasterRCNN(backbone, num_classes=num_classes)

    if norm is None:
        norm = _default_norm(in_channels)

    model.transform = GeneralizedRCNNTransform(
        min_size=norm.min_size,
        max_size=norm.max_size,
        image_mean=norm.mean,
        image_std=norm.std,
    )
    return model


# ---------------------------------------------------------------------
# Backwards-compatible alias (moved from tree_train.py)
# ---------------------------------------------------------------------
def get_faster_rcnn_5ch(
    num_classes: int,
    in_channels: int = 5,
    backbone_name: BackboneName = "resnet50",
    pretrained_backbone: bool = True,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    min_size: int = 800,
    max_size: int = 1333,
) -> FasterRCNN:
    """
    Compatibility wrapper for your old training script.
    Prefer get_faster_rcnn() in new code.
    """
    norm = NormalizeConfig(
        mean=image_mean if image_mean is not None else [0.5] * in_channels,
        std=image_std if image_std is not None else [0.25] * in_channels,
        min_size=min_size,
        max_size=max_size,
    )
    return get_faster_rcnn(
        num_classes=num_classes,
        in_channels=in_channels,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        norm=norm,
    )


# ---------------------------------------------------------------------
# DETR (optional)
# ---------------------------------------------------------------------
def get_detr(
    *,
    num_classes: int,
    in_channels: int = 5,
    pretrained_name: str = "facebook/detr-resnet-50",
) -> nn.Module:
    """
    Build DETR from transformers with expanded input channels.

    For HF DETR: num_classes is number of object classes (not including 'no object'),
    but HF heads manage background internally. We set the classifier to num_classes.
    """
    if DetrForObjectDetection is None:
        raise ImportError("transformers is not installed; cannot build DETR.")

    model = DetrForObjectDetection.from_pretrained(pretrained_name)

    old_conv = model.model.backbone.conv1
    model.model.backbone.conv1 = _expand_conv_in_channels(old_conv, in_channels)

    model.class_labels_classifier = nn.Linear(
        model.class_labels_classifier.in_features,
        num_classes,
    )
    return model


# ---------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------
def build_detector(
    *,
    detector: DetectorName,
    num_classes: int,
    in_channels: int,
    backbone: BackboneName = "resnet50",
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    pretrained_backbone: bool = True,
) -> nn.Module:
    """
    Unified factory used by training code.
    """
    if detector == "fasterrcnn":
        norm = None
        if norm_mean is not None or norm_std is not None:
            if norm_mean is None or norm_std is None:
                raise ValueError("Provide both norm_mean and norm_std or neither.")
            if len(norm_mean) != in_channels or len(norm_std) != in_channels:
                raise ValueError("norm_mean/std must match in_channels.")
            norm = NormalizeConfig(mean=norm_mean, std=norm_std)
        return get_faster_rcnn(
            num_classes=num_classes,
            in_channels=in_channels,
            backbone_name=backbone,
            pretrained_backbone=pretrained_backbone,
            norm=norm,
        )

    if detector == "detr":
        return get_detr(num_classes=num_classes, in_channels=in_channels)

    raise ValueError(f"Unknown detector: {detector}")
