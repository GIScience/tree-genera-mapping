"""
Detection model builder supporting Faster R-CNN and DETR for 3, 4, and 5-channel input images.

Backbones:
- ResNet50, ResNet101, ResNet152 (with FPN for Faster R-CNN)
- ResNet50 for DETR

Torchvision Models:
https://pytorch.org/vision/stable/models.html#object-detection

Original Papers:
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- DETR: https://arxiv.org/abs/2005.12872
"""
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from transformers import DetrForObjectDetection


# -------------------- Model Registry --------------------
MODEL_REGISTRY = {
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
}

# -------------------- Modify Conv1 for multi-channel input --------------------
def modify_resnet_input_channels(model, in_channels):
    old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    with torch.no_grad():
        # Copy pretrained RGB weights
        new_conv.weight[:, :3] = old_conv.weight
        if in_channels > 3:
            # Extra channels replicate mean of first 3
            extra = in_channels - 3
            mean_extra = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean_extra.repeat(1, extra, 1, 1)
    model.backbone.body.conv1 = new_conv
    return model

# -------------------- Get Faster R-CNN --------------------
def get_faster_rcnn(num_classes=2, in_channels=5, backbone_name="resnet50"):
    if backbone_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    backbone = resnet_fpn_backbone(MODEL_REGISTRY[backbone_name], pretrained=True)

    model = FasterRCNN(backbone, num_classes=num_classes)

    model = modify_resnet_input_channels(model, in_channels=in_channels)

    model.transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.5] * in_channels,
        image_std=[0.25] * in_channels
    )

    return model
# -------------------- Get DETR --------------------
def get_detr_5ch(num_classes=2):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Modify the first convolution layer in the ResNet backbone
    old_conv = model.model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=5, out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        if 5 > 3:
            extra = 5 - 3
            mean_extra = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean_extra.repeat(1, extra, 1, 1)

    model.model.backbone.conv1 = new_conv

    # Replace classification head
    model.class_labels_classifier = nn.Linear(model.class_labels_classifier.in_features, num_classes)

    return model
# -------------------- Main Function for Summary --------------------
if __name__ == "__main__":
    from torchinfo import summary
    import torch

    # Choose parameters
    num_classes = 2
    in_channels = 5  # Try 3, 4, or 5
    backbone = 'resnet101'  # 'resnet50', 'resnet101', 'resnet152'

    # Instantiate the model
    model = get_faster_rcnn(num_classes=num_classes, in_channels=in_channels, backbone_name=backbone)

    # Set model to eval mode for summary
    model.eval()

    # Generate dummy input for summary
    input_size = (1, in_channels, 640, 640)

    # Print summary (note: for detection models, this is mostly useful for backbone)
    summary(model.backbone.body, input_size=input_size, device="cpu")