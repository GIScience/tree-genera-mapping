import torch
import torch.nn as nn
from ultralytics import YOLO, settings
import random
import argparse
import numpy as np


# --------- helpers ------------
def multiband_photometrics(trainer):
    """
    Apply lightweight photometric augmentations for multi-band data (RGB+NIR+Height).
    Supports:
      - 4 channels: RGB + NIR
      - 5 channels: RGB + NIR + Height
    """
    if not trainer.training:
        return
    imgs = trainer.batch["img"]  # float32 [B, C, H, W] in [0,1]
    if imgs.ndim != 4 or imgs.shape[1] not in (4, 5):
        return

    B, C, H, W = imgs.shape
    for b in range(B):
        # --- NIR channel (always index 3) ---
        if random.random() < 0.6:
            gain = 0.85 + 0.30 * random.random()
            imgs[b, 3].mul_(gain).clamp_(0, 1)
        if random.random() < 0.3:
            std = 0.01 + 0.02 * random.random()
            imgs[b, 3].add_(torch.randn_like(imgs[b, 3]) * std).clamp_(0, 1)
        if random.random() < 0.03:  # rare NIR dropout
            imgs[b, 3].zero_()

        # --- Height channel (optional, index 4) ---
        if C == 5:
            if random.random() < 0.6:
                scale = 0.95 + 0.10 * random.random()
                shift = (random.random() - 0.5) * 0.10
                imgs[b, 4].mul_(scale).add_(shift).clamp_(0, 1)
            if random.random() < 0.25:
                imgs[b, 4].add_(torch.randn_like(imgs[b, 4]) * 0.01).clamp_(0, 1)
            if random.random() < 0.03:  # rare Height dropout
                imgs[b, 4].zero_()

    trainer.batch["img"] = imgs


def expand_first_conv_multiband(
        y: YOLO,
        n_bands: int,
        extra_strategies=None,
        src_for_rgb: YOLO = None,
):
    """
    Expand the very first Conv2d of a YOLO (Ultralytics) model to accept `n_bands` input channels.

    - Channels 0..2 (RGB) are *copied* from pretrained RGB weights.
    - Channels 3..(n_bands-1) are initialized per `extra_strategies`.

    Args:
        y:              a loaded YOLO model (e.g., YOLO('yolo11l.pt') or YOLO('runs/.../best.pt')).
        n_bands:        total input channels you want (>=3). Examples: 4 (RGB+NIR), 5 (RGB+NIR+Height).
        extra_strategies:
            How to init *each* extra channel (index 3..n_bands-1). You can pass:
              - a single string (applied to all extra channels), or
              - a list/tuple of strings, one per extra channel.
            Valid strings: 'red'|'green'|'blue'|'mean'|'zeros'|'he' (aka kaiming).
            Example for 5 bands (RGB + NIR + Height): ['red', 'he']
        src_for_rgb:    optional YOLO model to take RGB filters from (defaults to `y` itself).

    Returns:
        The same YOLO object `y` with first conv expanded and weights initialized.
    """
    assert n_bands >= 3, "n_bands must be >= 3"

    # ---- Locate first conv (Ultralytics Conv wrapper has attribute `.conv`) ----
    mdl = y.model  # BaseModel
    first = None
    first_wrapper = None

    # Prefer the top-most wrapper with .conv
    for m in mdl.model.modules():
        if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
            first_wrapper = m
            first = m.conv
            break
        if isinstance(m, nn.Conv2d) and first is None:
            first = m
            break
    if first is None:
        raise RuntimeError("Could not find first Conv2d in the YOLO model.")

    # ---- Choose source conv for RGB weights ----
    if src_for_rgb is not None and hasattr(src_for_rgb.model.model[0], "conv"):
        src = src_for_rgb.model.model[0].conv
    else:
        src = first  # use current model's first conv
    if src.in_channels < 3:
        raise ValueError("Source conv must have >= 3 input channels to copy RGB weights.")

    # ---- Prepare new conv with n_bands ----
    new_conv = nn.Conv2d(
        in_channels=n_bands,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=(first.bias is not None),
        padding_mode=first.padding_mode,
    ).to(first.weight.device, dtype=first.weight.dtype)

    # Normalize strategies input to a list matching the number of extra bands
    n_extra = n_bands - 3
    if n_extra > 0:
        if extra_strategies is None:
            extra_strategies = ["mean"] * n_extra
        elif isinstance(extra_strategies, str):
            extra_strategies = [extra_strategies] * n_extra
        else:
            extra_strategies = list(extra_strategies)
            if len(extra_strategies) != n_extra:
                raise ValueError(
                    f"extra_strategies length ({len(extra_strategies)}) must match extra channels ({n_extra})."
                )

    # ---- Fill weights ----
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = src.weight[:, :3, :, :]
        if first.bias is not None:
            new_conv.bias.copy_(first.bias)

        # Helper to init one channel by strategy
        def _fill_channel(dst_channel: int, strategy: str):
            s = strategy.lower()
            if s in ("red", "r"):
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, 0, :, :]
            elif s in ("green", "g"):
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, 1, :, :]
            elif s in ("blue", "b"):
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, 2, :, :]
            elif s == "mean":
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, :3, :, :].mean(1)
            elif s == "zeros":
                new_conv.weight[:, dst_channel, :, :].zero_()
            elif s in ("he", "kaiming", "rand"):
                nn.init.kaiming_normal_(
                    new_conv.weight[:, dst_channel:dst_channel + 1, :, :],
                    mode="fan_out", nonlinearity="relu"
                )
            else:
                # Fallback: mean of RGB
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, :3, :, :].mean(1)

        # Fill extra channels (3..n_bands-1)
        for off, strat in enumerate(extra_strategies or []):
            _fill_channel(3 + off, strat)

    # ---- Swap back into the model (robust replacement) ----
    replaced = False
    if first_wrapper is not None and hasattr(first_wrapper, "conv") and first_wrapper.conv is first:
        first_wrapper.conv = new_conv
        replaced = True
    else:
        # Try replacing inside the top-level sequential
        for i, block in enumerate(mdl.model):
            if hasattr(block, "conv") and block.conv is first:
                mdl.model[i].conv = new_conv
                replaced = True
                break
            if isinstance(block, nn.Conv2d) and block is first:
                mdl.model[i] = new_conv
                replaced = True
                break
    if not replaced:
        # Last-resort: search attributes and replace by identity check
        for name, module in mdl.named_modules():
            if module is first:
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = dict(mdl.named_modules())[parent_name] if parent_name else mdl
                setattr(parent, attr_name, new_conv)
                replaced = True
                break
    if not replaced:
        raise RuntimeError("Failed to replace the first Conv2d with the expanded one.")

    print(f"✅ First conv expanded to shape: {tuple(new_conv.weight.shape)}  "
          f"(out={new_conv.out_channels}, in={new_conv.in_channels})")
    return y


# ---------- Model Training -----------------
def train(conf):
    settings.update({"runs_dir": conf["run_dir"], "neptune": False})

    # Pick suffix based on num bands
    suffix = {3: "rgb", 4: "rgbi", 5: "rgbih"}.get(conf["num_bands"], "")
    extra_strategies = {4: "red", 5: ["red", "he"]}.get(conf["num_bands"], "")
    run_name = f"y11l_{suffix}_genus_cbam_{conf['img_size']}"
    if conf["restart"]:
        model = YOLO(f"{conf['run_dir']}/detect/{run_name}/weights/best.pt")
    else:
        model = YOLO(conf["base_model"]).load(f"{conf['run_dir']}/detect/y11l_visdrone_pretrain/weights/best.pt")
        if conf["num_bands"] > 3:
            model = expand_first_conv_multiband(model, n_bands=conf["num_bands"], extra_strategies=extra_strategies)
            model.add_callback("on_preprocess_batch_end", multiband_photometrics)

    model.train(
        data=conf["data"],
        imgsz=conf["img_size"],
        epochs=conf["epochs"],
        patience=20,
        batch=conf["batch"],
        device=conf["device"],
        resume=conf["restart"],
        optimizer="AdamW",
        lr0=5e-4, lrf=0.01,
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=5,
        cls=0.7,
        label_smoothing=0.05,
        mosaic=1.0, close_mosaic=30,
        mixup=0.10,
        copy_paste=0.30,
        fliplr=0.5, flipud=0.30,
        translate=0.10, scale=0.50, shear=0.0, degrees=0.0,
        perspective=0.0,
        hsv_h=0.005, hsv_s=0.6, hsv_v=0.3,
        erasing=0.15,
        cache="disk",
        workers=6,
        deterministic=True,
        name=run_name,
    )


# ------------ Main --------------

if __name__ == "__main__":
    # Config
    parser = argparse.ArgumentParser(description="YOLO11 Tree Genus Training")

    parser.add_argument("--num-bands", type=int, default=3,
                        help="Number of input bands (3=RGB, 4=RGB+NIR, 5=RGB+NIR+Height)")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--base-model", type=str, default="yolo11l.pt", help="Pretrained model path")
    parser.add_argument("--restart", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False,
                        help="Restart from checkpoint?")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory for YOLO runs")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", type=bool, default=False, help="Evaluate model")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence level")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU/NMS")

    args = parser.parse_args()

    print("✅ Training Model:")
    train(vars(args))

