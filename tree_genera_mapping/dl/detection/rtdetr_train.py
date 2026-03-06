import argparse
import random
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import RTDETR, settings


# -----------------------------
# 1) Multi-band-safe photometrics (NIR + Height)
# -----------------------------
def multiband_photometrics(trainer):
    """
    Lightweight photometric augmentations for multi-band data.
    Assumes:
      - channel 0..2: RGB in [0,1]
      - channel 3: NIR in [0,1]
      - channel 4: Height in [0,1] (optional)
    """
    if not trainer.training:
        return

    imgs = trainer.batch.get("img", None)  # float32 [B, C, H, W]
    if imgs is None or imgs.ndim != 4 or imgs.shape[1] not in (4, 5):
        return

    B, C, H, W = imgs.shape
    for b in range(B):
        # --- NIR (index 3) ---
        if random.random() < 0.6:
            gain = 0.85 + 0.30 * random.random()
            imgs[b, 3].mul_(gain).clamp_(0, 1)
        if random.random() < 0.3:
            std = 0.01 + 0.02 * random.random()
            imgs[b, 3].add_(torch.randn_like(imgs[b, 3]) * std).clamp_(0, 1)
        if random.random() < 0.03:
            imgs[b, 3].zero_()

        # --- Height (index 4) ---
        if C == 5:
            if random.random() < 0.6:
                scale = 0.95 + 0.10 * random.random()
                shift = (random.random() - 0.5) * 0.10
                imgs[b, 4].mul_(scale).add_(shift).clamp_(0, 1)
            if random.random() < 0.25:
                imgs[b, 4].add_(torch.randn_like(imgs[b, 4]) * 0.01).clamp_(0, 1)
            if random.random() < 0.03:
                imgs[b, 4].zero_()

    trainer.batch["img"] = imgs


# -----------------------------
# 2) Expand RT-DETR input conv: 3ch -> Nch
# -----------------------------
def _find_input_conv(mdl: nn.Module) -> Tuple[str, nn.Module, nn.Conv2d]:
    """
    Find the 'stem' Conv2d that consumes raw image channels.
    Heuristic: choose a conv with in_channels==3 and smallest depth in module name.
    Returns: (module_name, wrapper_or_conv_module, conv_module)
    """
    candidates = []
    for name, m in mdl.named_modules():
        if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
            conv = m.conv
            if conv.in_channels == 3:
                candidates.append((name, m, conv))
        elif isinstance(m, nn.Conv2d) and m.in_channels == 3:
            candidates.append((name, m, m))

    if not candidates:
        raise RuntimeError("No Conv2d with in_channels==3 found. Can't expand input.")

    # Pick the one closest to the root (fewest dots in name). Usually the stem.
    candidates.sort(key=lambda x: (x[0].count("."), x[0]))
    return candidates[0]


def expand_first_conv_multiband_rtdetr(
    model_obj,
    n_bands: int = 5,
    extra_strategies: Optional[List[str]] = None,
):
    """
    Expand RT-DETR first conv to accept n_bands input channels.
    - Copy pretrained RGB weights into channels 0..2
    - Init extra channels (3..n_bands-1) using strategies: 'mean'|'zeros'|'he'|'red'|'green'|'blue'
    """
    assert n_bands >= 3

    mdl = model_obj.model  # Ultralytics BaseModel
    name, wrapper, first = _find_input_conv(mdl)

    # Prepare strategies
    n_extra = n_bands - 3
    if n_extra > 0:
        if extra_strategies is None:
            extra_strategies = ["mean"] * n_extra
        elif isinstance(extra_strategies, str):
            extra_strategies = [extra_strategies] * n_extra
        else:
            if len(extra_strategies) != n_extra:
                raise ValueError(f"extra_strategies must have {n_extra} entries for {n_extra} extra channels.")

    # Build new conv
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

    with torch.no_grad():
        # copy RGB weights
        new_conv.weight[:, :3, :, :] = first.weight[:, :3, :, :]
        if first.bias is not None:
            new_conv.bias.copy_(first.bias)

        def fill(dst_ch: int, strat: str):
            s = strat.lower()
            if s in ("red", "r"):
                new_conv.weight[:, dst_ch, :, :] = first.weight[:, 0, :, :]
            elif s in ("green", "g"):
                new_conv.weight[:, dst_ch, :, :] = first.weight[:, 1, :, :]
            elif s in ("blue", "b"):
                new_conv.weight[:, dst_ch, :, :] = first.weight[:, 2, :, :]
            elif s == "mean":
                new_conv.weight[:, dst_ch, :, :] = first.weight[:, :3, :, :].mean(1)
            elif s == "zeros":
                new_conv.weight[:, dst_ch, :, :].zero_()
            elif s in ("he", "kaiming", "rand"):
                nn.init.kaiming_normal_(
                    new_conv.weight[:, dst_ch:dst_ch + 1, :, :],
                    mode="fan_out",
                    nonlinearity="relu",
                )
            else:
                new_conv.weight[:, dst_ch, :, :] = first.weight[:, :3, :, :].mean(1)

        for i, strat in enumerate(extra_strategies or []):
            fill(3 + i, strat)

    # Replace in model
    if hasattr(wrapper, "conv") and wrapper.conv is first:
        wrapper.conv = new_conv
    else:
        # replace by name on parent
        modules = dict(mdl.named_modules())
        parent_name = ".".join(name.split(".")[:-1])
        attr = name.split(".")[-1]
        parent = modules[parent_name] if parent_name else mdl
        setattr(parent, attr, new_conv)

    print(f"✅ RT-DETR input conv expanded: {name}  weight={tuple(new_conv.weight.shape)}")
    return model_obj


# -----------------------------
# 3) Train entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="dataset YAML")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="pretrained model path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-bands", type=int, default=5, choices=[3, 4, 5])
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--name", type=str, default="rtdetr_5ch")
    args = parser.parse_args()

    settings.update({"runs_dir": args.run_dir, "neptune": False})

    # Load RT-DETR (Ultralytics usage)
    # Docs example: model = RTDETR("rtdetr-l.pt"); model.train(data=..., epochs=..., imgsz=640)  :contentReference[oaicite:1]{index=1}
    model = RTDETR(args.model)
    model.info()

    # Expand input conv if multi-band
    if args.num_bands > 3:
        # Example init: NIR from mean-of-RGB, Height with He init
        extra = ["mean"] if args.num_bands == 4 else ["mean", "he"]
        model = expand_first_conv_multiband_rtdetr(model, n_bands=args.num_bands, extra_strategies=extra)

        # Add safe multi-band photometrics
        model.add_callback("on_preprocess_batch_end", multiband_photometrics)

    # IMPORTANT: many RGB-only color augmentations can misbehave on extra channels.
    # Easiest safe option: disable HSV jitter and rely on your multi-band callback.
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=5,
        # multi-band safety: disable HSV since it's RGB-specific
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        # keep geometric augs if you want (these are band-agnostic)
        fliplr=0.5,
        flipud=0.3,
        translate=0.1,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        # NOTE: mosaic/mixup/copy_paste support depends on Ultralytics version.
        # If you see shape/augment errors, comment them out first.
        mosaic=1.0,
        close_mosaic=30,
        mixup=0.10,
        copy_paste=0.30,
        name=args.name,
        deterministic=True,
    )

    print("Done:", results)


if __name__ == "__main__":
    main()