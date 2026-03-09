import argparse
import random
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

    B, C, _, _ = imgs.shape
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
    Find the likely image stem Conv2d that consumes raw image channels.

    Heuristic:
      - prefer Conv2d with in_channels == 3
      - choose the one nearest the root in the module tree

    Returns:
      (module_name, wrapper_or_conv_module, conv_module)
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

    candidates.sort(key=lambda x: (x[0].count("."), x[0]))
    return candidates[0]


def expand_first_conv_multiband_rtdetr(
    model_obj,
    n_bands: int = 5,
    extra_strategies: Optional[List[str]] = None,
):
    """
    Expand RT-DETR first input conv to accept n_bands channels.

    - channels 0..2 copy pretrained RGB weights
    - extra channels initialized using:
      'mean' | 'zeros' | 'he' | 'red' | 'green' | 'blue'
    """
    assert n_bands >= 3, "n_bands must be >= 3"

    mdl = model_obj.model
    name, wrapper, first = _find_input_conv(mdl)

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
                    f"extra_strategies must have {n_extra} entries for {n_extra} extra channels."
                )

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

    if hasattr(wrapper, "conv") and wrapper.conv is first:
        wrapper.conv = new_conv
    else:
        modules = dict(mdl.named_modules())
        parent_name = ".".join(name.split(".")[:-1])
        attr = name.split(".")[-1]
        parent = modules[parent_name] if parent_name else mdl
        setattr(parent, attr, new_conv)

    print(f"✅ RT-DETR input conv expanded: {name}  weight={tuple(new_conv.weight.shape)}")
    return model_obj


# -----------------------------
# 3) Train
# -----------------------------
def train(conf):
    settings.update({"runs_dir": conf["run_dir"], "neptune": False})

    run_name = conf["run_name"]
    if not run_name:
        suffix = {3: "rgb", 4: "rgbi", 5: "rgbih"}[conf["num_bands"]]
        run_name = f"rtdetr_l_{suffix}_{conf['stage']}_{conf['img_size']}"

    # resume or start fresh
    if conf["resume"]:
        model = RTDETR(f"{conf['run_dir']}/detect/{run_name}/weights/last.pt")
    else:
        init_weights = conf["pretrained_weights"] if conf["pretrained_weights"] else conf["model"]
        model = RTDETR(init_weights)

        if conf["num_bands"] > 3:
            extra = ["mean"] if conf["num_bands"] == 4 else ["mean", "he"]
            model = expand_first_conv_multiband_rtdetr(
                model,
                n_bands=conf["num_bands"],
                extra_strategies=extra,
            )
            model.add_callback("on_preprocess_batch_end", multiband_photometrics)

    model.info()

    results = model.train(
        data=conf["data"],
        imgsz=conf["img_size"],
        epochs=conf["epochs"],
        patience=conf["patience"],
        batch=conf["batch"],
        device=conf["device"],
        workers=conf["workers"],
        cache=conf["cache"],
        resume=conf["resume"],

        optimizer=conf["optimizer"],
        lr0=conf["lr0"],
        lrf=conf["lrf"],
        weight_decay=conf["weight_decay"],
        cos_lr=conf["cos_lr"],
        warmup_epochs=conf["warmup_epochs"],

        # RGB-specific color jitter disabled for multiband safety
        hsv_h=conf["hsv_h"],
        hsv_s=conf["hsv_s"],
        hsv_v=conf["hsv_v"],

        fliplr=conf["fliplr"],
        flipud=conf["flipud"],
        translate=conf["translate"],
        scale=conf["scale"],
        degrees=conf["degrees"],
        shear=conf["shear"],
        perspective=conf["perspective"],

        mosaic=conf["mosaic"],
        close_mosaic=conf["close_mosaic"],
        mixup=conf["mixup"],
        copy_paste=conf["copy_paste"],

        project=f"{conf['run_dir']}/detect",
        name=run_name,
        exist_ok=False,
        deterministic=True,
    )

    print("✅ Done:", results)


# -----------------------------
# 4) CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="RT-DETR training")

    parser.add_argument("--stage", type=str, default="tree", choices=["visdrone", "tree"])
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--data", type=str, required=True, help="Dataset YAML")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Base model path")
    parser.add_argument("--pretrained-weights", type=str, default="", help="Optional weights to initialize from")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint of run-name")

    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument("--num-bands", type=int, default=3, choices=[3, 4, 5])

    parser.add_argument("--run-dir", type=str, required=True, help="Root run directory")

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    parser.add_argument("--warmup-epochs", type=float, default=5)

    # Keep HSV disabled by default for multiband safety
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.0)
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.0)
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.0)

    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)

    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)

    args = parser.parse_args()

    print("✅ Training RT-DETR")
    train(vars(args))


if __name__ == "__main__":
    main()