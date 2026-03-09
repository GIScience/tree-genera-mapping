import argparse
import random
import torch
import torch.nn as nn
from ultralytics import YOLO, settings


# --------- helpers ------------
def multiband_photometrics(trainer):
    if not trainer.training:
        return
    imgs = trainer.batch["img"]
    if imgs.ndim != 4 or imgs.shape[1] not in (4, 5):
        return

    B, C, H, W = imgs.shape
    for b in range(B):
        if random.random() < 0.6:
            gain = 0.85 + 0.30 * random.random()
            imgs[b, 3].mul_(gain).clamp_(0, 1)
        if random.random() < 0.3:
            std = 0.01 + 0.02 * random.random()
            imgs[b, 3].add_(torch.randn_like(imgs[b, 3]) * std).clamp_(0, 1)
        if random.random() < 0.03:
            imgs[b, 3].zero_()

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


def expand_first_conv_multiband(
    y: YOLO,
    n_bands: int,
    extra_strategies=None,
    src_for_rgb: YOLO = None,
):
    assert n_bands >= 3, "n_bands must be >= 3"

    mdl = y.model
    first = None
    first_wrapper = None

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

    if src_for_rgb is not None and hasattr(src_for_rgb.model.model[0], "conv"):
        src = src_for_rgb.model.model[0].conv
    else:
        src = first
    if src.in_channels < 3:
        raise ValueError("Source conv must have >= 3 input channels to copy RGB weights.")

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

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = src.weight[:, :3, :, :]
        if first.bias is not None:
            new_conv.bias.copy_(first.bias)

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
                    mode="fan_out",
                    nonlinearity="relu",
                )
            else:
                new_conv.weight[:, dst_channel, :, :] = src.weight[:, :3, :, :].mean(1)

        for off, strat in enumerate(extra_strategies or []):
            _fill_channel(3 + off, strat)

    replaced = False
    if first_wrapper is not None and hasattr(first_wrapper, "conv") and first_wrapper.conv is first:
        first_wrapper.conv = new_conv
        replaced = True
    else:
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
        raise RuntimeError("Failed to replace the first Conv2d with the expanded one.")

    print(f"✅ First conv expanded to shape: {tuple(new_conv.weight.shape)}")
    return y


# ---------- Model Training -----------------
def train(conf):
    settings.update({"runs_dir": conf["run_dir"], "neptune": False})

    suffix = {3: "rgb", 4: "rgbi", 5: "rgbih"}.get(conf["num_bands"], "rgb")
    extra_strategies = {4: "red", 5: ["red", "he"]}.get(conf["num_bands"], None)

    run_name = conf["run_name"]
    if not run_name:
        run_name = f"y11{conf['model_size']}_{suffix}_{conf['stage']}_{conf['img_size']}"

    # choose initialization weights
    if conf["restart"]:
        model = YOLO(f"{conf['run_dir']}/detect/{run_name}/weights/last.pt")
    else:
        init_weights = conf["pretrained_weights"]
        if not init_weights:
            init_weights = conf["base_model"]

        model = YOLO(init_weights)

        if conf["num_bands"] > 3:
            model = expand_first_conv_multiband(
                model,
                n_bands=conf["num_bands"],
                extra_strategies=extra_strategies,
            )
            model.add_callback("on_preprocess_batch_end", multiband_photometrics)

    model.train(
        data=conf["data"],
        imgsz=conf["img_size"],
        epochs=conf["epochs"],
        patience=conf["patience"],
        batch=conf["batch"],
        device=conf["device"],
        resume=conf["restart"],

        optimizer=conf["optimizer"],
        lr0=conf["lr0"],
        lrf=conf["lrf"],
        weight_decay=conf["weight_decay"],
        cos_lr=conf["cos_lr"],
        warmup_epochs=conf["warmup_epochs"],

        box=conf["box"],
        cls=conf["cls"],
        dfl=conf["dfl"],
        label_smoothing=conf["label_smoothing"],

        mosaic=conf["mosaic"],
        close_mosaic=conf["close_mosaic"],
        mixup=conf["mixup"],
        copy_paste=conf["copy_paste"],
        fliplr=conf["fliplr"],
        flipud=conf["flipud"],
        translate=conf["translate"],
        scale=conf["scale"],
        shear=conf["shear"],
        degrees=conf["degrees"],
        perspective=conf["perspective"],
        hsv_h=conf["hsv_h"],
        hsv_s=conf["hsv_s"],
        hsv_v=conf["hsv_v"],
        erasing=conf["erasing"],

        cache=conf["cache"],
        workers=conf["workers"],
        deterministic=True,
        project=f"{conf['run_dir']}/detect",
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 training")

    parser.add_argument("--model-size", type=str, default="l", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--stage", type=str, default="tree", choices=["visdrone", "tree"])
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--num-bands", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)

    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--pretrained-weights", type=str, default="")
    parser.add_argument("--restart", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False)

    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    parser.add_argument("--warmup-epochs", type=float, default=5)

    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.7)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4)
    parser.add_argument("--erasing", type=float, default=0.0)

    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument("--workers", type=int, default=6)

    args = parser.parse_args()

    # if base-model not given, derive from size
    if not args.base_model:
        args.base_model = f"yolo11{args.model_size}.pt"

    print("✅ Training Model")
    train(vars(args))