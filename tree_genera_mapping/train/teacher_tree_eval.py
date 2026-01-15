import os, json, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from tree_genera_mapping.train.teacher_tree_train import (
    FiveBandImageDataset, autodetect_tabular_cols,
    make_resnet_5ch, MultiModalResNet, ID_TO_CLASS, NUM_CLASSES
)
import argparse

def plot_cm(cm, classes, out_png):
    cm = np.asarray(cm, dtype=np.float64)
    row = cm / cm.sum(axis=1, keepdims=True)
    row[np.isnan(row)] = 0
    fig, ax = plt.subplots(figsize=(9,9))
    im = ax.imshow(row, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i, f"{int(cm[i,j])}\n{row[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if row[i,j] > 0.6 else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ndvi_csv", required=True)
    ap.add_argument("--experiment", choices=["image_only","multimodal"], default="image_only")
    ap.add_argument("--backbone", choices=["resnet34","resnet50","resnet101"], default="resnet50")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_path = out_dir / f"{args.experiment}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", ID_TO_CLASS)
    tabular_cols = ckpt.get("tabular_cols", [])

    # Build datasets (VAL split only)
    # expects your folder structure where split is encoded in path
    img_dir = Path(args.images_dir)
    img_records = []
    for p in img_dir.rglob("*.tif"):
        split = "val" if "val" in p.parts else ("train" if "train" in p.parts else "unknown")
        if split != "val": continue
        class_name = p.parts[-2]
        img_records.append({"image_path": str(p), "class_name": class_name, "split": split})
    df = pd.DataFrame(img_records)
    df["class_id"] = df["class_name"].map({v:k for k,v in classes.items()}).astype(int)

    ndvi_df = pd.read_csv(args.ndvi_csv)
    # keep only columns that exist
    if "tree_id" in df.columns and "tree_id" in ndvi_df.columns:
        merged = df.merge(ndvi_df, on=["tree_id","class_name"], how="left")
    else:
        merged = df  # image-only

    from tree_genera_mapping.train.teacher_tree_train import ResizeCH, GeometricAugmentCH, PhotometricAugmentCH, NormalizeCH, read_5band_tiff
    # Reuse dataset class from training file
    val_set = FiveBandImageDataset(merged, img_size=args.img_size, augment=False,
                                   use_tabular=(args.experiment=="multimodal"),
                                   tabular_cols=tabular_cols)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Build model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.experiment == "multimodal":
        model = MultiModalResNet(args.backbone, len(classes), tabular_dim=len(tabular_cols))
    else:
        model = make_resnet_5ch(args.backbone, len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    y_true=[]; y_pred=[]
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in val_loader:
            if len(batch)==3:
                xi, xt, y = [b.to(device, non_blocking=True) for b in batch]
                out = model(xi, xt)
            else:
                xi, y = [b.to(device, non_blocking=True) for b in batch]
                out = model(xi)
            y_true.append(y.cpu().numpy())
            y_pred.append(out.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    print("Classification report:")
    print(pd.DataFrame.from_dict(
        classification_report(y_true, y_pred,
                              target_names=[classes[i] for i in range(len(classes))],
                              zero_division=0, output_dict=True)
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_cm(cm, [classes[i] for i in range(len(classes))], out_png=str(out_dir / "confusion_best_val.png"))
    print(f"Saved: {out_dir/'confusion_best_val.png'}")

if __name__ == "__main__":
    main()