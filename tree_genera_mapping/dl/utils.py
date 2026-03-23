from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def load_labels_csv(path):
    import pandas as pd

    df = pd.read_csv(path)

    # directly use your schema
    if "fid" not in df.columns or "genus" not in df.columns:
        raise ValueError(
            f"{path} must contain columns 'fid' and 'genus'. Found: {list(df.columns)}"
        )

    df = df[["fid", "genus"]].copy()
    df = df.dropna()

    # enforce types
    df["fid"] = df["fid"].astype(int)
    df["genus"] = df["genus"].astype(str)

    # IMPORTANT: normalize names to match folder naming
    def norm(x):
        return x.strip().replace(" ", "_")

    df["genus_norm"] = df["genus"].map(norm)

    id_to_class = dict(zip(df["fid"], df["genus_norm"]))
    class_to_id = {v: k for k, v in id_to_class.items()}

    return id_to_class, class_to_id
    