from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def load_labels_csv(labels_csv: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Load class mapping from CSV -> conf/genus_labels.csv .

    Supported column name variants:
      id / class_id / label_id
      class_name / genus / label

    Returns:
      id_to_class, class_to_id
    """
    df = pd.read_csv(labels_csv)

    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("class_id") or cols.get("label_id")
    name_col = cols.get("class_name") or cols.get("genus") or cols.get("label")

    if id_col is None or name_col is None:
        raise ValueError(
            f"{labels_csv} must contain an id column (id/class_id/label_id) "
            f"and a name column (class_name/genus/label). Found: {list(df.columns)}"
        )

    df = df[[id_col, name_col]].copy()
    df[id_col] = df[id_col].astype(int)
    df[name_col] = df[name_col].astype(str)

    id_to_class = dict(zip(df[id_col].tolist(), df[name_col].tolist()))
    class_to_id = {v: k for k, v in id_to_class.items()}

    return id_to_class, class_to_id
