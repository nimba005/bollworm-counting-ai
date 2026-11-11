"""Shared helpers: seeding, config, ID parsing, WKT parsing, metric."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

CLASSES = ["abw", "pbw"]  # class index 0 = ABW (American bollworm), 1 = PBW (pink bollworm)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_stem(name: str) -> str:
    """'id_abc.jpg' -> 'id_abc'; works for bare IDs and paths too."""
    name = Path(str(name).strip()).name
    stem, ext = os.path.splitext(name)
    return stem if ext.lower() in IMG_EXTS else name


_ID_RE = re.compile(r"^(?P<img>.+?)_(?P<cls>abw|pbw)$", re.IGNORECASE)


def split_image_class_id(row_id: str) -> tuple[str, str]:
    """Split a submission/train row ID into (image_stem, class).

    Handles 'id_x_abw', 'ID_X_ABW', 'id_x.jpg_pbw', 'id_x_pbw.jpg'.
    """
    s = image_stem(row_id)  # drop trailing extension if present
    m = _ID_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse class suffix from ID: {row_id!r}")
    return image_stem(m.group("img")), m.group("cls").lower()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def wkt_to_xyxy(wkt: str) -> tuple[float, float, float, float] | None:
    """Bounding box (x1, y1, x2, y2) of a WKT POLYGON / MULTIPOLYGON string."""
    if not isinstance(wkt, str) or not wkt.strip():
        return None
    nums = [float(n) for n in _NUM_RE.findall(wkt)]
    if len(nums) < 4 or len(nums) % 2:
        return None
    xs, ys = nums[0::2], nums[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def competition_mae(y_true: dict, y_pred: dict) -> float:
    """Competition metric.

    y_true / y_pred: {image_stem: (abw_count, pbw_count)}.
    Per image error = |abw err| + |pbw err|; score = mean over images.
    (Zindi computes MAE over submission rows, which is exactly half of this and
    ranks identically. We report the per-row version below too.)
    """
    errs = []
    for k, (ta, tp) in y_true.items():
        pa, pp = y_pred.get(k, (0, 0))
        errs.append(abs(ta - pa) + abs(tp - pp))
    return float(np.mean(errs)) if errs else float("nan")


def row_mae(y_true: dict, y_pred: dict) -> float:
    """MAE over (image, class) rows -- what the Zindi leaderboard shows."""
    return competition_mae(y_true, y_pred) / 2.0
