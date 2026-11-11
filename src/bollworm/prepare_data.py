"""Turn the raw Zindi download into a YOLO dataset.

Expected in data/raw (zips are extracted automatically):
  Train.csv              per-(image, class) worm counts
  Test.csv               test image IDs
  SampleSubmission.csv   submission format
  images_bboxes.csv      one row per box, WKT polygon geometry (name may vary: *bbox*.csv)
  images/ or images.zip  all train + test images

Outputs in data/processed:
  images/{train,val,test}/<stem>.jpg   resized, EXIF orientation baked in, metadata stripped
  labels/{train,val}/<stem>.txt        YOLO labels (class cx cy w h, normalised)
  dataset.yaml                         for ultralytics
  meta.csv                             stem, split, gt_abw, gt_pbw, n_boxes_abw, n_boxes_pbw
  prep_report.json                     orientation decision + sanity stats

Usage: python -m bollworm.prepare_data --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bollworm.utils import (
    CLASSES,
    IMG_EXTS,
    image_stem,
    load_config,
    resolve,
    set_seed,
    split_image_class_id,
    wkt_to_xyxy,
)

Image.MAX_IMAGE_PIXELS = None
EXIF_ORIENTATION = 0x0112


# --------------------------------------------------------------------------- files
def extract_zips(raw: Path) -> None:
    for z in sorted(raw.glob("*.zip")):
        marker = raw / f".{z.stem}.extracted"
        if marker.exists():
            continue
        print(f"[unzip] {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():  # guard against path traversal
                target = (raw / m.filename).resolve()
                if not str(target).startswith(str(raw.resolve())):
                    raise RuntimeError(f"Unsafe path in zip: {m.filename}")
            zf.extractall(raw)
        marker.touch()


def find_csv(raw: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        hits = sorted(p for p in raw.rglob(pat) if not p.name.startswith("._"))
        if hits:
            return hits[0]
    return None


def pick_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Could not find {what} column. Have: {list(df.columns)}")


def index_images(raw: Path) -> dict[str, Path]:
    idx = {}
    for p in raw.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"):
            idx.setdefault(image_stem(p.name), p)
    return idx


# --------------------------------------------------------------------------- labels
def load_train_counts(path: Path) -> dict[str, list[int]]:
    df = pd.read_csv(path)
    id_col = pick_col(df, ["image_id_worm", "Image_ID", "image_id", "ID"], "train id")
    n_col = pick_col(df, ["number_of_worms", "Target", "count", "label"], "train count")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rid, n in zip(df[id_col], df[n_col]):
        stem, cls = split_image_class_id(rid)
        counts[stem][CLASSES.index(cls)] = int(n) if pd.notna(n) else 0
    return dict(counts)


def load_boxes(path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    df = pd.read_csv(path)
    img_col = pick_col(df, ["image_id", "Image_ID", "image"], "bbox image")
    cls_col = pick_col(df, ["worm_type", "class", "label"], "bbox class")
    geo_col = pick_col(df, ["geometry", "bbox", "wkt"], "bbox geometry")
    boxes: dict[str, list] = defaultdict(list)
    bad = 0
    for img, cls, geo in zip(df[img_col], df[cls_col], df[geo_col]):
        stem = image_stem(img)
        boxes.setdefault(stem, [])  # image with no worms still recorded
        if not isinstance(cls, str) or cls.strip().lower() not in CLASSES:
            continue
        xyxy = wkt_to_xyxy(geo)
        if xyxy is None:
            bad += 1
            continue
        boxes[stem].append((CLASSES.index(cls.strip().lower()), *xyxy))
    if bad:
        print(f"[warn] {bad} boxes with unparseable geometry skipped")
    return dict(boxes)


# --------------------------------------------------------------------------- orientation
def raw_size_and_orientation(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        w, h = im.size
        try:
            o = int(im.getexif().get(EXIF_ORIENTATION, 1))
        except Exception:
            o = 1
    return w, h, o


def decide_orientation(boxes, img_index, sample: int = 3000, seed: int = 0) -> dict:
    """Are box coordinates in raw-pixel space or EXIF-rotated space?

    Only images whose EXIF orientation swaps width/height (5-8) and that have boxes
    can tell the two apart. We count which interpretation keeps the boxes inside the image.
    """
    rng = np.random.default_rng(seed)
    stems = [s for s, b in boxes.items() if b and s in img_index]
    rng.shuffle(stems)
    votes = {"raw": 0, "exif": 0, "both": 0, "neither": 0}
    for s in tqdm(stems[:sample], desc="orientation check"):
        w, h, o = raw_size_and_orientation(img_index[s])
        if o not in (5, 6, 7, 8):
            continue
        b = np.array([x[1:] for x in boxes[s]])
        mx, my = b[:, 2].max(), b[:, 3].max()
        tol = 1.01
        fit_raw = mx <= w * tol and my <= h * tol
        fit_exif = mx <= h * tol and my <= w * tol
        key = "both" if fit_raw and fit_exif else "raw" if fit_raw else "exif" if fit_exif else "neither"
        votes[key] += 1
    mode = "exif" if votes["exif"] > votes["raw"] else "raw"
    return {"mode": mode, "votes": votes}


# --------------------------------------------------------------------------- per-image worker
def process_one(args):
    stem, src, split, boxes, out_root, max_side, quality, mode = args
    dst_img = out_root / "images" / split / f"{stem}.jpg"
    dst_lbl = out_root / "labels" / split / f"{stem}.txt"
    if dst_img.exists() and (split == "test" or dst_lbl.exists()):
        return stem, None
    try:
        with Image.open(src) as im:
            if mode == "exif":
                im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            W, H = im.size
            scale = min(1.0, max_side / max(W, H))
            if scale < 1.0:
                im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
            W, H = im.size  # boxes were pre-scaled to this size in _prescale()
            im.save(dst_img, "JPEG", quality=quality)  # no exif written -> no double rotation later
    except Exception as e:  # corrupt image
        return stem, f"{type(e).__name__}: {e}"

    if split != "test":
        lines = []
        for cls, x1, y1, x2, y2 in boxes or []:
            x1, x2 = np.clip([x1, x2], 0, W)
            y1, y2 = np.clip([y1, y2], 0, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(
                f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
            )
        dst_lbl.write_text("\n".join(lines))
    return stem, None


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    P = cfg["prep"]
    raw, out = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])

    extract_zips(raw)
    train_csv = find_csv(raw, "Train.csv", "train.csv")
    test_csv = find_csv(raw, "Test.csv", "test.csv")
    sample_csv = find_csv(raw, "SampleSubmission.csv", "*ample*ubmission*.csv")
    bbox_csv = find_csv(raw, "images_bboxes.csv", "*bbox*.csv")
    for name, p in [("Train.csv", train_csv), ("Test.csv", test_csv), ("bbox csv", bbox_csv)]:
        if p is None:
            raise SystemExit(f"Missing {name} in {raw}. See README -> 'Get the data'.")
    print(f"[files] train={train_csv.name} test={test_csv.name} bbox={bbox_csv.name} sample={sample_csv and sample_csv.name}")

    img_index = index_images(raw)
    print(f"[images] {len(img_index)} image files found")

    counts = load_train_counts(train_csv)
    boxes = load_boxes(bbox_csv)
    test_df = pd.read_csv(test_csv)
    test_col = pick_col(test_df, ["image_id", "Image_ID", "ID"], "test id")
    test_stems = sorted({image_stem(x) for x in test_df[test_col]})

    train_stems = sorted(set(counts) | set(boxes))
    missing = [s for s in train_stems if s not in img_index]
    train_stems = [s for s in train_stems if s in img_index]
    missing_test = [s for s in test_stems if s not in img_index]
    print(f"[train] {len(train_stems)} images ({len(missing)} missing on disk); [test] {len(test_stems)} ({len(missing_test)} missing)")

    # Ground-truth counts: prefer Train.csv, fall back to box counts.
    box_counts = {s: [sum(1 for b in boxes.get(s, []) if b[0] == c) for c in range(2)] for s in train_stems}
    gt = {s: counts.get(s, box_counts[s]) for s in train_stems}
    mism = sum(1 for s in train_stems if s in counts and s in boxes and counts[s] != box_counts[s])
    print(f"[check] images where Train.csv count != number of boxes: {mism}")

    # Orientation
    if P["orientation"] == "auto":
        orient = decide_orientation(boxes, img_index, seed=cfg["seed"])
    else:
        orient = {"mode": P["orientation"], "votes": None}
    print(f"[orientation] mode={orient['mode']} votes={orient['votes']}")

    # Split, stratified on total count bucket
    tot = np.array([sum(gt[s]) for s in train_stems])
    bins = np.digitize(tot, [0.5, 2.5, 5.5, 10.5, 20.5, 50.5])
    tr, va = train_test_split(train_stems, test_size=P["val_fraction"], random_state=cfg["seed"], stratify=bins)
    split_of = {**{s: "train" for s in tr}, **{s: "val" for s in va}}

    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    jobs = [
        (s, img_index[s], split_of[s], boxes.get(s, []), out, P["max_side"], P["jpeg_quality"], orient["mode"])
        for s in train_stems
    ] + [
        (s, img_index[s], "test", None, out, P["max_side"], P["jpeg_quality"], orient["mode"])
        for s in test_stems if s in img_index
    ]
    # Boxes are in original pixel space; scale them to the resized image (header read only).
    jobs = [_prescale(j) for j in tqdm(jobs, desc="reading sizes")]

    errors = {}
    with ProcessPoolExecutor(max_workers=P["workers"]) as ex:
        for stem, err in tqdm(ex.map(process_one, jobs, chunksize=8), total=len(jobs), desc="processing"):
            if err:
                errors[stem] = err
    if errors:
        print(f"[warn] {len(errors)} images failed: {list(errors.items())[:5]}")

    meta = pd.DataFrame(
        {
            "stem": train_stems,
            "split": [split_of[s] for s in train_stems],
            "gt_abw": [gt[s][0] for s in train_stems],
            "gt_pbw": [gt[s][1] for s in train_stems],
            "n_boxes_abw": [box_counts[s][0] for s in train_stems],
            "n_boxes_pbw": [box_counts[s][1] for s in train_stems],
        }
    )
    meta = meta[~meta.stem.isin(errors)]
    meta.to_csv(out / "meta.csv", index=False)
    pd.Series(test_stems, name="stem").to_csv(out / "test_stems.csv", index=False)

    (out / "dataset.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: abw\n  1: pbw\n"
    )
    report = {
        "orientation": orient,
        "n_train": int((meta.split == "train").sum()),
        "n_val": int((meta.split == "val").sum()),
        "n_test": len(test_stems),
        "missing_train_images": len(missing),
        "missing_test_images": missing_test[:50],
        "count_box_mismatch": mism,
        "failed_images": errors,
        "sample_submission": str(sample_csv) if sample_csv else None,
        "gt_stats": meta[["gt_abw", "gt_pbw"]].describe().to_dict(),
    }
    (out / "prep_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("gt_stats", "failed_images")}, indent=2, default=str))
    print(f"[done] dataset at {out}")


def _prescale(job):
    """Scale box coordinates into the resized image's pixel space."""
    stem, src, split, boxes, out, max_side, q, mode = job
    if split == "test" or not boxes:
        return job
    w, h, o = raw_size_and_orientation(src)
    if mode == "exif" and o in (5, 6, 7, 8):
        w, h = h, w
    s = min(1.0, max_side / max(w, h))
    boxes = [(c, x1 * s, y1 * s, x2 * s, y2 * s) for c, x1, y1, x2, y2 in boxes]
    return stem, src, split, boxes, out, max_side, q, mode


if __name__ == "__main__":
    main()
