"""Detect -> count -> tune per-class thresholds on val -> write submission.

Usage:
  python -m bollworm.predict --config configs/default.yaml [--weights path/to/best.pt]

Outputs (in outputs/):
  dets_<name>_{val,test}.json   per-image sorted confidences per class (re-usable, e.g. ensembling)
  thresholds_<name>.json        tuned thresholds + val MAE
  submission_<name>.csv         ready to upload to Zindi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bollworm.utils import CLASSES, competition_mae, image_stem, load_config, resolve, row_mae, set_seed, split_image_class_id


def run_detector(weights: str, paths: list[Path], P: dict, device=None) -> dict[str, list[list[float]]]:
    from ultralytics import YOLO

    model = YOLO(weights)
    dets: dict[str, list[list[float]]] = {}
    bs = P.get("batch", 8)
    for i in tqdm(range(0, len(paths), bs), desc="detect"):
        chunk = paths[i : i + bs]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=P["imgsz"],
            conf=P["min_conf"],
            iou=P["iou"],
            max_det=P["max_det"],
            augment=P.get("tta", False),
            device=device,
            verbose=False,
        )
        for p, r in zip(chunk, results):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            dets[image_stem(p.name)] = [sorted(conf[cls == c].tolist(), reverse=True) for c in range(len(CLASSES))]
    return dets


def counts_at(dets: dict, thr: list[float]) -> dict[str, tuple[int, int]]:
    return {s: tuple(int(np.sum(np.asarray(d[c]) >= thr[c])) for c in range(2)) for s, d in dets.items()}


def tune_thresholds(dets: dict, gt: dict) -> tuple[list[float], float]:
    """Per-class grid search; the metric is separable across classes."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    stems = [s for s in gt if s in dets]
    best = []
    for c in range(2):
        y = np.array([gt[s][c] for s in stems])
        confs = [np.asarray(dets[s][c]) for s in stems]
        maes = [np.mean(np.abs(np.array([np.sum(cf >= t) for cf in confs]) - y)) for t in grid]
        best.append(float(grid[int(np.argmin(maes))]))
    mae = competition_mae({s: gt[s] for s in stems}, counts_at({s: dets[s] for s in stems}, best))
    return best, mae


def write_submission(counts: dict, sample_csv: Path | None, test_stems: list[str], out_path: Path, fallback=(0, 0)):
    if sample_csv is not None and sample_csv.exists():
        sub = pd.read_csv(sample_csv)
        id_col, tgt_col = sub.columns[0], sub.columns[1]
        vals, missing = [], 0
        for rid in sub[id_col]:
            stem, cls = split_image_class_id(rid)
            if stem not in counts:
                missing += 1
            vals.append(int(counts.get(stem, fallback)[CLASSES.index(cls)]))
        sub[tgt_col] = vals
        if missing:
            print(f"[warn] {missing} submission rows had no prediction -> filled with fallback {fallback}")
    else:
        rows = [(f"{s}_{c}", int(counts.get(s, fallback)[i])) for s in test_stems for i, c in enumerate(CLASSES)]
        sub = pd.DataFrame(rows, columns=["Image_ID", "Target"])
    sub.to_csv(out_path, index=False)
    print(f"[done] submission -> {out_path}  ({len(sub)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse saved detections instead of re-running the model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    P, name = cfg["predict"], cfg["train"]["name"]
    proc, out = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    weights = args.weights or json.loads((out / "last_model.json").read_text())["weights"]
    limit = P.get("limit")

    meta = pd.read_csv(proc / "meta.csv")
    val = meta[meta.split == "val"]
    if limit:
        val = val.head(limit)
    gt = {r.stem: (int(r.gt_abw), int(r.gt_pbw)) for r in val.itertuples()}
    test_stems = pd.read_csv(proc / "test_stems.csv")["stem"].tolist()
    run_test = test_stems[:limit] if limit else test_stems

    det_files = {k: out / f"dets_{name}_{k}.json" for k in ("val", "test")}
    device = cfg["train"].get("device")
    if args.reuse and all(f.exists() for f in det_files.values()):
        dets_val = json.loads(det_files["val"].read_text())
        dets_test = json.loads(det_files["test"].read_text())
    else:
        dets_val = run_detector(weights, [proc / "images/val" / f"{s}.jpg" for s in gt], P, device)
        dets_test = run_detector(weights, [proc / "images/test" / f"{s}.jpg" for s in run_test if (proc / "images/test" / f"{s}.jpg").exists()], P, device)
        det_files["val"].write_text(json.dumps(dets_val))
        det_files["test"].write_text(json.dumps(dets_test))

    thr, mae = tune_thresholds(dets_val, gt)
    naive = counts_at(dets_val, [0.25, 0.25])
    report = {
        "weights": weights,
        "thresholds": dict(zip(CLASSES, thr)),
        "val_mae_per_image": mae,
        "val_mae_zindi_rows": mae / 2,
        "val_mae_rows_at_0.25": row_mae(gt, naive),
        "val_mae_rows_predict_zero": row_mae(gt, {}),
        "n_val": len(gt),
    }
    (out / f"thresholds_{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    raw = Path(resolve(cfg["paths"]["raw"]))
    sample = next(iter(sorted(raw.rglob("SampleSubmission.csv"))), None)
    write_submission(counts_at(dets_test, thr), sample, test_stems, out / f"submission_{name}.csv")


if __name__ == "__main__":
    main()
