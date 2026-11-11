"""No-model baseline: predict the training median count for every test image.

Use it to check the submission format is accepted by Zindi before training.
Usage: python -m bollworm.baseline --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from bollworm.predict import write_submission
from bollworm.utils import load_config, resolve, row_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc, out, raw = (resolve(cfg["paths"][k]) for k in ("processed", "outputs", "raw"))
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(proc / "meta.csv")
    tr, va = meta[meta.split == "train"], meta[meta.split == "val"]
    med = (int(tr.gt_abw.median()), int(tr.gt_pbw.median()))
    gt = {r.stem: (r.gt_abw, r.gt_pbw) for r in va.itertuples()}
    print(f"median counts (abw, pbw) = {med}; val row-MAE = {row_mae(gt, {s: med for s in gt}):.3f}")

    test_stems = pd.read_csv(proc / "test_stems.csv")["stem"].tolist()
    sample = next(iter(sorted(raw.rglob("SampleSubmission.csv"))), None)
    write_submission({s: med for s in test_stems}, sample, test_stems, out / "submission_baseline_median.csv")


if __name__ == "__main__":
    main()
