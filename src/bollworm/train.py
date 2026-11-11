"""Train a YOLOv8 detector on the prepared dataset.

Usage: python -m bollworm.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil

from bollworm.utils import load_config, resolve, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true", help="resume the run named in the config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    T = cfg["train"]

    from ultralytics import YOLO

    data_yaml = resolve(cfg["paths"]["processed"]) / "dataset.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found -- run bollworm.prepare_data first")
    project = resolve(cfg["paths"]["runs"])

    if args.resume:
        model = YOLO(project / T["name"] / "weights" / "last.pt")
        model.train(resume=True)
    else:
        model = YOLO(T["model"])
        model.train(
            data=str(data_yaml),
            imgsz=T["imgsz"],
            epochs=T["epochs"],
            batch=T["batch"],
            patience=T["patience"],
            workers=T["workers"],
            fraction=T.get("fraction", 1.0),
            device=T.get("device"),
            project=str(project),
            name=T["name"],
            exist_ok=True,
            seed=cfg["seed"],
            deterministic=True,
            # augmentations suited to top-down trap photos
            fliplr=0.5,
            flipud=0.5,
            degrees=10.0,
            mosaic=1.0,
            close_mosaic=5,
            plots=True,
        )

    best = project / T["name"] / "weights" / "best.pt"
    out = resolve(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out / f"{T['name']}_best.pt")
    (out / "last_model.json").write_text(json.dumps({"weights": str(out / f"{T['name']}_best.pt")}))
    print(f"[done] best weights -> {out / (T['name'] + '_best.pt')}")


if __name__ == "__main__":
    main()
