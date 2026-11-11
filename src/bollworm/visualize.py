"""Draw YOLO labels on a few processed images to check boxes line up with the moths.

Usage: python -m bollworm.visualize --config configs/default.yaml --n 12
Writes outputs/viz/*.jpg  (red = ABW, blue = PBW)
"""
from __future__ import annotations

import argparse
import random

from PIL import Image, ImageDraw

from bollworm.utils import load_config, resolve

COLORS = [(230, 40, 40), (40, 90, 230)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = resolve(cfg["paths"]["processed"])
    dst = resolve(cfg["paths"]["outputs"]) / "viz"
    dst.mkdir(parents=True, exist_ok=True)

    labels = [p for p in (proc / "labels" / args.split).glob("*.txt") if p.stat().st_size > 0]
    random.Random(cfg["seed"]).shuffle(labels)
    for lbl in labels[: args.n]:
        im = Image.open(proc / "images" / args.split / f"{lbl.stem}.jpg").convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLORS[c], width=3)
        im.save(dst / f"{lbl.stem}.jpg", quality=85)
    print(f"[done] {min(args.n, len(labels))} images -> {dst}")


if __name__ == "__main__":
    main()
