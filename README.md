# Wadhwani AI Bollworm Counting Challenge (Zindi)

This project counts **ABW** (American bollworm) and **PBW** (pink bollworm) moths in photos of pheromone traps.
It is scored by MAE (mean absolute error) over the per-image counts.

**Approach:** a YOLOv8 object detector finds every moth. The number of detections above a confidence
threshold is the count. The threshold for each class is tuned on a held-out validation split to
minimise MAE directly.

```
data/raw (Zindi files) ──prepare_data──▶ data/processed (resized JPGs + YOLO labels + split)
                                              │
                                     train (YOLOv8, GPU)
                                              │
                    predict: detect val+test ─▶ tune per-class thresholds on val ─▶ submission.csv
```

## Project layout

| Path | Purpose |
|---|---|
| `src/bollworm/prepare_data.py` | Unzips data, reads the CSVs, fixes image orientation, resizes, writes YOLO labels, makes a stratified train/val split |
| `src/bollworm/visualize.py` | Draws the labels on sample images (`outputs/viz/`) so you can check that the boxes sit on the moths |
| `src/bollworm/baseline.py` | Median-count submission with no model, used to test the upload format |
| `src/bollworm/train.py` | Trains YOLOv8 (ultralytics) |
| `src/bollworm/predict.py` | Runs detection, tunes the thresholds, computes val MAE and writes the submission |
| `src/bollworm/utils.py` | Metric, ID parsing, WKT box parsing, seeding |
| `configs/default.yaml` | Real run settings (GPU) |
| `configs/cpu_smoke.yaml` | Short CPU run to check the pipeline works |
| `notebooks/train_gpu_colab.ipynb` | Full run on a free Colab T4 GPU |
| `tests/` | Unit tests (`pytest`) |

## 1. Environment

**Local (WSL, CPU only, used for development and smoke tests):**
```bash
bash scripts/setup_wsl.sh               # creates ~/.venvs/bollworm with CPU torch + pinned deps
source ~/.venvs/bollworm/bin/activate
pytest -q
```

**GPU (needed for real training):** use Google Colab (T4) through `notebooks/train_gpu_colab.ipynb`, or Kaggle
(add the data as a **private** dataset). This laptop has no GPU, and training on the CPU would take days.

## 2. Get the data

The data is only available through your Zindi login (it can't be fetched by script):
**Zindi → Wadhwani AI Bollworm Counting Challenge → Data**. Download every file into `data/raw/`:

```
data/raw/Train.csv
data/raw/Test.csv
data/raw/SampleSubmission.csv
data/raw/images_bboxes.csv
data/raw/images.zip          # extracted automatically
```

> Disk: drive C: has only about 24 GB free. The image zip plus its extracted copy plus the resized copy can take
> 10–15 GB. Once `prepare_data` has finished you can delete `images.zip` and the extracted raw images.
>
> Zindi rules: never upload this data anywhere public (GitHub, public Kaggle datasets, …). `data/` is git-ignored.

## 3. Run

```bash
python -m bollworm.prepare_data --config configs/default.yaml   # check outputs/viz after the next line!
python -m bollworm.visualize    --config configs/default.yaml --n 12
python -m bollworm.baseline     --config configs/default.yaml   # optional: format check on Zindi
python -m bollworm.train        --config configs/default.yaml   # GPU
python -m bollworm.predict      --config configs/default.yaml   # -> outputs/submission_<name>.csv
```

CPU smoke test (about 10 minutes, scores are meaningless): use `--config configs/cpu_smoke.yaml` for prepare, train and predict.

## Details that matter

- **Orientation:** phone photos carry an EXIF rotation tag. If the boxes were drawn on the raw pixels but the image is
  loaded rotated (or the other way round), every label is wrong. `prepare_data` checks which interpretation keeps the
  boxes inside the image (`prep_report.json → orientation`), applies that choice to train and test, and saves JPGs with
  no EXIF data so nothing gets rotated twice. **Always check `outputs/viz/`.**
- **Ground truth** comes from `Train.csv` counts, and the box file supplies the detector labels. The prep step reports how many images disagree.
- **Many moths per image:** `max_det: 1500`. Moths are small, so don't shrink the images too much (`max_side 1536`, `imgsz 1024`).
- **Thresholds** are tuned on val for each class (0.05–0.95). The report compares the result with the fixed 0.25 threshold and with predicting zero.
- **Metric:** `val_mae_zindi_rows` is the number to compare with the leaderboard (MAE over submission rows).
- **Reproducibility:** fixed seed, `deterministic=True`, a seeded stratified split and pinned `requirements.txt`.

## Ideas to improve the score

1. Use a bigger model or larger images (`yolov8m.pt`, `imgsz 1280`), which needs more GPU time.
2. Train on train+val after choosing the thresholds (keep the thresholds and the number of epochs).
3. Turn on TTA (`predict.tta: true`), or ensemble 2–3 models by combining their saved `dets_*.json` confidences.
4. Use tiled inference (SAHI-style) for very crowded traps.
5. Add a count-correction regressor (e.g. on the sorted confidence vector) trained on out-of-fold val predictions.

## Zindi reproducibility documentation

- **Data used:** only the Zindi competition files above. No external data. The only pretrained weights are the public COCO YOLOv8 weights (ultralytics).
- **Outputs:** `data/processed/` (prepared dataset, `meta.csv`, `prep_report.json`), `runs/<name>/` (training logs and weights),
  `outputs/` (`*_best.pt`, `dets_*.json`, `thresholds_*.json`, `submission_*.csv`).
- **Features:** pixels only. No image metadata is used as a model input (EXIF orientation is used only to display the pixels correctly).
- **Environment:** development on WSL2 Ubuntu, Python 3.12, CPU. Training on Google Colab with an NVIDIA T4 (16 GB).
- **Expected run time (T4):** prepare 15–25 min · train 5–7 h (yolov8s, 1024, 40 epochs) · predict about 15 min.
