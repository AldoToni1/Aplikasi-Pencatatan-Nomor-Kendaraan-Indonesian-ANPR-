# Real Public Test Samples (`data/samples/`)

Place real Indonesian vehicle test images (`.jpg` / `.png`) here along with their ground-truth labels in `ground_truth.csv` (or pass a YOLO `.txt` labels folder to `benchmark.py --ground-truth`).

## Recommended Public Dataset
- **Kaggle:** `juanthomaswijaya/indonesian-license-plate-dataset`
- Suggested coverage: 20-30 real samples spanning black-background TNKB, white-background TNKB, cars, and motorcycles (including slight angles and expiry-date rows).

## Legality & Privacy Rules
- Use only open-licensed public datasets (e.g., CC BY 4.0 / MIT).
- Do not include unconsented personal street captures or unblurred human faces.

## Running CPU Benchmark
```bash
# Using ground_truth.csv
python benchmark.py --images-dir data/samples --ground-truth data/samples/ground_truth.csv

# Or directly using a Kaggle YOLO character labels directory
python benchmark.py --images-dir /path/to/images --ground-truth /path/to/labels --limit 30
```
