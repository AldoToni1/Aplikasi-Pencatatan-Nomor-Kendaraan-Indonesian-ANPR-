from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

from plate_pipeline import CHAR_CLASS_NAMES, filter_main_plate_row, load_models, process_vehicle_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_COLUMN_CANDIDATES = ("filename", "image", "image_name", "file", "image_path", "path")
LABEL_COLUMN_CANDIDATES = ("ground_truth", "plate_number", "plate", "label", "text")


def normalize_plate_text(text: str) -> str:
    """Strip non-alphanumeric characters and uppercase plate text for exact matching."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    lower_map = {col.strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    raise ValueError(f"Missing {role} column in CSV. Expected one of: {', '.join(candidates)}")


def _parse_yolo_char_label(label_path: Path) -> str:
    detections: list[dict[str, float | str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        char = CHAR_CLASS_NAMES.get(cls_id, "")
        if not char:
            continue
        detections.append(
            {
                "char": char,
                "conf": 1.0,
                "x_c": float(parts[1]) * 1000.0,
                "y_c": float(parts[2]) * 1000.0,
                "w": float(parts[3]) * 1000.0,
                "h": float(parts[4]) * 1000.0,
            }
        )
    main_row = filter_main_plate_row(detections, crop_height=1000)
    return "".join(str(item["char"]) for item in main_row)


def load_ground_truth(gt_source: Path, images_dir: Path, limit: int | None = None) -> list[tuple[Path, str]]:
    """Load image paths and ground-truth plate labels from a CSV file or YOLO label directory."""
    samples: list[tuple[Path, str]] = []
    if gt_source.is_dir():
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = gt_source / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            label = _parse_yolo_char_label(label_path)
            if label:
                samples.append((img_path, label))
            if limit is not None and len(samples) >= limit:
                break
        return samples

    df = pd.read_csv(gt_source, dtype=str).fillna("")
    img_col = _pick_column(df, IMAGE_COLUMN_CANDIDATES, "image")
    label_col = _pick_column(df, LABEL_COLUMN_CANDIDATES, "ground-truth label")

    for _, row in df.iterrows():
        raw_name = row[img_col].strip()
        label = row[label_col].strip()
        if not raw_name or not label:
            continue
        img_path = Path(raw_name)
        if not img_path.is_absolute():
            img_path = images_dir / raw_name
        if img_path.exists():
            samples.append((img_path, label))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def classify_error(plate_detected: bool, raw_text: str, is_valid_format: bool, exact_match: bool) -> str:
    """Categorize a benchmark prediction outcome into an error bucket."""
    if exact_match:
        return "none"
    if not plate_detected:
        return "plate_not_detected"
    if not raw_text:
        return "no_chars_recognized"
    if not is_valid_format:
        return "invalid_format"
    return "char_mismatch"


def run_benchmark(
    images_dir: Path,
    ground_truth_csv: Path,
    output_csv: Path,
    det_conf: float = 0.25,
    rec_conf: float = 0.25,
    limit: int | None = None,
) -> pd.DataFrame:
    """Evaluate the 2-stage CPU ANPR pipeline on labeled images and export an error analysis CSV."""
    samples = load_ground_truth(ground_truth_csv, images_dir, limit=limit)
    if not samples:
        raise FileNotFoundError(
            f"No valid image samples matched {ground_truth_csv} inside {images_dir}."
        )

    detector, recognizer = load_models()
    records: list[dict[str, object]] = []

    for img_path, gt_label in samples:
        with Image.open(img_path) as img:
            image_rgb = np.array(img.convert("RGB"))

        start = time.perf_counter()
        result = process_vehicle_image(
            image_rgb=image_rgb,
            detector=detector,
            recognizer=recognizer,
            det_conf=det_conf,
            rec_conf=rec_conf,
        )
        elapsed = time.perf_counter() - start

        gt_norm = normalize_plate_text(gt_label)
        pred_norm = normalize_plate_text(result.formatted_plate or result.raw_text)
        exact_match = gt_norm == pred_norm
        error_type = classify_error(
            result.plate_detected,
            result.raw_text,
            result.is_valid_format,
            exact_match,
        )

        records.append(
            {
                "filename": img_path.name,
                "ground_truth": gt_label,
                "raw_text": result.raw_text,
                "predicted_plate": result.formatted_plate,
                "predicted_region": result.region,
                "plate_detected": result.plate_detected,
                "is_valid_format": result.is_valid_format,
                "exact_match": exact_match,
                "error_type": error_type,
                "confidence": result.confidence,
                "inference_sec": round(elapsed, 4),
            }
        )

    results_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    total = len(results_df)
    det_rate = 100.0 * float(results_df["plate_detected"].sum()) / total
    exact_rate = 100.0 * float(results_df["exact_match"].sum()) / total
    avg_time = float(results_df["inference_sec"].mean())

    print("=== Indonesian ANPR CPU Benchmark Report ===")
    print(f"Total Tested                   : {total}")
    print(f"Stage 1 Plate Detection Rate   : {det_rate:.2f}%")
    print(f"End-to-End Exact Match Accuracy: {exact_rate:.2f}%")
    print(f"Avg CPU Inference Time         : {avg_time:.4f} sec/image")
    print(f"Error Analysis CSV             : {output_csv}")

    failures = results_df[~results_df["exact_match"]]
    if not failures.empty:
        print("\nTop Failure Cases (by Category):")
        for err_type, count in Counter(failures["error_type"]).most_common():
            print(f"  - {err_type}: {count}")
        print("\nSample Mismatches (up to 5):")
        for _, row in failures.head(5).iterrows():
            print(
                f"  - {row['filename']}: GT='{row['ground_truth']}' | "
                f"Pred='{row['predicted_plate']}' | Raw='{row['raw_text']}' | Err={row['error_type']}"
            )
    else:
        print("\nTop Failure Cases: None (100% exact match).")

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CPU benchmark runner."""
    parser = argparse.ArgumentParser(
        description="CPU benchmark for 2-stage Indonesian ANPR pipeline on real labeled datasets."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/samples/ground_truth.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/benchmark_error_analysis.csv"))
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--rec-conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the CPU benchmark CLI entry point."""
    args = parse_args()
    run_benchmark(
        images_dir=args.images_dir,
        ground_truth_csv=args.ground_truth,
        output_csv=args.output_csv,
        det_conf=args.det_conf,
        rec_conf=args.rec_conf,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
