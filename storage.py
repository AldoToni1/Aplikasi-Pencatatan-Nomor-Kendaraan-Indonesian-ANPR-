from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

DEFAULT_CSV_PATH = Path("data") / "vehicle_logs.csv"
DEFAULT_CAPTURES_DIR = Path("data") / "captures"
DEFAULT_MAX_IMAGE_WIDTH = 800

LOG_COLUMNS: list[str] = [
    "record_id",
    "plate_number",
    "region",
    "check_in_time",
    "check_out_time",
    "duration_minutes",
    "status",
    "confidence",
    "image_path",
]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
STATUS_CHECK_IN = "Masuk"
STATUS_CHECK_OUT = "Keluar"
STATUS_ALL = "Semua"
UNKNOWN_REGION = "Tidak Diketahui"


def _ensure_csv_file(csv_path: str | Path = DEFAULT_CSV_PATH) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)
    return path


def init_storage(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    captures_dir: str | Path = DEFAULT_CAPTURES_DIR,
) -> Path:
    """Ensure the CSV log file and local vehicle captures directory exist."""
    path = _ensure_csv_file(csv_path)
    Path(captures_dir).mkdir(parents=True, exist_ok=True)
    return path


def save_vehicle_image(
    image_bytes: bytes,
    record_id: str,
    captures_dir: str | Path = DEFAULT_CAPTURES_DIR,
    max_width: int = DEFAULT_MAX_IMAGE_WIDTH,
) -> str:
    """Resize and save a vehicle check-in photo as JPEG, returning its POSIX path."""
    target_dir = Path(captures_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(io.BytesIO(image_bytes)) as opened_img:
        img = opened_img.convert("RGB")
        if max_width > 0 and img.width > max_width:
            new_height = max(1, int(round(img.height * (max_width / img.width))))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        output_path = target_dir / f"{record_id}.jpg"
        img.save(output_path, format="JPEG", quality=85, optimize=True)

    return output_path.as_posix()


def compute_elapsed_minutes(
    check_in_time: str,
    now: datetime | None = None,
) -> float:
    """Compute elapsed parking duration in minutes from a check-in timestamp string."""
    if not check_in_time:
        return 0.0
    try:
        check_in_dt = datetime.strptime(str(check_in_time).strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return 0.0

    current_time = now if now is not None else datetime.now()
    return max(0.0, round((current_time - check_in_dt).total_seconds() / 60.0, 1))


def format_duration(duration_minutes: float) -> str:
    """Format a duration in minutes into a human-readable Indonesian string."""
    if duration_minutes < 1.0:
        return "< 1 menit"

    total_minutes = int(round(duration_minutes))
    if total_minutes < 60:
        return f"{total_minutes} menit"

    hours, minutes = divmod(total_minutes, 60)
    return f"{hours} jam" if minutes == 0 else f"{hours} jam {minutes} menit"


def _normalize_plate(plate_number: str) -> str:
    normalized = " ".join(str(plate_number).strip().upper().split())
    if not normalized:
        raise ValueError("Plate number must not be empty.")
    return normalized


def _normalize_region(region: str) -> str:
    return region.strip() if isinstance(region, str) and region.strip() else UNKNOWN_REGION


def _row_to_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    try:
        record["confidence"] = float(record["confidence"])
    except (TypeError, ValueError):
        record["confidence"] = 0.0

    raw_duration = str(record.get("duration_minutes", "")).strip()
    if raw_duration:
        try:
            record["duration_minutes"] = float(raw_duration)
        except ValueError:
            pass
    return record


def _read_raw_logs(path: Path) -> pd.DataFrame:
    _ensure_csv_file(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[LOG_COLUMNS]


def _next_record_id(df: pd.DataFrame) -> str:
    max_seq = 0
    for rid in df["record_id"]:
        if isinstance(rid, str) and rid.startswith("LOG-") and rid[4:].isdigit():
            max_seq = max(max_seq, int(rid[4:]))
    return f"LOG-{max_seq + 1:04d}"


def load_logs(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load all vehicle logs sorted by latest check-in time first."""
    path = _ensure_csv_file(csv_path)
    df = _read_raw_logs(path)
    if df.empty:
        return df

    ordered = df.copy()
    ordered["_row_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        by=["check_in_time", "_row_order"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns=["_row_order"])
    return ordered.reset_index(drop=True)


def check_in_vehicle(
    plate_number: str,
    region: str,
    confidence: float,
    image_bytes: bytes | None = None,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    captures_dir: str | Path = DEFAULT_CAPTURES_DIR,
) -> dict[str, Any]:
    """Record a new active vehicle check-in and optional photo, rejecting active duplicates."""
    normalized_plate = _normalize_plate(plate_number)
    clean_region = _normalize_region(region)
    clean_confidence = round(max(0.0, min(1.0, float(confidence))), 4)

    path = init_storage(csv_path, captures_dir)
    df = _read_raw_logs(path)

    active_mask = (df["plate_number"] == normalized_plate) & (df["status"] == STATUS_CHECK_IN)
    active_indices = df.index[active_mask].tolist()
    if active_indices:
        existing_record = _row_to_record(df.loc[active_indices[-1]])
        return {"recorded": False, "reason": "duplicate", "record": existing_record}

    record_id = _next_record_id(df)
    image_path = (
        save_vehicle_image(image_bytes, record_id, captures_dir=captures_dir)
        if image_bytes is not None
        else ""
    )

    now_str = datetime.now().strftime(TIMESTAMP_FORMAT)
    record: dict[str, Any] = {
        "record_id": record_id,
        "plate_number": normalized_plate,
        "region": clean_region,
        "check_in_time": now_str,
        "check_out_time": "",
        "duration_minutes": "",
        "status": STATUS_CHECK_IN,
        "confidence": clean_confidence,
        "image_path": image_path,
    }
    csv_row = {k: str(v) for k, v in record.items()}
    pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True).to_csv(path, index=False)
    return {"recorded": True, "reason": "", "record": record}


def update_record_plate(
    record_id: str,
    new_plate_number: str,
    new_region: str,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Any]:
    """Update the plate number and region of an active check-in record in place."""
    normalized_new_plate = _normalize_plate(new_plate_number)
    clean_region = _normalize_region(new_region)
    clean_record_id = str(record_id).strip()

    path = _ensure_csv_file(csv_path)
    df = _read_raw_logs(path)

    matching_indices = df.index[df["record_id"] == clean_record_id].tolist()
    if not matching_indices:
        return {"updated": False, "reason": "not_found", "record": None}

    target_idx = matching_indices[-1]
    if df.at[target_idx, "status"] == STATUS_CHECK_OUT:
        return {
            "updated": False,
            "reason": "already_checked_out",
            "record": _row_to_record(df.loc[target_idx]),
        }

    conflict_mask = (
        (df["record_id"] != clean_record_id)
        & (df["status"] == STATUS_CHECK_IN)
        & (df["plate_number"] == normalized_new_plate)
    )
    conflict_indices = df.index[conflict_mask].tolist()
    if conflict_indices:
        return {
            "updated": False,
            "reason": "duplicate",
            "record": _row_to_record(df.loc[conflict_indices[-1]]),
        }

    df.at[target_idx, "plate_number"] = normalized_new_plate
    df.at[target_idx, "region"] = clean_region
    df.to_csv(path, index=False)

    return {
        "updated": True,
        "reason": "",
        "record": _row_to_record(df.loc[target_idx]),
    }


def check_out_vehicle(
    record_id: str,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Any]:
    """Mark an active vehicle record as checked out and compute its parking duration."""
    clean_record_id = str(record_id).strip()
    path = _ensure_csv_file(csv_path)
    df = _read_raw_logs(path)

    active_mask = (df["record_id"] == clean_record_id) & (df["status"] == STATUS_CHECK_IN)
    active_indices = df.index[active_mask].tolist()
    if not active_indices:
        return {
            "checked_out": False,
            "reason": "not_active",
            "duration_text": "",
            "record": None,
        }

    target_idx = active_indices[-1]
    now = datetime.now()
    now_str = now.strftime(TIMESTAMP_FORMAT)
    elapsed = compute_elapsed_minutes(str(df.at[target_idx, "check_in_time"]), now=now)

    df.at[target_idx, "check_out_time"] = now_str
    df.at[target_idx, "duration_minutes"] = str(elapsed)
    df.at[target_idx, "status"] = STATUS_CHECK_OUT
    df.to_csv(path, index=False)

    updated_record = _row_to_record(df.loc[target_idx])
    updated_record["duration_minutes"] = elapsed
    return {
        "checked_out": True,
        "reason": "",
        "duration_text": format_duration(elapsed),
        "record": updated_record,
    }


def get_active_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a reset-index copy of records currently checked in."""
    if df.empty:
        return df.copy()
    return df[df["status"] == STATUS_CHECK_IN].copy().reset_index(drop=True)


def filter_logs(
    df: pd.DataFrame,
    query: str = "",
    status: str = STATUS_ALL,
) -> pd.DataFrame:
    """Filter vehicle log rows by optional status and partial plate query."""
    if df.empty:
        return df.copy()

    filtered = df.copy()
    if status and status != STATUS_ALL:
        filtered = filtered[filtered["status"] == status]

    clean_query = query.strip().upper() if isinstance(query, str) else ""
    if clean_query:
        compact_query = clean_query.replace(" ", "")
        plate_upper = filtered["plate_number"].astype(str).str.upper()
        plate_compact = plate_upper.str.replace(" ", "", regex=False)
        mask = plate_upper.str.contains(clean_query, regex=False) | plate_compact.str.contains(
            compact_query, regex=False
        )
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


def clear_logs(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    captures_dir: str | Path = DEFAULT_CAPTURES_DIR,
) -> int:
    """Remove all vehicle log rows and stored check-in capture images."""
    path = init_storage(csv_path, captures_dir)
    cleared_count = len(_read_raw_logs(path))
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)

    captures_path = Path(captures_dir)
    if captures_path.is_dir():
        for file_path in captures_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                file_path.unlink(missing_ok=True)

    return cleared_count
