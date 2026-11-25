from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
from ultralytics import YOLO

from storage import UNKNOWN_REGION

DETECTOR_REPO_ID = "carlravel/indonesian-plate-detector-yolo11"
DETECTOR_ONNX_FILE = "plate_detector_yolo11n.onnx"
DETECTOR_PT_FILE = "plate_detector_yolo11n.pt"

RECOGNIZER_REPO_ID = "carlravel/indonesian-plate-recognizer-yolo11"
RECOGNIZER_ONNX_FILE = "plate_recognizer_yolo11n.onnx"
RECOGNIZER_PT_FILE = "plate_recognizer_yolo11n.pt"

CHAR_CLASS_NAMES: dict[int, str] = {
    **{i: str(i) for i in range(10)},
    **{10 + i: chr(ord("A") + i) for i in range(26)},
}

INDONESIAN_REGION_CODES: dict[str, str] = {
    "BL": "Aceh",
    "BB": "Sumatera Utara",
    "BK": "Sumatera Utara",
    "BA": "Sumatera Barat",
    "BM": "Riau",
    "BP": "Kepulauan Riau",
    "BH": "Jambi",
    "BD": "Bengkulu",
    "BG": "Sumatera Selatan",
    "BN": "Bangka Belitung",
    "BE": "Lampung",
    "A": "Banten",
    "B": "DKI Jakarta, Depok, Tangerang, Bekasi",
    "D": "Bandung Raya",
    "E": "Cirebon/Indramayu/Majalengka/Kuningan",
    "F": "Bogor/Sukabumi/Cianjur",
    "T": "Purwakarta/Karawang/Subang",
    "Z": "Garut/Tasikmalaya/Ciamis/Sumedang",
    "G": "Pekalongan/Tegal/Brebes",
    "H": "Semarang/Salatiga/Kendal/Demak",
    "K": "Pati/Kudus/Jepara/Rembang/Blora",
    "R": "Banyumas/Cilacap/Purbalingga/Banjarnegara",
    "AA": "Kedu/Magelang/Kebumen/Wonosobo",
    "AB": "DI Yogyakarta",
    "AD": "Surakarta/Solo Raya",
    "L": "Surabaya",
    "M": "Madura",
    "N": "Malang/Pasuruan/Probolinggo/Lumajang",
    "P": "Besuki/Jember/Banyuwangi",
    "S": "Bojonegoro/Tuban/Lamongan/Mojokerto",
    "W": "Sidoarjo/Gresik",
    "AE": "Madiun/Ponorogo/Ngawi",
    "AG": "Kediri/Blitar/Tulungagung",
    "DK": "Bali",
    "DR": "Nusa Tenggara Barat",
    "EA": "Nusa Tenggara Barat",
    "DH": "Nusa Tenggara Timur",
    "EB": "Nusa Tenggara Timur",
    "ED": "Nusa Tenggara Timur",
    "KB": "Kalimantan Barat",
    "DA": "Kalimantan Selatan",
    "KH": "Kalimantan Tengah",
    "KT": "Kalimantan Timur",
    "KU": "Kalimantan Utara",
    "DB": "Sulawesi Utara",
    "DL": "Sulawesi Utara",
    "DM": "Gorontalo",
    "DN": "Sulawesi Tengah",
    "DC": "Sulawesi Barat",
    "DD": "Sulawesi Selatan",
    "DP": "Sulawesi Selatan",
    "DW": "Sulawesi Selatan",
    "DT": "Sulawesi Tenggara",
    "DE": "Maluku",
    "DG": "Maluku Utara",
    "PA": "Papua",
    "PB": "Papua Barat",
}

TNKB_PATTERN = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$")
DIGIT_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"}


@dataclass
class PlateResult:
    raw_text: str
    formatted_plate: str
    region: str
    is_valid_format: bool
    confidence: float
    annotated_image: np.ndarray
    cropped_plate: np.ndarray | None
    plate_detected: bool


def _download_weight(repo_id: str, filename: str) -> str:
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)


def _load_yolo_with_fallback(
    repo_id: str,
    onnx_filename: str,
    pt_filename: str,
    expected_classes: int | None = None,
) -> YOLO:
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        onnx_path = _download_weight(repo_id, onnx_filename)
        model = YOLO(onnx_path, task="detect")
        warmup = model.predict(dummy_img, device="cpu", verbose=False)
        if expected_classes is not None:
            names = warmup[0].names if warmup else {}
            if len(names) != expected_classes:
                raise RuntimeError(f"Unexpected ONNX class count: {len(names)}")
        return model
    except Exception:
        pt_path = _download_weight(repo_id, pt_filename)
        model = YOLO(pt_path, task="detect")
        model.predict(dummy_img, device="cpu", verbose=False)
        return model


def load_models() -> tuple[YOLO, YOLO]:
    """Download and load the 2-stage YOLO11n plate detector and character recognizer on CPU."""
    detector = _load_yolo_with_fallback(
        repo_id=DETECTOR_REPO_ID,
        onnx_filename=DETECTOR_ONNX_FILE,
        pt_filename=DETECTOR_PT_FILE,
    )
    recognizer = _load_yolo_with_fallback(
        repo_id=RECOGNIZER_REPO_ID,
        onnx_filename=RECOGNIZER_ONNX_FILE,
        pt_filename=RECOGNIZER_PT_FILE,
        expected_classes=len(CHAR_CLASS_NAMES),
    )
    return detector, recognizer


def _compute_box_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    ax1, ay1 = box_a["x_c"] - box_a["w"] / 2.0, box_a["y_c"] - box_a["h"] / 2.0
    ax2, ay2 = box_a["x_c"] + box_a["w"] / 2.0, box_a["y_c"] + box_a["h"] / 2.0
    bx1, by1 = box_b["x_c"] - box_b["w"] / 2.0, box_b["y_c"] - box_b["h"] / 2.0
    bx2, by2 = box_b["x_c"] + box_b["w"] / 2.0, box_b["y_c"] + box_b["h"] / 2.0

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = (
        max(0.0, box_a["w"]) * max(0.0, box_a["h"])
        + max(0.0, box_b["w"]) * max(0.0, box_b["h"])
        - inter_area
    )
    return inter_area / union_area if union_area > 0.0 else 0.0


def deduplicate_char_boxes(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.45,
    center_dist_ratio: float = 0.35,
) -> list[dict[str, Any]]:
    """Suppress overlapping duplicate character detections using IoU and center-distance NMS."""
    accepted: list[dict[str, Any]] = []
    for candidate in sorted(detections, key=lambda d: d["conf"], reverse=True):
        is_duplicate = False
        for kept in accepted:
            avg_width = 0.5 * (candidate["w"] + kept["w"])
            center_dist = math.hypot(
                candidate["x_c"] - kept["x_c"],
                candidate["y_c"] - kept["y_c"],
            )
            if _compute_box_iou(candidate, kept) > iou_threshold or (
                avg_width > 0 and center_dist < center_dist_ratio * avg_width
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            accepted.append(candidate)
    return accepted


def filter_main_plate_row(detections: list[dict[str, Any]], crop_height: int) -> list[dict[str, Any]]:
    """Filter out lower expiry-date digits and sort main-row characters left to right."""
    if not detections:
        return []

    heights = sorted((d["h"] for d in detections), reverse=True)
    top_k = min(3, len(heights))
    ref_height = sum(heights[:top_k]) / top_k

    # Discard shorter expiry-date digits (~40-55% of main character height)
    tall_chars = [d for d in detections if d["h"] >= 0.62 * ref_height] or detections
    y_centers = sorted(d["y_c"] for d in tall_chars)
    median_y = y_centers[len(y_centers) // 2]

    # Discard characters sitting below the main plate row baseline
    main_row = [
        d
        for d in tall_chars
        if (d["y_c"] - median_y) <= (0.55 * ref_height)
        and d["y_c"] <= (0.78 * crop_height)
    ]
    return sorted(main_row or tall_chars, key=lambda d: d["x_c"])


def _convert_segment(segment: str, to_letters: bool) -> tuple[str | None, int]:
    mapping = DIGIT_TO_LETTER if to_letters else LETTER_TO_DIGIT
    out_chars: list[str] = []
    edits = 0
    for ch in segment:
        if (to_letters and ch.isalpha()) or (not to_letters and ch.isdigit()):
            out_chars.append(ch)
        elif ch in mapping:
            out_chars.append(mapping[ch])
            edits += 1
        else:
            return None, 0
    return "".join(out_chars), edits


def format_indonesian_plate(raw_text: str) -> tuple[str, bool]:
    """Normalize raw plate text to Indonesian TNKB spacing and report format validity."""
    cleaned = re.sub(r"[^A-Z0-9]", "", str(raw_text).upper())
    if not cleaned:
        return "", False

    match = TNKB_PATTERN.match(cleaned)
    if match:
        prefix, number, suffix = match.groups()
        return f"{prefix} {number} {suffix}".strip(), True

    # Require both letters and digits with minimum length 3 before fuzzy OCR fallback
    if len(cleaned) < 3 or not (
        any(ch.isalpha() for ch in cleaned) and any(ch.isdigit() for ch in cleaned)
    ):
        return cleaned, False

    # Conservative positional correction fallback (max 2 edits)
    best_candidate: tuple[int, int, str] | None = None
    total_len = len(cleaned)
    for p_len in (1, 2):
        for n_len in (1, 2, 3, 4):
            s_len = total_len - p_len - n_len
            if not (0 <= s_len <= 3):
                continue

            prefix, e1 = _convert_segment(cleaned[:p_len], to_letters=True)
            number, e2 = _convert_segment(cleaned[p_len : p_len + n_len], to_letters=False)
            suffix, e3 = _convert_segment(cleaned[p_len + n_len :], to_letters=True)
            if prefix is None or number is None or suffix is None:
                continue

            total_edits = e1 + e2 + e3
            if total_edits > 2 or not TNKB_PATTERN.match(f"{prefix}{number}{suffix}"):
                continue

            region_penalty = 0 if prefix in INDONESIAN_REGION_CODES else 1
            candidate = (total_edits, region_penalty, f"{prefix} {number} {suffix}".strip())
            if best_candidate is None or candidate[:2] < best_candidate[:2]:
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate[2], True
    return cleaned, False


def get_plate_region(plate: str) -> str:
    """Resolve the Indonesian registration region name from a plate's letter prefix."""
    prefix_match = re.match(r"^([A-Z]{1,2})", str(plate).strip().upper())
    if not prefix_match:
        return UNKNOWN_REGION
    return INDONESIAN_REGION_CODES.get(prefix_match.group(1), UNKNOWN_REGION)


def _resolve_char_label(cls_id: int, model_names: Any) -> str:
    if isinstance(model_names, dict) and cls_id in model_names:
        label = str(model_names[cls_id]).strip().upper()
        if len(label) == 1 and label.isalnum():
            return label
    return CHAR_CLASS_NAMES.get(cls_id, "")


def process_vehicle_image(
    image_rgb: np.ndarray,
    detector: YOLO,
    recognizer: YOLO,
    det_conf: float = 0.25,
    rec_conf: float = 0.25,
) -> PlateResult:
    """Run 2-stage plate detection, character recognition, and TNKB post-processing on CPU."""
    annotated_image = image_rgb.copy()
    img_h, img_w = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Stage 1: Plate detection
    det_results = detector.predict(image_bgr, conf=det_conf, device="cpu", verbose=False)
    det_boxes = det_results[0].boxes if det_results else None
    if det_boxes is None or len(det_boxes) == 0:
        return PlateResult(
            raw_text="",
            formatted_plate="",
            region=UNKNOWN_REGION,
            is_valid_format=False,
            confidence=0.0,
            annotated_image=annotated_image,
            cropped_plate=None,
            plate_detected=False,
        )

    confs = det_boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    plate_conf = float(confs[best_idx])
    x1, y1, x2, y2 = det_boxes.xyxy[best_idx].cpu().numpy().tolist()

    # Draw green bounding box and confidence score
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(img_w - 1, int(round(x2))), min(img_h - 1, int(round(y2)))
    thickness = max(2, int(round(min(img_h, img_w) / 250)))
    cv2.rectangle(annotated_image, (ix1, iy1), (ix2, iy2), (0, 220, 0), thickness)
    cv2.putText(
        annotated_image,
        f"Plat {plate_conf:.2f}",
        (ix1, max(20, iy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, thickness * 0.25),
        (0, 220, 0),
        thickness,
        cv2.LINE_AA,
    )

    # Crop plate ROI with 5% padding
    pad_x = int(round(0.05 * max(1.0, x2 - x1)))
    pad_y = int(round(0.05 * max(1.0, y2 - y1)))
    cx1, cy1 = max(0, int(math.floor(x1)) - pad_x), max(0, int(math.floor(y1)) - pad_y)
    cx2, cy2 = min(img_w, int(math.ceil(x2)) + pad_x), min(img_h, int(math.ceil(y2)) + pad_y)
    cropped_plate = image_rgb[cy1:cy2, cx1:cx2].copy()
    if cropped_plate.size == 0:
        return PlateResult(
            raw_text="",
            formatted_plate="",
            region=UNKNOWN_REGION,
            is_valid_format=False,
            confidence=round(plate_conf, 4),
            annotated_image=annotated_image,
            cropped_plate=None,
            plate_detected=True,
        )

    # Stage 2: Character recognition
    cropped_bgr = cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2BGR)
    rec_results = recognizer.predict(cropped_bgr, conf=rec_conf, device="cpu", verbose=False)
    rec_result = rec_results[0] if rec_results else None
    rec_boxes = rec_result.boxes if rec_result is not None else None

    # Keep only characters whose horizontal center lies inside the Stage 1 plate box
    inner_left = x1 - cx1
    inner_right = x2 - cx1
    char_detections: list[dict[str, Any]] = []
    if rec_boxes is not None and len(rec_boxes) > 0:
        xywh_list = rec_boxes.xywh.cpu().numpy()
        cls_list = rec_boxes.cls.cpu().numpy().astype(int)
        conf_list = rec_boxes.conf.cpu().numpy()
        for xywh, cls_id, char_conf in zip(xywh_list, cls_list, conf_list):
            char_label = _resolve_char_label(int(cls_id), rec_result.names)
            char_x_center = float(xywh[0])
            if char_label and inner_left <= char_x_center <= inner_right:
                char_detections.append(
                    {
                        "char": char_label,
                        "conf": float(char_conf),
                        "x_c": char_x_center,
                        "y_c": float(xywh[1]),
                        "w": float(xywh[2]),
                        "h": float(xywh[3]),
                    }
                )

    # 4-step deterministic post-processing
    deduped = deduplicate_char_boxes(char_detections)
    main_row = filter_main_plate_row(deduped, cropped_plate.shape[0])
    raw_text = "".join(item["char"] for item in main_row)
    formatted_plate, is_valid_format = format_indonesian_plate(raw_text)
    region = get_plate_region(formatted_plate if is_valid_format else raw_text)

    combined_conf = (
        round((plate_conf + sum(item["conf"] for item in main_row) / len(main_row)) / 2.0, 4)
        if main_row
        else 0.0
    )

    return PlateResult(
        raw_text=raw_text,
        formatted_plate=formatted_plate,
        region=region,
        is_valid_format=is_valid_format,
        confidence=combined_conf,
        annotated_image=annotated_image,
        cropped_plate=cropped_plate,
        plate_detected=True,
    )
