# TECHNICAL SPECIFICATION & AI AGENT BRIEF
## Aplikasi Pencatatan Nomor Kendaraan (Indonesian ANPR)
**Lightweight CPU-First Streamlit Application Powered by 2-Stage YOLO11n (`carlravel` / Ultralytics)**

| Specification Field | Details |
| :--- | :--- |
| **Project Name** | Aplikasi Pencatatan Nomor Kendaraan (Indonesian License Plate Logging) |
| **Purpose** | E-Learning Course Material & Industry-Ready Portfolio Project (Streamlit) |
| **Project Team** | Nabil Hibban Hardian, Diassafira MullyanDaru, Muhammad Aldo Toni Saputra |
| **Locked-In ML Stack** | 2-Stage YOLO11n via Ultralytics — `carlravel/indonesian-plate-detector-yolo11` + `carlravel/indonesian-plate-recognizer-yolo11` (Hugging Face Hub, MIT License) |
| **Target Runtime** | CPU-Only Local Execution (No GPU required, Zero Paid APIs / Zero API Keys) |
| **Document Audience** | AI Coding Agent & Course Engineering Team (Single Source of Truth) |

> ### DIRECTIVE FOR AI CODING AGENT (READ FIRST)
> This document is the authoritative specification for implementing the Streamlit application. Follow the exact scope, 2-stage `carlravel` YOLO11n pipeline, post-processing algorithm, flat architecture, and language rules defined below.
>
> **CRITICAL CONSTRAINTS:**
> - Keep code minimal and readable (**KISS & YAGNI**).
> - **DO NOT** add custom CSS styling hacks, multi-page frameworks, authentication, payment systems, live RTSP video streaming, or extra vehicle classification models.
> - Write **ALL** code, file names, variables, function names, docstrings, and comments in **ENGLISH**.
> - Write **ALL** Streamlit user-facing labels, buttons, notifications, and table headers in **INDONESIAN**.

---

## 1. Project Context & E-Learning Engineering Standards

This project serves as a practical, portfolio-ready e-learning module demonstrating how to build an Automatic Number Plate Recognition (ANPR) logging tool for Indonesian vehicles (cars, motorcycles, and other motorized vehicles) using Python and Streamlit. Every architectural decision is governed by five non-negotiable course design standards:

| Standard | Implementation Mandate |
| :--- | :--- |
| **1. Industry Product Alignment** | The workflow must mirror real production/enterprise ALPR products so learners graduate with a credible portfolio piece rather than a toy demo. |
| **2. Focused Scope + Unique Angle** | Keep the feature set tight and domain-specific. Avoid bloated requirements; distinguish the app with practical Indonesian-specific post-processing (expiry row filtering, regex normalization, region lookup, and check-in/check-out dwell time). |
| **3. Internal Benchmark & CPU-First** | Technology selection must be validated against real public datasets. Because learners may not own GPUs or paid cloud accounts, the entire pipeline must run reliably on standard CPU laptops with zero subscription costs. |
| **4. Real Public Data & Ethics** | Use real public datasets containing ground-truth annotations for testing and benchmarking (synthetic data only as a last resort). Respect licensing, legality, and privacy (avoid unconsented personal street captures or blur faces in demo samples). |
| **5. Minimal Code & UX-First** | Prioritize working functionality and clean UX using default Streamlit components. Leave visual UI embellishment to learner creativity. Enforce English-only codebase (Indonesian UI text only), KISS, YAGNI, and Single Source of Truth. |

---

## 2. Industry Product Benchmark & Scope Rationale

To ensure the e-learning project reflects authentic industry workflows, the core user journey is distilled from established global and Indonesian production ALPR systems:

| Reference Product | Segment | Key Production Capabilities | Adopted into E-Learning Scope |
| :--- | :--- | :--- | :--- |
| **Plate Recognizer (Snapshot & Stream)** | Global SaaS / SDK | Snapshot photo upload without GPU, 2-line plate support, region extraction, entry/exit dwell time tracking, searchable log dashboard. | Primary blueprint for our Snapshot upload/camera mode, region parser, and dwell time logging. |
| **Rekor Scout (ex-OpenALPR)** | Global Enterprise | Web dashboard with full and partial license plate search, watchlist status, and structured vehicle history. | Inspires our partial-match plate search filter and clean tabular log view. |
| **PARKEE LPR & Secure Parking** | Indonesia Smart Parking | Ticketless vehicle entry/exit using Indonesian plates (TNKB), timestamp logging, and reducing manual operator input errors. | Validates our Check-In / Check-Out workflow and Human-in-the-Loop operator verification form. |
| **eS-Parking (Fatahsoft)** | Indonesia Smart Parking | ANPR capture paired with automated parking duration reporting and exportable operational logs. | Validates automated duration (dwell time) calculation and CSV report export. |

### 2.1 Strict Scope Boundary (In-Scope vs. Out-of-Scope)

#### IN-SCOPE (MUST BUILD)
- **Flexible Image Input:** Upload photo (`st.file_uploader`) OR capture via webcam/phone (`st.camera_input`).
- **2-Stage Indonesian ANPR Inference:** Plate detection + character recognition for cars, motorcycles, etc.
- **Indonesian Plate Post-Processing:** Filter bottom expiry-date row (e.g., `08.27`), sort left-to-right, regex normalize (e.g., `H 1234 AB`), and auto-detect Indonesian regional origin.
- **Visual Verification:** Side-by-side display of annotated vehicle photo (plate bounding box) and cropped plate ROI.
- **Human-in-the-Loop Correction Form:** Pre-filled `st.text_input` allowing operator quick edits before saving.
- **Automated Check-In / Check-Out & Dwell Time:** First scan logs `Masuk`; subsequent scan of an active plate logs `Keluar` and computes duration.
- **Searchable Log & CSV Export:** Filter by partial plate text or status, summary KPIs, `st.dataframe`, and `st.download_button`.

#### OUT-OF-SCOPE (DO NOT BUILD)
- No live CCTV/RTSP streaming, OpenCV `VideoCapture` loops, or WebRTC.
- No separate vehicle-type (car vs. motorcycle), color, or brand classification model (Stage 1 detects plates directly regardless of vehicle type).
- No user login, authentication, role-based access control, or multi-tenant architecture.
- No payment gateway, parking tariff calculation, QRIS, or hardware/IoT boomgate integration.
- No FastAPI/Flask backend split, Docker setup, Redis, PostgreSQL, or custom CSS/HTML UI hacks.

---

## 3. Locked-In ML Architecture: `carlravel` 2-Stage YOLO11n

The project is locked into **Option 2**: the pre-trained 2-Stage YOLO11n model suite by `carlravel` on Hugging Face Hub. Both models run under a single inference library (`ultralytics`), eliminate heavy OCR dependencies (no PaddleOCR, EasyOCR, or Tesseract), require no API keys, and are specifically trained on Indonesian license plate fonts and layouts (TNKB).

| Pipeline Stage | Hugging Face Repo ID | Model Weight Files | Validated Metrics & License |
| :--- | :--- | :--- | :--- |
| **Stage 1: Plate Detector** | `carlravel/indonesian-plate-detector-yolo11` | `plate_detector_yolo11n.onnx` (primary) or `plate_detector_yolo11n.pt` (fallback) | mAP@0.5: 98.84% \| Precision: 94.13% \| Recall: 95.53% \| License: MIT |
| **Stage 2: Character Recognizer** | `carlravel/indonesian-plate-recognizer-yolo11` | `plate_recognizer_yolo11n.onnx` (~10.6 MB) or `plate_recognizer_yolo11n.pt` (~5.45 MB) | mAP@0.5: 98.40% \| Precision: 96.24% \| Recall: 96.34% \| 36 classes (`0-9`, `A-Z`) \| License: MIT |

### 3.1 Model Downloading, Caching & Format Verification Rules
- **Zero-Config Model Fetching:** Use `huggingface_hub.hf_hub_download(repo_id=..., filename=...)` to automatically download and cache weights locally on first launch. No Hugging Face token is needed because both repos are public.
- **Streamlit Resource Caching:** Wrap model loading inside Streamlit's `@st.cache_resource` so both YOLO11n models are loaded once into memory per session rather than reloaded on every widget interaction.
- **Mandatory Agent Verification Check:** Before finalizing inference code, the coding agent **MUST** inspect the actual filenames in both Hugging Face repos and verify whether Ultralytics reads the 36 class names (`0-9`, `A-Z`) correctly from the `.onnx` metadata. If `.onnx` metadata lacks class names or triggers an ONNX Runtime issue on CPU, seamlessly fall back to the `.pt` weights (only ~5.45 MB each, fully CPU-compatible via Ultralytics) OR load the verified class map dictionary as a constant.
- **Safe CPU Execution:** Force CPU inference cleanly (`device="cpu"`) so laptops with mismatched CUDA drivers never crash.

### 3.2 End-to-End Inference Flow

```text
[Uploaded / Captured Image (Car, Motorcycle, etc.)]
          │
          ▼
[Stage 1: Plate Detector (carlravel YOLO11n)]
  ├─ Detects plate bounding box(es) -> select highest-confidence box
  ├─ Draw bounding box + confidence on preview image
  └─ Crop plate ROI with 5% safe padding margin
          │
          ▼
[Stage 2: Character Recognizer (carlravel YOLO11n)]
  └─ Detects individual alphanumeric character boxes: (x_c, y_c, w, h, conf, label)
          │
          ▼
[Deterministic Post-Processing (~15–25 lines of Python)]
  ├─ 1. Deduplicate overlapping character boxes (horizontal proximity / IoU NMS)
  ├─ 2. Filter out bottom expiry-date row (e.g., "08.27") via y_center & box height
  ├─ 3. Sort remaining main-row characters left-to-right by x_center
  ├─ 4. Normalize & format via Indonesian TNKB Regex (e.g., "H 1234 AB")
  └─ 5. Lookup regional prefix code (e.g., "H" -> "Semarang & Sekitarnya")
          │
          ▼
[Streamlit Human-in-the-Loop Form -> Single Source of Truth (data/vehicle_logs.csv)]
```

---

## 4. Deterministic Post-Processing Specification

Because Stage 2 is an object detector that outputs individual character bounding boxes rather than ordered text, clean post-processing is the most critical algorithmic piece of this project. Indonesian license plates (TNKB) contain two rows: a large primary row (e.g., `H 1234 AB`) and a smaller bottom row indicating plate validity month/year (e.g., `08.27`). The agent **MUST** implement the following four steps cleanly inside `plate_pipeline.py`:

### Step 1: Character Overlap Deduplication (Char-Level NMS)
Sometimes YOLO detects two competing characters on the exact same glyph (e.g., both `'0'` and `'O'` or `'B'` and `'8'` with overlapping boxes). Sort all detected character boxes by confidence descending and discard any lower-confidence box whose IoU with an already accepted box exceeds `0.45` (or whose horizontal/vertical center distance is less than `35%` of the average character width).

### Step 2: Filtering Out the Expiry-Date Row (Bulan/Tahun)
On standard Indonesian plates, main characters occupy the upper ~60–65% of the plate height and have significantly taller bounding boxes than the 4 small expiry digits at the bottom. Use a robust dual geometric rule (height + vertical position) so it works on slightly tilted car and motorcycle plates:

```python
def filter_main_plate_row(detections: list[dict], crop_height: int) -> list[dict]:
    """Keep only main license plate characters and discard bottom expiry-date digits.

    Each item in detections:
        {'char': str, 'conf': float, 'x_c': float, 'y_c': float, 'w': float, 'h': float}
    """
    if not detections:
        return []

    # Reference height from the tallest detected characters (top 3 average)
    heights = sorted([d["h"] for d in detections], reverse=True)
    ref_height = sum(heights[: min(3, len(heights))]) / min(3, len(heights))

    # 1. Remove characters significantly shorter than the main row (expiry digits are ~40-55% height)
    tall_chars = [d for d in detections if d["h"] >= 0.62 * ref_height]
    if not tall_chars:
        tall_chars = detections

    # 2. Compute median vertical center of tall characters and reject lower outliers
    y_centers = sorted([d["y_c"] for d in tall_chars])
    median_y = y_centers[len(y_centers) // 2]

    # Discard detections sitting significantly below the primary row baseline
    main_row = [
        d
        for d in tall_chars
        if (d["y_c"] - median_y) <= (0.55 * ref_height)
        and d["y_c"] <= (0.78 * crop_height)
    ]

    # 3. Sort left-to-right by horizontal center
    return sorted(main_row if main_row else tall_chars, key=lambda d: d["x_c"])
```

### Step 3: Indonesian TNKB Regex Formatting & Positional Normalization
Join the sorted characters into a raw uppercase alphanumeric string (e.g., `H1234AB`). Parse and format it according to standard Indonesian license plate grammar: `[1–2 Prefix Letters] + [1–4 Digits] + [0–3 Suffix Letters]`.
- **Standard Pattern:** Regex pattern `^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$` -> format with single spaces as `f"{prefix} {number} {suffix}".strip()`.
- **Graceful Fallback:** Keep error correction lightweight and safe: if regex fails on raw text due to a single misread digit/letter at a known positional boundary (for instance, `'O'` inside the middle numeric block or `'0'` at the prefix), apply conservative positional mapping (`0<->O`, `1<->I`, `8<->B`, `5<->S`, `2<->Z`) only when needed to match the valid regex. If still unmatched, return the cleaned raw string and flag `is_valid_format = False` so the operator sees a warning in the Human-in-the-Loop form.

### Step 4: Indonesian Regional Prefix Lookup (Unique Value-Add Feature)
Extract the 1- or 2-letter prefix from the formatted plate and map it to its Indonesian region using a clean, static Python dictionary (`INDONESIAN_REGION_CODES`) inside `plate_pipeline.py`. Include all official Indonesian regional codes so any plate tested by learners resolves accurately:

| Region Cluster | Prefix Mapping to Include in `INDONESIAN_REGION_CODES` |
| :--- | :--- |
| **Sumatra** | `BL` (Aceh), `BB`/`BK` (Sumatera Utara), `BA` (Sumatera Barat), `BM` (Riau), `BP` (Kep. Riau), `BH` (Jambi), `BD` (Bengkulu), `BG` (Sumatera Selatan), `BN` (Bangka Belitung), `BE` (Lampung) |
| **DKI, Banten & Jabar** | `A` (Banten), `B` (DKI Jakarta, Depok, Tangerang, Bekasi), `D` (Bandung Raya), `E` (Cirebon/Indramayu/Majalengka/Kuningan), `F` (Bogor/Sukabumi/Cianjur), `T` (Purwakarta/Karawang/Subang), `Z` (Garut/Tasikmalaya/Ciamis/Sumedang) |
| **Jateng & DIY** | `G` (Pekalongan/Tegal/Brebes), `H` (Semarang/Salatiga/Kendal/Demak), `K` (Pati/Kudus/Jepara/Rembang/Blora), `R` (Banyumas/Cilacap/Purbalingga/Banjarnegara), `AA` (Kedu/Magelang/Kebumen/Wonosobo), `AB` (DI Yogyakarta), `AD` (Surakarta/Solo Raya) |
| **Jatim, Bali & Nusa Tenggara** | `L` (Surabaya), `M` (Madura), `N` (Malang/Pasuruan/Probolinggo/Lumajang), `P` (Besuki/Jember/Banyuwangi), `S` (Bojonegoro/Tuban/Lamongan/Mojokerto), `W` (Sidoarjo/Gresik), `AE` (Madiun/Ponorogo/Ngawi), `AG` (Kediri/Blitar/Tulungagung), `DK` (Bali), `DR`/`EA` (NTB), `DH`/`EB`/`ED` (NTT) |
| **Kalimantan, Sulawesi, Maluku & Papua** | `KB` (Kalbar), `DA` (Kalsel), `KH` (Kalteng), `KT` (Kaltim), `KU` (Kaltara), `DB`/`DL` (Sulut), `DM` (Gorontalo), `DN` (Sulteng), `DC` (Sulbar), `DD`/`DP`/`DW` (Sulsel), `DT` (Sultra), `DE` (Maluku), `DG` (Maluku Utara), `PA` (Papua), `PB` (Papua Barat) |

---

## 5. Application UX Flow & Data Persistence Design

The Streamlit interface must prioritize operator ergonomics using standard built-in Streamlit widgets (`st.tabs`, `st.columns`, `st.file_uploader`, `st.camera_input`, `st.form`, `st.metric`, `st.dataframe`, `st.download_button`). Avoid complex CSS styling or visual clutter.

### 5.1 Screen Layout & User Journey (Indonesian UI Labels)

| UI Block | Default Streamlit Implementation & Indonesian Copy |
| :--- | :--- |
| **Header** | Title: `"Aplikasi Pencatatan Nomor Kendaraan"` with concise 1-line subtitle explaining automated Indonesian plate detection and check-in/check-out logging. |
| **Section A: Input Gambar** | Use `st.tabs(["Upload Foto", "Ambil dari Kamera"])` containing `st.file_uploader("Unggah foto kendaraan (JPG/PNG)", type=["jpg", "jpeg", "png"])` and `st.camera_input("Ambil foto kendaraan")`. |
| **Section B: Hasil Deteksi Visual** | Two-column layout (`st.columns(2)`): Left column displays the original image with green plate bounding box & confidence score (`"Deteksi Plat Nomor"`); Right column displays the cropped plate ROI (`"Potongan Area Plat"`). If no plate is detected, display `st.warning("Plat nomor tidak terdeteksi. Anda tetap dapat memasukkan nomor secara manual di bawah.")`. |
| **Section C: Form Verifikasi (Human-in-the-Loop)** | Inside `st.form("record_form")`:<br>• `st.text_input("Nomor Kendaraan", value=detected_plate)` — editable by operator.<br>• Dynamic preview/info of auto-detected region (`"Asal Wilayah"`) and format validity indicator.<br>• Optional `st.text_input("Catatan (Opsional)", placeholder="Contoh: Tamu / Kurir / Penghuni")`.<br>• Submit button: `st.form_submit_button("Catat Kendaraan", type="primary")`.<br>• Smart feedback via `st.success` showing whether the vehicle was logged as `"Masuk"` or `"Keluar"` (with calculated duration). |
| **Section D: Log Pencatatan & Ekspor** | Top KPI row using 3 `st.metric` cards: `"Total Catatan"`, `"Sedang di Lokasi (Masuk)"`, `"Sudah Keluar"`.<br>Filter bar: `st.text_input("Cari Nomor Plat (parsial)")` + `st.selectbox("Filter Status", ["Semua", "Masuk", "Keluar"])`.<br>Table: `st.dataframe` displaying newest records first with Indonesian column headers.<br>Export button: `st.download_button("Unduh Log (CSV)", ...)`. |

### 5.2 Data Schema & Check-In / Check-Out Business Logic (Single Source of Truth)
All vehicle records are stored in a single CSV file at `data/vehicle_logs.csv` and accessed exclusively through `storage.py`. When saving a plate number (always normalized to uppercase trimmed format, e.g., `"H 1234 AB"`):
1. **Lookup Active Session:** Check if there is an existing row for that exact `plate_number` where `status == "Masuk"`.
2. **Check-In Action:** If **NO** active row exists -> append a new row with `check_in_time = now` (`YYYY-MM-DD HH:MM:SS`), `check_out_time = ""`, `duration_minutes = ""`, `status = "Masuk"`.
3. **Check-Out Action:** If an active row **EXISTS** -> update that row with `check_out_time = now`, calculate `duration_minutes = round((check_out - check_in).total_seconds() / 60, 1)` (plus human-readable display string in UI, e.g., `"1 jam 15 menit"` or `"< 1 menit"`), and set `status = "Keluar"`.

| Column Name (CSV) | Data Type | Description |
| :--- | :--- | :--- |
| `record_id` | string (UUID / sequential ID) | Unique identifier for each vehicle visit session (e.g., `LOG-0001`). |
| `plate_number` | string | Normalized Indonesian license plate (e.g., `H 1234 AB`). |
| `region` | string | Resolved Indonesian regional origin (e.g., `Semarang/Salatiga/Kendal/Demak`) or `Tidak Diketahui`. |
| `check_in_time` | string (ISO-like timestamp) | Entry timestamp formatted as `YYYY-MM-DD HH:MM:SS`. |
| `check_out_time` | string (nullable/empty) | Exit timestamp formatted as `YYYY-MM-DD HH:MM:SS` (empty while status is `Masuk`). |
| `duration_minutes` | float (nullable/empty) | Elapsed dwell time in minutes calculated automatically on Check-Out. |
| `status` | string enum | Either `Masuk` (currently inside) or `Keluar` (checked out). |
| `confidence` | float (`0.0` - `1.0`) | Combined detection/recognition confidence score (or `0.0` if entered manually). |
| `notes` | string (optional) | Optional operator remarks (e.g., `Tamu`, `Pengiriman`). |

---

## 6. File Architecture & Module Contracts

Maintain a completely flat, learner-friendly folder structure. Every file has a single clear responsibility so course participants can read the entire codebase in under 15 minutes.

```text
indonesian-anpr-streamlit/
├── app.py                  # Streamlit UI/UX layer ONLY (zero ML/CSV logic inside)
├── plate_pipeline.py       # HF download, Stage 1+2 inference, row filtering, regex & region lookup
├── storage.py              # Single Source of Truth for data/vehicle_logs.csv & check-in/out logic
├── benchmark.py            # Standalone CPU evaluation script (latency + Exact Match on real dataset)
├── requirements.txt        # Pinned minimal CPU-friendly dependencies
├── README.md               # Setup guide, model attribution (MIT), and usage instructions
└── data/
    ├── vehicle_logs.csv    # Auto-created on first run if missing
    └── samples/            # Optional small folder of real demo car/motorcycle test images
```

| Module | Strict Responsibility & Function Contract |
| :--- | :--- |
| `app.py` | Orchestrates Streamlit UI, caches model via `@st.cache_resource`, invokes `plate_pipeline.process_vehicle_image()`, renders preview + correction form, calls `storage.record_vehicle()`, and displays filtered log + CSV export. |
| `plate_pipeline.py` | Exports:<br>• `load_models() -> tuple[YOLO, YOLO]`<br>• `process_vehicle_image(image_rgb: np.ndarray, detector, recognizer) -> PlateResult` (dataclass containing `raw_text`, `formatted_plate`, `region`, `is_valid_format`, `confidence`, `annotated_image`, `cropped_plate`, `plate_detected`)<br>• `format_indonesian_plate(raw_text: str) -> tuple[str, bool]`<br>• `get_plate_region(plate: str) -> str` |
| `storage.py` | Exports:<br>• `init_storage(csv_path) -> Path`<br>• `load_logs(csv_path) -> pd.DataFrame`<br>• `record_vehicle(plate_number: str, region: str, confidence: float, notes: str = "", csv_path=...) -> dict` (returns `action="Masuk"\|"Keluar"`, `duration_text`, `record`)<br>• `filter_logs(df: pd.DataFrame, query: str, status: str) -> pd.DataFrame` |
| `benchmark.py` | CLI script accepting a folder of real test images + ground-truth CSV/labels (e.g., from Kaggle `juanthomaswijaya/indonesian-license-plate-dataset`). Reports: Total Tested, Plate Detection Rate (%), Exact Match Accuracy (%), Avg CPU Inference Time (sec/image), and saves error analysis CSV. |
| `requirements.txt` | Minimal packages: `streamlit`, `ultralytics`, `onnxruntime`, `huggingface_hub`, `opencv-python-headless`, `pandas`, `Pillow`. |

---

## 7. Dataset, Internal Benchmark & Ethics Protocol

In accordance with Standards 3 and 4, `benchmark.py` enables objective CPU validation using real Indonesian vehicle images before publishing the e-learning module:
- **Primary Public Dataset:** Use the public Kaggle dataset *"Indonesian License Plate Dataset"* by Juan Thomas Wijaya (contains real Indonesian car and motorcycle images with plate bounding box and character ground-truth labels).
- **Test Sample Coverage:** Evaluate at least 20–30 mixed real samples covering: black-background plates (older TNKB), white-background plates (newer TNKB), car plates, and motorcycle plates (including slight angles).
- **Benchmark KPIs:** Measure and print: (1) Stage 1 Plate Detection Rate (%), (2) End-to-End Exact Match Accuracy (%) after post-processing, (3) Average CPU Latency per image (seconds), and (4) Top failure cases (e.g., motion blur or severe tilt).
- **Data Legality & Privacy:** Ensure all demo images placed in `data/samples/` come from open-licensed public datasets (e.g., CC BY 4.0 / MIT) and never expose unblurred human faces or private non-public captures. Include proper MIT license attribution for `carlravel`'s Hugging Face models in `README.md`.

---

## 8. Agent Execution & Acceptance Checklist

Before marking implementation complete, the coding agent **MUST** verify every item on this checklist:

- [ ] **1. Model Weights Verified:** Confirm exact filenames in `carlravel/indonesian-plate-detector-yolo11` and `carlravel/indonesian-plate-recognizer-yolo11` on Hugging Face Hub, verify 36-class label mapping (`0-9`, `A-Z`) on CPU, and confirm automatic download via `huggingface_hub` works without any API key.
- [ ] **2. Expiry Row Filtered:** Test with both car and motorcycle Indonesian plates containing bottom expiry dates (e.g., `08.27`) and verify expiry digits are excluded from the final formatted plate string.
- [ ] **3. Regex & Region Parser Works:** Verify raw strings like `"H1234AB"` or `"B9012XYZ"` normalize to `"H 1234 AB"` and `"B 9012 XYZ"` and resolve accurate Indonesian regions.
- [ ] **4. Human-in-the-Loop & Check-In/Out:** Verify operator can edit detected plate text before submitting; scanning the same plate once records `"Masuk"`, scanning a second time records `"Keluar"` with valid dwell duration in `data/vehicle_logs.csv`.
- [ ] **5. Search, Filter & CSV Export:** Verify partial plate search (e.g., typing `"1234"`), status filtering, KPI counters, and CSV download work cleanly without runtime errors.
- [x] **6. Code Standards Audit:** Confirm 100% English codebase (files, functions, variables, comments), 100% Indonesian UI copy, zero custom CSS hacks, zero out-of-scope features, and clean CPU-only execution via `streamlit run app.py`.

---

## 9. Approved Design Adjustments (Synchronized Implementation Contract)

To better match real-world parking gate workflows while staying 100% within the lightweight CPU-first, zero-database Streamlit scope, the following intentional design refinements are officially part of the project contract:

1. **Auto Check-In + Inline In-Place Correction (`Kendaraan Masuk` Tab):** Upon uploading a photo (`Unggah Foto`) or capturing from camera (`Ambil dari Kamera` via direct `st.camera_input`), detected plates are automatically recorded as `Masuk` (with active-plate duplicate prevention). Operators can immediately correct misread plates in-place via `update_record_plate()` without creating duplicate rows or entering a separate notes field.
2. **Dedicated Check-Out Validation (`Validasi Keluar` Tab):** Check-out is separated into its own tab featuring a searchable dropdown of active (`Masuk`) vehicles, real-time parking duration estimate, check-in evidence photo preview, and an explicit approval button (`check_out_vehicle()`).
3. **Local Check-In Evidence Photos (`data/captures/`):** Each check-in optionally stores a resized JPEG evidence photo (max width 800 px) in `data/captures/LOG-xxxx.jpg`, referenced via the `image_path` column in `data/vehicle_logs.csv` (replacing the unused `notes` column while maintaining legacy CSV read compatibility).
4. **Confirmed Log Reset (`Bersihkan Log`):** A popover-confirmed `Bersihkan Log` button invokes `clear_logs()` to reset `data/vehicle_logs.csv` and delete stored capture images in `data/captures/` for clean operational resets and demos.
