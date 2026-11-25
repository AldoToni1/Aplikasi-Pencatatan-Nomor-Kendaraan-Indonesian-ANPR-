# Aplikasi Pencatatan Nomor Kendaraan (Indonesian ANPR)

Lightweight CPU-first Streamlit application for Indonesian license plate (TNKB) detection, character recognition, regional origin lookup, and automated vehicle check-in/check-out logging with dwell time calculation.

Built with a 2-stage YOLO11n pipeline (`carlravel` / Ultralytics) running fully on local CPU — **no GPU and no paid API keys required**.

---

## Features
- **2-Tab Parking Operations:**
  - **Kendaraan Masuk (Check-In):** Switch between `Unggah Foto` (JPG/PNG) and `Ambil dari Kamera` (direct `st.camera_input` tab). Detected plates are automatically checked in as `Masuk` with check-in evidence photo capture, active-plate duplicate protection, and a fast inline plate correction form (in-place update, no manual notes input).
  - **Validasi Keluar (Check-Out Validation):** Search or select an active (`Masuk`) plate from a searchable dropdown, inspect the check-in vehicle photo preview alongside plate, region, entry time, and real-time duration estimate, then click the approval button to record `Keluar` and finalize parking duration.
- **2-Stage Indonesian ANPR (CPU):**
  - Stage 1: License plate detection + green bounding box preview (`Plat ...`) + 5% padded ROI crop.
  - Stage 2: 36-class alphanumeric character recognition (`0-9`, `A-Z`).
- **Indonesian TNKB Post-Processing:**
  - Character-level overlap deduplication (NMS).
  - Automatic expiry-date row filtering (e.g., discards bottom `08.27`).
  - TNKB regex normalization (e.g., `H1234AB` → `H 1234 AB`) with conservative OCR correction that rejects incomplete fragments (e.g., `1234`, `AB`).
  - Automatic Indonesian regional origin lookup (`INDONESIAN_REGION_CODES`).
- **Lightweight Local Storage (Zero External Database):** Persists all parking records in `data/vehicle_logs.csv` and compressed check-in vehicle photos (max width 800 px, JPEG) in `data/captures/` with backwards compatibility for legacy CSV logs.
- **Searchable Log, CSV Export & Log Reset:** Always-visible bottom panel with 3 KPI summary cards, active-first sorting, partial plate search, status segmented filter (`Semua`, `Masuk`, `Keluar`), one-click CSV download, and a confirmed **Bersihkan Log** action that clears logs and stored capture photos.

---

## Project Structure

```text
.
├── app.py                  # Streamlit 2-tab UI/UX layer (Indonesian UI copy, zero ML/CSV logic)
├── plate_pipeline.py       # HF model loader (CPU), Stage 1+2 inference, 4-step post-processing
├── storage.py              # Single Source of Truth for CSV logs, local photo storage & parking state
├── benchmark.py            # Standalone CPU evaluation CLI on real labeled datasets
├── requirements.txt        # Minimal CPU-only dependencies
├── README.md               # Setup, benchmark, ethics, and attribution guide
└── data/
    ├── vehicle_logs.csv    # Auto-created local parking log
    ├── captures/           # Auto-created compressed check-in vehicle photos (LOG-xxxx.jpg)
    └── samples/            # Real public test samples & ground_truth.csv
```

---

## Installation & Running on CPU (No GPU / No API Key)

1. **Create and activate a virtual environment (recommended, Python 3.10+):**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

2. **Install minimal CPU dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   On first launch, both `carlravel` YOLO11n `.onnx` weights (with `.pt` fallback) are automatically downloaded and cached from Hugging Face Hub without any API key, and executed strictly on `device="cpu"`.

---

## Running the CPU Benchmark (`benchmark.py`)

Evaluate Stage 1 plate detection rate, end-to-end exact match accuracy, average CPU inference time, and export an error analysis CSV using real public Indonesian vehicle datasets (recommended: Kaggle [`juanthomaswijaya/indonesian-license-plate-dataset`](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)):

```bash
# Option A: Using a ground-truth CSV (columns: filename, ground_truth)
python benchmark.py \
  --images-dir data/samples \
  --ground-truth data/samples/ground_truth.csv \
  --output-csv data/benchmark_error_analysis.csv

# Option B: Directly using a Kaggle YOLO character labels directory
python benchmark.py \
  --images-dir /path/to/kaggle/images \
  --ground-truth /path/to/kaggle/labels \
  --limit 30 \
  --output-csv data/benchmark_error_analysis.csv
```

---

## Data Ethics, Legality & Privacy
- Use **real public datasets** with open licenses (e.g., CC BY 4.0 / MIT) and ground-truth labels for testing and benchmarking (such as Kaggle `juanthomaswijaya/indonesian-license-plate-dataset`).
- Do **not** include unconsented personal street captures or unblurred human faces in `data/samples/` or published demos.
- Follow local privacy regulations when logging vehicle plate numbers in operational environments.

---

## Model Attribution (MIT License)
This project uses two pre-trained YOLO11n models by **Carleano Ravelza Wongso (`carlravel`)** hosted on Hugging Face Hub under the **MIT License**:
- **Stage 1 (Plate Detector):** [`carlravel/indonesian-plate-detector-yolo11`](https://huggingface.co/carlravel/indonesian-plate-detector-yolo11) — MIT License
- **Stage 2 (Character Recognizer):** [`carlravel/indonesian-plate-recognizer-yolo11`](https://huggingface.co/carlravel/indonesian-plate-recognizer-yolo11) — MIT License
