# Laporan Validasi & Audit Akhir — Aplikasi Pencatatan Nomor Kendaraan (Indonesian ANPR)

**Tanggal Audit:** 2026-11-25  
**Cakupan:** Perbaikan terfokus & sinkronisasi kontrak dengan mempertahankan seluruh alur fitur operasional (`auto check-in`, tab `Validasi Keluar` terpisah, foto bukti check-in di `data/captures/`, dan tombol `Bersihkan Log`).

---

## 1. Ringkasan Perubahan yang Diterapkan

1. **Penyederhanaan Input Kamera & Tab (`app.py`)**
   - Mengganti `st.radio` beserta tombol `Nyalakan Kamera` / `Matikan Kamera` dan state `camera_active` / `pending_camera_*` menjadi `st.tabs(["Unggah Foto", "Ambil dari Kamera"])` dengan `st.camera_input` aktif langsung.
   - Mempertahankan penuh alur auto check-in berpelindung hash MD5, koreksi cepat in-place, tab `Validasi Keluar` dengan preview foto & estimasi durasi, serta tombol `Bersihkan Log` dengan konfirmasi popover.

2. **Standar Bahasa (100% UI Indonesia, 100% Kode Inggris)**
   - Mengubah `"Upload Foto"` menjadi `"Unggah Foto"` di `app.py`.
   - Mengubah label bounding box anotasi gambar dari `"Plate ..."` menjadi `"Plat ..."` di `plate_pipeline.py`.
   - Menambahkan docstring bahasa Inggris singkat (1–2 baris) pada seluruh fungsi publik di `app.py`, `plate_pipeline.py`, `storage.py`, dan `benchmark.py`.

3. **Perbaikan Bug & Kerapian Repositori**
   - Memperbaiki logika fallback regex pada `format_indonesian_plate()` di `plate_pipeline.py` dengan mensyaratkan panjang minimal 3 karakter serta mengandung minimal 1 huruf dan 1 angka sebelum koreksi OCR posisional dijalankan, sehingga input tidak lengkap seperti `"1234"` dan `"AB"` wajib mengembalikan `is_valid=False`.
   - Menyatukan konstanta `UNKNOWN_REGION = "Tidak Diketahui"` di `storage.py` sebagai Single Source of Truth (diimpor oleh `plate_pipeline.py` dan `app.py`).
   - Membuat `.gitignore` (mengabaikan `.ultralytics/`, `__pycache__/`, `*.py[cod]`, `.venv/`, `venv/`, serta foto sementara di `data/captures/`).
   - Membersihkan data uji dummy (`LOG-0001` pada `data/vehicle_logs.csv` dan `data/captures/LOG-0001.jpg`).
   - Mem-pin versi seluruh dependensi CPU di `requirements.txt` (`streamlit==1.67.0`, `ultralytics==8.4.215`, `onnxruntime==1.32.0`, `huggingface-hub==2.2.0`, `opencv-python-headless==4.9.0.80`, `pandas==3.1.1`, `Pillow==13.0.0`).
   - Membiarkan `data/samples/` tanpa gambar sintetis agar siap diisi foto nyata dari dataset Kaggle.

4. **Sinkronisasi Dokumentasi**
   - Memperbarui `README.md` agar selaras dengan tab `Unggah Foto` / `Ambil dari Kamera`, auto check-in, tab `Validasi Keluar`, foto bukti `data/captures/`, dan fitur `Bersihkan Log`.
   - Menambahkan bagian penyesuaian desain (*Approved Design Adjustments*) pada `project_specification_indonesian_anpr.md` agar kontrak spesifikasi sinkron 100% dengan implementasi.

---

## 2. Hasil Verifikasi & Audit Otomatis

| Pemeriksaan | Metode Uji | Hasil |
| :--- | :--- | :--- |
| **Sintaks & Kompilasi Modul** | `python -m py_compile storage.py app.py plate_pipeline.py benchmark.py` | **LULUS** (exit code 0) |
| **Regex TNKB Valid & Koreksi OCR** | Uji `"H1234AB"`, `"B 1963 SSI"`, `"B 1"`, `"AA3385AM"`, `"81234AB"`, `"H1234A8"`, `"H1I34AB"` | **LULUS** (seluruhnya terformat rapi & `is_valid=True`) |
| **Regex Kasus Tidak Lengkap** | Uji `"1234"`, `"AB"`, `"123456"`, `"ABCD"`, `"1B"`, `"B"`, `"1"`, `""` | **LULUS** (seluruhnya `is_valid=False`, tidak dipaksa jadi `I 234` / `A 8`) |
| **Penyimpanan Lokal (`storage.py`)** | Uji di `tempfile.TemporaryDirectory` (`check_in_vehicle`, duplikat aktif, resize JPEG ≤ 800 px, `update_record_plate`, `check_out_vehicle`, `clear_logs`) | **LULUS** |
| **Tunggalnya `UNKNOWN_REGION`** | Pemeriksaan referensi objek & teks sumber `plate_pipeline.py` | **LULUS** (hanya didefinisikan di `storage.py`) |
| **Standar Bahasa UI & Docstring Publik** | Audit AST docstring fungsi publik di 4 file + cek `"Unggah Foto"` & `"Plat ..."` | **LULUS** (100% fungsi publik ber-docstring Inggris, 0 label Inggris di UI) |
| **Kebersihan Workspace & Data Dummy** | Hapus `LOG-0001`, `data/captures/*.jpg`, skrip scratch, dan `__pycache__` | **LULUS** (`data/vehicle_logs.csv` kosong ber-header, `data/captures/` kosong, `data/samples/` tetap tanpa gambar sintetis) |

---

## 3. Status Akhir

Seluruh poin perbaikan, pembersihan repositori, sinkronisasi dokumentasi, dan verifikasi otomatis telah **LULUS 100%**. Aplikasi siap dijalankan langsung di CPU melalui:

```powershell
streamlit run app.py
```
