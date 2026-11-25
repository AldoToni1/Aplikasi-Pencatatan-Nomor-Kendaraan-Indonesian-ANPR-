from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from plate_pipeline import (
    PlateResult,
    format_indonesian_plate,
    get_plate_region,
    load_models,
    process_vehicle_image,
)
from storage import (
    STATUS_ALL,
    STATUS_CHECK_IN,
    STATUS_CHECK_OUT,
    UNKNOWN_REGION,
    check_in_vehicle,
    check_out_vehicle,
    clear_logs,
    compute_elapsed_minutes,
    filter_logs,
    format_duration,
    get_active_vehicles,
    init_storage,
    load_logs,
    update_record_plate,
)

DISPLAY_COLUMN_HEADERS: dict[str, str] = {
    "record_id": "ID Catatan",
    "plate_number": "Nomor Plat",
    "region": "Asal Wilayah",
    "check_in_time": "Waktu Masuk",
    "check_out_time": "Waktu Keluar",
    "duration_display": "Durasi",
    "status_display": "Status",
    "confidence_pct": "Skor Kepercayaan",
}


@st.cache_resource(show_spinner="Memuat model deteksi dan pengenalan plat nomor (CPU)...")
def get_cached_models() -> tuple[Any, Any]:
    """Load and cache the 2-stage YOLO11n plate detection and recognition models."""
    return load_models()


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "camera_nonce": 0,
        "upload_nonce": 0,
        "checkout_select_nonce": 0,
        "last_upload_hash": None,
        "last_camera_hash": None,
        "processed_image_hash": None,
        "last_image_bytes": None,
        "last_result": None,
        "last_record_id": None,
        "last_entry_state": None,
        "last_record_info": None,
        "correction_input": "",
        "pending_correction_input": None,
        "correction_feedback": None,
        "checkout_feedback": None,
        "clear_logs_feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_check_in_session_state() -> None:
    st.session_state["last_upload_hash"] = None
    st.session_state["last_camera_hash"] = None
    st.session_state["processed_image_hash"] = None
    st.session_state["last_image_bytes"] = None
    st.session_state["last_result"] = None
    st.session_state["last_record_id"] = None
    st.session_state["last_entry_state"] = None
    st.session_state["last_record_info"] = None
    st.session_state["pending_correction_input"] = ""
    st.session_state["correction_feedback"] = None
    st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
    st.session_state["camera_nonce"] = st.session_state.get("camera_nonce", 0) + 1
    st.session_state["checkout_select_nonce"] = st.session_state.get("checkout_select_nonce", 0) + 1


def _process_incoming_vehicle_image(image_bytes: bytes, image_hash: str) -> None:
    detector, recognizer = get_cached_models()
    image_rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    with st.spinner("Mendeteksi dan membaca plat nomor kendaraan..."):
        result = process_vehicle_image(image_rgb, detector, recognizer)

    st.session_state["processed_image_hash"] = image_hash
    st.session_state["last_image_bytes"] = image_bytes
    st.session_state["last_result"] = result
    st.session_state["correction_input"] = result.formatted_plate
    st.session_state["correction_feedback"] = None

    if result.formatted_plate:
        outcome = check_in_vehicle(
            plate_number=result.formatted_plate,
            region=result.region,
            confidence=result.confidence,
            image_bytes=image_bytes,
        )
        if outcome["recorded"]:
            st.session_state["last_record_id"] = outcome["record"]["record_id"]
            st.session_state["last_entry_state"] = "recorded"
            st.session_state["last_record_info"] = outcome["record"]
        else:
            st.session_state["last_record_id"] = None
            st.session_state["last_entry_state"] = "duplicate"
            st.session_state["last_record_info"] = outcome["record"]
    else:
        st.session_state["last_record_id"] = None
        st.session_state["last_entry_state"] = "undetected"
        st.session_state["last_record_info"] = None


def _render_check_in_tab() -> None:
    incoming_bytes: bytes | None = None
    incoming_hash: str | None = None

    input_tab_upload, input_tab_camera = st.tabs(["Unggah Foto", "Ambil dari Kamera"])

    with input_tab_upload:
        uploaded_file = st.file_uploader(
            "Unggah foto kendaraan (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"file_uploader_{st.session_state['upload_nonce']}",
        )

    with input_tab_camera:
        camera_file = st.camera_input(
            "Ambil foto kendaraan",
            key=f"camera_input_{st.session_state['camera_nonce']}",
        )

    upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    upload_hash = hashlib.md5(upload_bytes).hexdigest() if upload_bytes else None
    camera_bytes = camera_file.getvalue() if camera_file is not None else None
    camera_hash = hashlib.md5(camera_bytes).hexdigest() if camera_bytes else None

    if camera_hash is not None and camera_hash != st.session_state.get("last_camera_hash"):
        st.session_state["last_camera_hash"] = camera_hash
        incoming_bytes = camera_bytes
        incoming_hash = camera_hash
    elif upload_hash is not None and upload_hash != st.session_state.get("last_upload_hash"):
        st.session_state["last_upload_hash"] = upload_hash
        incoming_bytes = upload_bytes
        incoming_hash = upload_hash

    if uploaded_file is None:
        prev_upload_hash = st.session_state.get("last_upload_hash")
        if prev_upload_hash is not None:
            st.session_state["last_upload_hash"] = None
            if st.session_state.get("processed_image_hash") == prev_upload_hash:
                st.session_state["processed_image_hash"] = None

    if camera_file is None:
        prev_camera_hash = st.session_state.get("last_camera_hash")
        if prev_camera_hash is not None:
            st.session_state["last_camera_hash"] = None
            if st.session_state.get("processed_image_hash") == prev_camera_hash:
                st.session_state["processed_image_hash"] = None

    # Hash-gated inference & auto check-in (runs at most once per new image)
    if (
        incoming_bytes is not None
        and incoming_hash is not None
        and incoming_hash != st.session_state.get("processed_image_hash")
    ):
        _process_incoming_vehicle_image(incoming_bytes, incoming_hash)

    # Apply deferred text input update before widget instantiation
    pending_correction = st.session_state.pop("pending_correction_input", None)
    if pending_correction is not None:
        st.session_state["correction_input"] = pending_correction

    last_result: PlateResult | None = st.session_state.get("last_result")
    if last_result is None:
        st.info("Unggah foto atau ambil gambar dari kamera untuk memulai pencatatan kendaraan masuk.")
        return

    st.divider()
    col_det, col_crop = st.columns(2)
    with col_det:
        st.image(last_result.annotated_image, caption="Deteksi Plat Nomor", width="stretch")
    with col_crop:
        if last_result.cropped_plate is not None:
            st.image(last_result.cropped_plate, caption="Potongan Area Plat", width="stretch")
        else:
            st.info("Potongan area plat tidak tersedia.")

    entry_state = st.session_state.get("last_entry_state")
    record_info = st.session_state.get("last_record_info")
    last_record_id = st.session_state.get("last_record_id")

    if entry_state == "recorded" and record_info:
        st.success(
            f"Kendaraan **{record_info['plate_number']}** ({record_info['region']}) otomatis tercatat **Masuk** "
            f"— ID: `{record_info['record_id']}` | Waktu Masuk: `{record_info['check_in_time']}`."
        )
        _, current_valid = format_indonesian_plate(str(record_info["plate_number"]))
        if not last_result.is_valid_format and not current_valid:
            st.warning(
                "Format plat nomor belum sesuai pola standar TNKB Indonesia. "
                "Mohon periksa atau koreksi nomor plat di bawah."
            )
    elif entry_state == "duplicate" and record_info:
        st.warning(
            f"Kendaraan **{record_info['plate_number']}** sudah berstatus **Masuk** sejak "
            f"{record_info['check_in_time']}. Gunakan tab **Validasi Keluar** untuk memproses keluar, "
            "atau koreksi nomor plat di bawah jika hasil baca AI keliru."
        )
    elif entry_state == "undetected":
        st.warning(
            "Plat nomor belum terbaca otomatis. Masukkan nomor plat secara manual di bawah untuk mencatat masuk."
        )

    correction_feedback = st.session_state.pop("correction_feedback", None)
    if correction_feedback:
        fb_level, fb_message = correction_feedback
        if fb_level == "success":
            st.success(fb_message)
        else:
            st.error(fb_message)

    input_label = (
        "Masukkan Nomor Plat Manual"
        if entry_state == "undetected"
        else "Salah baca? Koreksi Nomor Plat"
    )
    submit_label = "Simpan Koreksi" if last_record_id is not None else "Catat Masuk"

    with st.form("quick_correction_form"):
        col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")
        with col_input:
            corrected_raw = st.text_input(
                input_label,
                key="correction_input",
                placeholder="Contoh: H 1234 AB",
            )
        with col_btn:
            submitted = st.form_submit_button(submit_label, type="primary", width="stretch")

        current_plate_value = str(st.session_state.get("correction_input", "")).strip()
        if current_plate_value:
            preview_formatted, preview_valid = format_indonesian_plate(current_plate_value)
            preview_region = get_plate_region(preview_formatted or current_plate_value)
            validity_label = "Valid (TNKB)" if preview_valid else "Format Perlu Diperiksa"
            st.caption(f"Asal Wilayah: **{preview_region}** | Status Format: **{validity_label}**")
        else:
            st.caption(
                f"Asal Wilayah: **{UNKNOWN_REGION}** (terdeteksi otomatis dari kode huruf awal plat)."
            )

    if submitted:
        raw_entry = corrected_raw.strip()
        if not raw_entry:
            st.error("Nomor plat wajib diisi.")
            return

        normalized_plate, _ = format_indonesian_plate(raw_entry)
        final_plate = normalized_plate if normalized_plate else " ".join(raw_entry.upper().split())
        final_region = get_plate_region(final_plate)

        if last_record_id is not None:
            update_outcome = update_record_plate(last_record_id, final_plate, final_region)
            if update_outcome["updated"]:
                st.session_state["last_record_info"] = update_outcome["record"]
                st.session_state["pending_correction_input"] = final_plate
                st.session_state["correction_feedback"] = (
                    "success",
                    f"Nomor plat pada catatan **{last_record_id}** berhasil diperbarui menjadi "
                    f"**{final_plate}** ({final_region}).",
                )
                st.rerun()
            elif update_outcome["reason"] == "duplicate":
                conflict = update_outcome["record"]
                st.error(
                    f"Gagal menyimpan koreksi: nomor plat **{final_plate}** sudah aktif tercatat "
                    f"(ID `{conflict['record_id']}`, Masuk {conflict['check_in_time']})."
                )
            elif update_outcome["reason"] == "already_checked_out":
                st.error(
                    f"Catatan **{last_record_id}** sudah berstatus Keluar dan tidak dapat diubah lagi."
                )
            else:
                st.error(f"Catatan **{last_record_id}** tidak ditemukan di log.")
        else:
            record_confidence = (
                last_result.confidence
                if (last_result.plate_detected and final_plate == last_result.formatted_plate)
                else 0.0
            )
            checkin_outcome = check_in_vehicle(
                plate_number=final_plate,
                region=final_region,
                confidence=record_confidence,
                image_bytes=st.session_state.get("last_image_bytes"),
            )
            if checkin_outcome["recorded"]:
                saved = checkin_outcome["record"]
                st.session_state["last_record_id"] = saved["record_id"]
                st.session_state["last_entry_state"] = "recorded"
                st.session_state["last_record_info"] = saved
                st.session_state["pending_correction_input"] = final_plate
                st.session_state["correction_feedback"] = (
                    "success",
                    f"Kendaraan **{final_plate}** ({final_region}) berhasil dicatat **Masuk** "
                    f"(ID `{saved['record_id']}`).",
                )
                st.rerun()
            else:
                existing = checkin_outcome["record"]
                st.error(
                    f"Kendaraan **{final_plate}** sudah berstatus **Masuk** sejak "
                    f"{existing['check_in_time']} (ID `{existing['record_id']}`)."
                )


def _render_check_out_tab(logs_df: Any) -> None:
    checkout_feedback = st.session_state.pop("checkout_feedback", None)
    if checkout_feedback:
        st.success(checkout_feedback)

    active_df = get_active_vehicles(logs_df)
    if active_df.empty:
        st.info("Saat ini tidak ada kendaraan yang sedang parkir (berstatus Masuk).")
        return

    active_map: dict[str, dict[str, Any]] = {
        str(row["record_id"]): row for row in active_df.to_dict(orient="records")
    }
    record_ids = list(active_map.keys())

    selected_id = st.selectbox(
        "Pilih atau ketik nomor plat kendaraan yang akan keluar",
        options=record_ids,
        index=None,
        placeholder="Ketik nomor plat (contoh: B 1168 SPW) lalu tekan Enter...",
        key=f"checkout_plate_select_{st.session_state['checkout_select_nonce']}",
        format_func=lambda rid: (
            f"{active_map[rid]['plate_number']} — Masuk {active_map[rid]['check_in_time']} ({rid})"
        ),
    )

    if selected_id is None:
        st.info(
            "Ketik atau pilih nomor plat kendaraan di atas lalu tekan Enter untuk menampilkan foto dan detail validasi keluar."
        )
        return

    selected_row = active_map[selected_id]
    col_photo, col_info = st.columns([1.2, 1])

    with col_photo:
        img_path_str = str(selected_row.get("image_path", "")).strip()
        if img_path_str and Path(img_path_str).is_file():
            st.image(img_path_str, caption="Foto Kendaraan Saat Masuk", width="stretch")
        else:
            st.info("Foto kendaraan saat masuk tidak tersedia untuk catatan lama ini.")

    with col_info:
        elapsed_minutes = compute_elapsed_minutes(str(selected_row["check_in_time"]))
        elapsed_text = format_duration(elapsed_minutes)

        st.markdown(f"### {selected_row['plate_number']}")
        st.markdown(f"**Asal Wilayah:** {selected_row['region']}")
        st.markdown(f"**ID Catatan:** `{selected_row['record_id']}`")
        st.markdown(f"**Waktu Masuk:** {selected_row['check_in_time']}")
        st.markdown(f"**Estimasi Durasi Parkir:** {elapsed_text}")
        st.write("")

        if st.button(
            "Setujui & Catat Keluar",
            type="primary",
            key=f"approve_checkout_{selected_id}",
            width="stretch",
        ):
            outcome = check_out_vehicle(selected_id)
            if outcome["checked_out"]:
                rec = outcome["record"]
                st.session_state["checkout_feedback"] = (
                    f"Kendaraan **{rec['plate_number']}** ({rec['region']}) berhasil dicatat **Keluar** "
                    f"pada {rec['check_out_time']} (Durasi: **{outcome['duration_text']}**)."
                )
                st.session_state["checkout_select_nonce"] = (
                    st.session_state.get("checkout_select_nonce", 0) + 1
                )
                st.rerun()
            else:
                st.error("Kendaraan ini sudah tidak berstatus Masuk.")


def _sort_logs_active_first(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ordered = df.copy()
    ordered["_status_rank"] = ordered["status"].map(
        lambda val: 0 if val == STATUS_CHECK_IN else 1
    )
    ordered = ordered.sort_values(by="_status_rank", ascending=True, kind="mergesort").drop(
        columns=["_status_rank"]
    )
    return ordered.reset_index(drop=True)


def _format_confidence_pct(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(num):
        return 0.0
    return round(max(0.0, min(1.0, num)) * 100.0, 1)


def _format_duration_cell(row: pd.Series) -> str:
    if row["status"] == STATUS_CHECK_IN:
        return "-"
    val = row.get("duration_minutes", "")
    if pd.isna(val):
        return "-"
    raw_minutes = str(val).strip()
    if not raw_minutes:
        return "-"
    try:
        minutes = float(raw_minutes)
    except (TypeError, ValueError):
        return "-"
    return format_duration(minutes)


def _build_display_logs(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame(columns=list(DISPLAY_COLUMN_HEADERS.values()))

    display = filtered_df.copy()
    display["check_out_time"] = display["check_out_time"].apply(
        lambda val: str(val).strip() if pd.notna(val) and str(val).strip() else "-"
    )
    display["duration_display"] = display.apply(_format_duration_cell, axis=1)
    display["status_display"] = display["status"].apply(
        lambda val: "🟢 Masuk" if val == STATUS_CHECK_IN else "⚪ Keluar"
    )
    display["confidence_pct"] = display["confidence"].apply(_format_confidence_pct)

    display_columns = [col for col in DISPLAY_COLUMN_HEADERS if col in display.columns]
    return display[display_columns].rename(columns=DISPLAY_COLUMN_HEADERS)


def _render_logs_section(logs_df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Log Pencatatan & Ekspor")

    clear_feedback = st.session_state.pop("clear_logs_feedback", None)
    if clear_feedback:
        st.success(clear_feedback)

    total_count = len(logs_df)
    inside_count = int((logs_df["status"] == STATUS_CHECK_IN).sum()) if total_count > 0 else 0
    exited_count = int((logs_df["status"] == STATUS_CHECK_OUT).sum()) if total_count > 0 else 0

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Total Catatan", total_count, border=True)
    kpi_col2.metric("Sedang di Lokasi (Masuk)", inside_count, border=True)
    kpi_col3.metric("Sudah Keluar", exited_count, border=True)

    tool_search, tool_status, tool_export, tool_clear = st.columns(
        [2.2, 1.5, 1.0, 1.0],
        vertical_alignment="bottom",
    )
    with tool_search:
        search_query = st.text_input(
            "Cari Nomor Plat (parsial)",
            placeholder="Contoh: 1234 atau H 1234",
        )
    with tool_status:
        selected_status = st.segmented_control(
            "Filter Status",
            options=[STATUS_ALL, STATUS_CHECK_IN, STATUS_CHECK_OUT],
            default=STATUS_ALL,
            selection_mode="single",
        )
    status_filter = selected_status or STATUS_ALL
    filtered_df = _sort_logs_active_first(
        filter_logs(logs_df, query=search_query, status=status_filter)
    )

    with tool_export:
        st.download_button(
            label="Unduh Log (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_logs.csv",
            mime="text/csv",
            width="stretch",
            disabled=filtered_df.empty,
        )

    with tool_clear:
        with st.popover("Bersihkan Log", disabled=(total_count == 0), width="stretch"):
            st.warning(
                "Semua catatan log dan foto kendaraan yang tersimpan akan dihapus permanen. "
                "Unduh CSV terlebih dahulu jika masih diperlukan."
            )
            if st.button(
                "Ya, Hapus Semua Log",
                type="primary",
                width="stretch",
                key="confirm_clear_logs",
            ):
                removed = clear_logs()
                _reset_check_in_session_state()
                st.session_state["clear_logs_feedback"] = (
                    f"Berhasil menghapus {removed} catatan log beserta foto kendaraan."
                )
                st.rerun()

    st.caption(f"Menampilkan **{len(filtered_df)}** dari **{total_count}** catatan.")

    if total_count == 0:
        st.info("Belum ada data pencatatan kendaraan.")
    elif filtered_df.empty:
        st.info("Tidak ada catatan yang sesuai dengan pencarian atau filter status.")
    else:
        display_df = _build_display_logs(filtered_df)
        column_config = {
            "ID Catatan": st.column_config.TextColumn("ID Catatan", width="small"),
            "Nomor Plat": st.column_config.TextColumn("Nomor Plat", width="small"),
            "Asal Wilayah": st.column_config.TextColumn("Asal Wilayah", width="medium"),
            "Waktu Masuk": st.column_config.TextColumn("Waktu Masuk", width="medium"),
            "Waktu Keluar": st.column_config.TextColumn("Waktu Keluar", width="medium"),
            "Durasi": st.column_config.TextColumn("Durasi", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Skor Kepercayaan": st.column_config.ProgressColumn(
                "Skor Kepercayaan",
                format="%.0f%%",
                min_value=0,
                max_value=100,
                width="small",
            ),
        }
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )


def main() -> None:
    """Run the Streamlit vehicle plate logging application."""
    st.set_page_config(page_title="Aplikasi Pencatatan Nomor Kendaraan", layout="wide")
    init_storage()
    _init_session_state()

    st.title("Aplikasi Pencatatan Nomor Kendaraan")
    st.caption(
        "Sistem pencatatan parkir masuk-keluar otomatis dengan deteksi plat nomor Indonesia (TNKB) berbasis CPU."
    )

    selected_mode = st.segmented_control(
        "Mode Operasional",
        options=["Kendaraan Masuk", "Validasi Keluar"],
        default="Kendaraan Masuk",
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
    )
    active_mode = selected_mode or "Kendaraan Masuk"

    with st.container(border=True):
        if active_mode == "Kendaraan Masuk":
            _render_check_in_tab()
            logs_df = load_logs()
        else:
            logs_df = load_logs()
            _render_check_out_tab(logs_df)

    _render_logs_section(logs_df)


if __name__ == "__main__":
    main()
