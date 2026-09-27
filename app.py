# -*- coding: utf-8 -*-

import base64
import io
import os
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import Image

from database import save_report
from thermal_logic import (
    ThermalAnalysisError,
    get_station_from_filename,
    process_image,
)


st.set_page_config(
    page_title="KRC Thermal Fault Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_logo_base64():
    path = os.path.join(
        os.path.dirname(__file__),
        "konkan_logo.png",
    )

    try:
        with open(path, "rb") as file:
            return base64.b64encode(
                file.read()
            ).decode()
    except Exception:
        return None


logo = get_logo_base64()

if logo:
    logo_html = (
        '<img src="data:image/png;base64,'
        f'{logo}" style="width:72px;height:72px;'
        'object-fit:contain;">'
    )
else:
    logo_html = (
        '<div class="logo-placeholder">KRCL</div>'
    )


st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp {
    background:white;
}

.block-container {
    max-width:1000px;
    padding-top:1rem;
}

.krc-header {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:18px;
    padding:14px;
    border-bottom:2px solid #3a007a;
}

.krc-org {
    font-size:25px;
    font-weight:900;
    color:#00008b;
    font-family:Arial,sans-serif;
    text-align:center;
}

.logo-placeholder {
    width:72px;
    height:72px;
    display:flex;
    align-items:center;
    justify-content:center;
    border:1px solid #aaa;
    color:#00008b;
    font-weight:bold;
}

.krc-subtitle {
    text-align:center;
    padding:10px;
    font-weight:700;
    border-bottom:1px solid #ddd;
}

.info-badge,
.station-badge {
    background:#f5f8ff;
    border:1px solid #c0caee;
    border-radius:5px;
    padding:9px 14px;
    margin:9px 0;
}

.metric-row {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:15px;
    margin:12px 0;
}

.metric-key {
    min-width:220px;
    text-align:right;
    font-weight:700;
}

.metric-val {
    min-width:190px;
    padding:7px 18px;
    border-radius:5px;
    text-align:center;
    color:white;
    font-weight:700;
}

.val-blue {background:#00008b;}
.val-green {background:#1a7a1a;}
.val-yellow {background:#b8860b;}
.val-red {background:#cc0000;}

.attend-line {
    text-align:center;
    font-size:19px;
    font-weight:700;
    margin-top:20px;
    padding-top:15px;
    border-top:1px solid #ddd;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""
<div class="krc-header">
    {logo_html}
    <div class="krc-org">
        Konkan Railway Corporation Limited
    </div>
</div>
<div class="krc-subtitle">
    AI-assisted OHE Wire Fault Detection System
</div>
""",
    unsafe_allow_html=True,
)


st.markdown("### Upload thermal inspection image")

uploaded_file = st.file_uploader(
    "Thermal image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(uploaded_bytes)
    ).convert("RGB")

    st.image(
        image,
        caption=uploaded_file.name,
        width="stretch",
    )

    basename = os.path.splitext(
        uploaded_file.name
    )[0]

    parts = basename.split("-")

    extracted_date = "Unknown"
    extracted_time = "Unknown"

    if len(parts) >= 2:
        try:
            image_datetime = datetime.strptime(
                parts[0] + parts[1],
                "%Y%m%d%H%M%S",
            )

            extracted_date = (
                image_datetime.strftime("%d/%m/%Y")
            )

            extracted_time = (
                image_datetime.strftime("%H:%M:%S")
            )

        except ValueError:
            pass

    st.markdown(
        f"""
<div class="info-badge">
    Date: <b>{extracted_date}</b>
    &nbsp; | &nbsp;
    Time: <b>{extracted_time}</b>
</div>
""",
        unsafe_allow_html=True,
    )

    station = get_station_from_filename(
        uploaded_file.name
    )

    if station:
        st.markdown(
            f"""
<div class="station-badge">
    Section: <b>{station["section"]}</b>
    &nbsp; | &nbsp;
    OHE Mast: <b>{station["ohe_mast"]}</b>
    &nbsp; | &nbsp;
    Matched: <b>{station["matched_time"]}</b>
    (±{station["diff_seconds"]}s)
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No Google Sheet station/mast match was found within "
            "5 minutes. Location will be saved as Unknown."
        )

    analyse_clicked = st.button(
        "Analyse Thermal Image",
        type="primary",
        width="stretch",
    )

    if analyse_clicked:

        image_path = None

        try:
            suffix = (
                os.path.splitext(
                    uploaded_file.name
                )[1].lower()
                or ".jpg"
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_bytes
                )

                image_path = temp_file.name

            with st.spinner(
                "Gemini is identifying the OHE region and "
                "the thermal engine is calculating temperatures..."
            ):

                result = process_image(
                    image_path
                )

            status = result["status"]

            if status == "CRITICAL":
                colour_class = "val-red"
            elif status in (
                "WARNING",
                "MONITOR",
            ):
                colour_class = "val-yellow"
            else:
                colour_class = "val-green"

            st.markdown(
                "<hr>"
                '<div style="text-align:center;'
                'font-size:18px;font-weight:900;'
                'text-decoration:underline;">'
                "THERMAL INSPECTION REPORT"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
<div class="metric-row">
    <span class="metric-key">Scale Maximum:</span>
    <span class="metric-val val-blue">
        {result["scale_t_max"]:.1f} °C
    </span>
</div>

<div class="metric-row">
    <span class="metric-key">Scale Minimum:</span>
    <span class="metric-val val-blue">
        {result["scale_t_min"]:.1f} °C
    </span>
</div>

<div class="metric-row">
    <span class="metric-key">Wire Maximum:</span>
    <span class="metric-val val-blue">
        {result["max_temp"]:.1f} °C
    </span>
</div>

<div class="metric-row">
    <span class="metric-key">Wire Minimum:</span>
    <span class="metric-val val-blue">
        {result["min_temp"]:.1f} °C
    </span>
</div>

<div class="metric-row">
    <span class="metric-key">Temperature Difference:</span>
    <span class="metric-val {colour_class}">
        {result["delta"]:.1f} °C
    </span>
</div>

<div class="metric-row">
    <span class="metric-key">Status:</span>
    <span class="metric-val {colour_class}">
        {status}
    </span>
</div>

<div class="attend-line">
    {result["attend_in"]}
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander(
                "AI validation details"
            ):

                st.write(
                    "Detected components:",
                    result["ai_info"].get(
                        "labels",
                        [],
                    ),
                )

                st.write(
                    "Valid thermal pixels:",
                    result["valid_pixel_count"],
                )

                st.write(
                    "Gemini model:",
                    result["ai_info"].get(
                        "model"
                    ),
                )

                st.image(
                    result["wire_mask"] * 255,
                    caption=(
                        "AI OHE segmentation mask"
                    ),
                    clamp=True,
                    width="stretch",
                )

            # Save only after successful analysis.
            india_time = ZoneInfo(
                "Asia/Kolkata"
            )

            section = (
                station["section"]
                if station
                else "Unknown"
            )

            ohe_mast = (
                station["ohe_mast"]
                if station
                else "Unknown"
            )

            try:
                save_report(
                    {
                        "timestamp": datetime.now(
                            india_time
                        ).isoformat(),

                        "image_name":
                            uploaded_file.name,

                        "capture_date":
                            extracted_date,

                        "capture_time":
                            extracted_time,

                        "section":
                            section,

                        "ohe_mast":
                            ohe_mast,

                        "scale_max":
                            result["scale_t_max"],

                        "scale_min":
                            result["scale_t_min"],

                        "wire_max":
                            result["max_temp"],

                        "wire_min":
                            result["min_temp"],

                        "delta_t":
                            result["delta"],

                        "status":
                            result["status"],

                        "attend_in":
                            result["attend_in"],
                    }
                )

                st.success(
                    "Report saved to Supabase."
                )

            except Exception as database_error:
                st.error(
                    "Thermal analysis succeeded, but the "
                    "database save failed: "
                    f"{database_error}"
                )

        except ThermalAnalysisError as analysis_error:

            st.error(
                f"Analysis stopped safely: "
                f"{analysis_error}"
            )

            st.info(
                "No temperature report was generated because "
                "the thermal scale or OHE region could not be "
                "validated."
            )

        except Exception as unexpected_error:

            st.error(
                f"Unexpected analysis error: "
                f"{unexpected_error}"
            )

        finally:

            if (
                image_path
                and os.path.exists(image_path)
            ):
                try:
                    os.remove(image_path)
                except OSError:
                    pass
