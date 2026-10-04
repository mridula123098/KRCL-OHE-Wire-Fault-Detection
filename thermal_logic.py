# -*- coding: utf-8 -*-
"""
AI-assisted thermal analysis for railway OHE inspection images.

Gemini identifies the physical OHE region.
Python/OpenCV performs numerical temperature calculation from the
image's own thermal colour scale with enhanced spatial precision.
"""

import os
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from gemini_vision import detect_ohe_mask


if os.name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


class ThermalAnalysisError(RuntimeError):
    pass


def _ocr_number(crop_bgr):
    """Read the numeric part of a thermal scale label with multi-threshold fallback."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    big = cv2.resize(
        crop_bgr,
        (max(1, crop_bgr.shape[1] * 8),
         max(1, crop_bgr.shape[0] * 8)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    candidates = []

    for threshold in (100, 120, 140, 160, 180, 200):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        for psm in (6, 7, 8, 13):
            config = (
                f"--psm {psm} "
                "-c tessedit_char_whitelist=0123456789.-"
            )

            try:
                text = pytesseract.image_to_string(
                    Image.fromarray(binary),
                    config=config,
                ).strip()
            except Exception:
                continue

            for match in re.findall(r"-?\d+(?:\.\d+)?", text):
                try:
                    value = float(match)
                    if -100 <= value <= 200:
                        candidates.append(value)
                except ValueError:
                    pass

    if not candidates:
        return None

    return candidates[0]


def has_minus_sign(crop_bgr):
    """Explicit minus-sign detection via structural morphology and OCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    big = cv2.resize(
        crop_bgr,
        (max(1, crop_bgr.shape[1] * 8),
         max(1, crop_bgr.shape[0] * 8)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    for threshold in (100, 120, 140, 160, 180, 200):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        for psm in (6, 7, 8, 13):
            try:
                text = pytesseract.image_to_string(
                    Image.fromarray(binary),
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=0123456789.-"
                    ),
                )
                if "-" in text:
                    return True
            except Exception:
                pass

    dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, gray.shape[1] // 8), 1),
    )

    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if (
            w >= max(4, gray.shape[1] * 0.12)
            and h <= max(10, gray.shape[0] * 0.18)
        ):
            return True

    return False


def _signed_ocr(crop_bgr):
    value = _ocr_number(crop_bgr)

    if value is None:
        return None

    value = abs(float(value))

    if has_minus_sign(crop_bgr):
        return -value

    return value


def extract_temperature_scale(image_bgr):
    """Extract top and bottom temperatures from the right-side scale."""
    height, width = image_bgr.shape[:2]

    x1 = int(width * 0.86)
    x2 = int(width * 0.995)

    scale = image_bgr[:, x1:x2]

    if scale.size == 0:
        raise ThermalAnalysisError("Temperature scale region is missing.")

    scale_height = scale.shape[0]

    top = scale[
        int(scale_height * 0.10):
        int(scale_height * 0.28)
    ]

    bottom = scale[
        int(scale_height * 0.72):
        int(scale_height * 0.92)
    ]

    t_max = _signed_ocr(top)
    t_min = _signed_ocr(bottom)

    if t_max is None or t_min is None:
        raise ThermalAnalysisError(
            "Could not reliably read both temperature-scale values."
        )

    if t_max <= t_min:
        raise ThermalAnalysisError(
            f"Invalid temperature scale detected: top={t_max}, bottom={t_min}."
        )

    if t_max - t_min > 300:
        raise ThermalAnalysisError(
            f"Unrealistic temperature range: {t_max} to {t_min} °C."
        )

    return scale, float(t_max), float(t_min)


def build_lut(scale, t_max, t_min, n_samples=1024):
    """Build fine-grained high-resolution colour-to-temperature lookup."""
    scale_height = scale.shape[0]

    bar_start = int(scale_height * 0.20)
    bar_end = int(scale_height * 0.80)

    bar = scale[bar_start:bar_end]

    if bar.size == 0:
        raise ThermalAnalysisError("Thermal colour bar could not be extracted.")

    # Trim margin noise along edges of bar strip
    if bar.shape[1] > 4:
        bar = bar[:, 2:-2]

    row_colours = bar.mean(axis=1)

    rows = np.linspace(0, len(row_colours) - 1, n_samples).astype(int)

    colours = row_colours[rows].astype(np.float32)
    temperatures = np.linspace(t_max, t_min, n_samples, dtype=np.float32)

    return colours, temperatures


def map_pixels_to_temperature(image_bgr, scale, t_max, t_min):
    """Map pixels to temperature using distance-weighted sub-pixel interpolation."""
    lut_colours, lut_temperatures = build_lut(
        scale,
        t_max,
        t_min,
        n_samples=1024
    )

    height, width = image_bgr.shape[:2]
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)

    temperature_flat = np.empty(len(pixels), dtype=np.float32)
    distance_flat = np.empty(len(pixels), dtype=np.float32)

    batch_size = 20000

    for start in range(0, len(pixels), batch_size):
        end = min(start + batch_size, len(pixels))
        batch = pixels[start:end]

        difference = batch[:, None, :] - lut_colours[None, :, :]
        distance_squared = np.sum(difference * difference, axis=2)

        nearest = np.argmin(distance_squared, axis=1)

        temperature_flat[start:end] = lut_temperatures[nearest]
        distance_flat[start:end] = np.sqrt(
            distance_squared[np.arange(len(batch)), nearest]
        )

    return (
        temperature_flat.reshape(height, width),
        distance_flat.reshape(height, width),
    )


def _validate_ai_mask(mask, image_shape):
    height, width = image_shape[:2]

    if mask is None or mask.shape != (height, width):
        raise ThermalAnalysisError("AI did not return a valid OHE mask.")

    mask = (mask > 0).astype(np.uint8)

    # Exclude UI sidebars and clock overlays
    mask[:, int(width * 0.88):] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area = int(mask.sum())
    total = height * width
    fraction = area / float(total)

    if area < max(100, int(total * 0.0002)):
        raise ThermalAnalysisError("AI detected too little OHE structure.")

    if fraction > 0.45:
        raise ThermalAnalysisError(
            "AI mask covers too much of the image; segmentation rejected."
        )

    return mask


def classify_delta(delta_t):
    if delta_t > 20:
        return "CRITICAL", "To be attended within 24 hrs"
    if delta_t > 10:
        return "WARNING", "To be attended within 10 days"
    if delta_t > 5:
        return "MONITOR", "To be attended within 1 month"
    return "NORMAL", "Normal — No fault detected"


def compute_wire_temperature(temperature_map, colour_distance, ai_mask):
    """
    Computes precise wire segment max and min temperatures by filtering out
    single-pixel noise artifacts and non-thermal overlay lines.
    """
    mask = _validate_ai_mask(ai_mask, temperature_map.shape)

    valid = (mask == 1) & np.isfinite(temperature_map)

    # Reject non-thermal or overlay pixels with high color deviation
    valid &= colour_distance <= 65.0

    wire_temperatures = temperature_map[valid]

    if wire_temperatures.size < 100:
        raise ThermalAnalysisError(
            "Too few valid thermal pixels remained inside the AI OHE mask."
        )

    # Max Temp: Average top 0.5% hotspot pixels to suppress single bad pixels
    top_k = max(1, int(wire_temperatures.size * 0.005))
    max_temperature = float(np.mean(np.partition(wire_temperatures, -top_k)[-top_k:]))

    # Min Temp: Robust 10th percentile for ambient wire reference
    min_temperature = float(np.percentile(wire_temperatures, 10.0))

    if max_temperature < min_temperature:
        raise ThermalAnalysisError("Temperature statistics are inconsistent.")

    delta = float(max_temperature - min_temperature)
    status, attend_in = classify_delta(delta)

    return {
        "max_temp": max_temperature,
        "min_temp": min_temperature,
        "delta": delta,
        "status": status,
        "attend_in": attend_in,
        "wire_mask": mask,
        "valid_pixel_count": int(wire_temperatures.size),
    }


def _find_col(df, keywords):
    for column in df.columns:
        name = str(column).strip().lower()
        if any(keyword in name for keyword in keywords):
            return column
    return None


def _parse_datetime(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    text = str(value).strip().replace(" UTC", "")
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    return None


def _image_datetime(image_filename):
    basename = os.path.splitext(os.path.basename(image_filename))[0]
    parts = basename.split("-")

    if len(parts) < 2:
        return None

    try:
        return datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def get_station_from_filename(image_filename, sheet_id=None):
    """Match image date/time against the Google Sheet."""
    image_datetime = _image_datetime(image_filename)
    if image_datetime is None:
        return None

    sheet_id = sheet_id or os.getenv(
        "KRCL_SHEET_ID",
        "13W4XDKVK384EfZ5rxtccApLsMkca_Jz22qzz-uyrHf8",
    )

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    try:
        dataframe = pd.read_csv(url)
    except Exception as exc:
        print(f"[station lookup] Google Sheet read failed: {exc}")
        return None

    section_column = _find_col(dataframe, ["section", "station", "name"])
    ohe_column = _find_col(dataframe, ["ohe mast", "ohe_mast", "ohe", "mast"])
    datetime_column = _find_col(dataframe, ["datetime", "date time", "date", "time"])

    if section_column is None or datetime_column is None:
        print("[station lookup] Required columns not found.")
        return None

    dataframe["_parsed_datetime"] = dataframe[datetime_column].apply(_parse_datetime)
    dataframe = dataframe.dropna(subset=["_parsed_datetime"]).copy()

    if dataframe.empty:
        return None

    same_date = dataframe[
        dataframe["_parsed_datetime"].apply(
            lambda value: value.date() == image_datetime.date()
        )
    ].copy()

    if same_date.empty:
        same_date = dataframe.copy()

    same_date["_difference_seconds"] = same_date["_parsed_datetime"].apply(
        lambda value: abs((value - image_datetime).total_seconds())
    )

    nearest = same_date.loc[same_date["_difference_seconds"].idxmin()]
    difference_seconds = int(nearest["_difference_seconds"])

    if difference_seconds > 300:
        return None

    ohe_value = nearest[ohe_column] if ohe_column is not None else "N/A"
    if pd.isna(ohe_value):
        ohe_value = "N/A"

    return {
        "section": str(nearest[section_column]).strip(),
        "ohe_mast": str(ohe_value).strip().replace(".0", ""),
        "matched_time": nearest["_parsed_datetime"].strftime("%H:%M:%S"),
        "diff_seconds": difference_seconds,
    }


def process_image(image_path):
    """
    Full safe pipeline:
    image -> scale OCR -> thermal colour LUT -> Gemini OHE segmentation
          -> local temperature calculation -> delta/status
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ThermalAnalysisError(f"Cannot load image: {image_path}")

    scale, t_max, t_min = extract_temperature_scale(image_bgr)

    temperature_map, colour_distance = map_pixels_to_temperature(
        image_bgr,
        scale,
        t_max,
        t_min,
    )

    ai_mask, ai_info = detect_ohe_mask(image_path)

    statistics = compute_wire_temperature(
        temperature_map,
        colour_distance,
        ai_mask,
    )

    return {
        "scale_t_max": t_max,
        "scale_t_min": t_min,
        "max_temp": round(statistics["max_temp"], 1),
        "min_temp": round(statistics["min_temp"], 1),
        "delta": round(statistics["delta"], 1),
        "status": statistics["status"],
        "attend_in": statistics["attend_in"],
        "temp_map": temperature_map,
        "wire_mask": statistics["wire_mask"],
        "ai_info": ai_info,
        "valid_pixel_count": statistics["valid_pixel_count"],
    }
