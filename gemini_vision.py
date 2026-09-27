# -*- coding: utf-8 -*-
"""
Gemini Vision is used ONLY to identify/segment physical railway OHE
components. It does NOT calculate temperatures.
"""

import os
from typing import List

import cv2
import numpy as np
from google import genai
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


class SegmentationItem(BaseModel):
    box_2d: List[int] = Field(
        description="Bounding box [ymin, xmin, ymax, xmax], normalized 0-1000."
    )
    mask: List[List[int]] = Field(
        description="Polygon [x,y] normalized 0-1000 inside the bounding box."
    )
    label: str = Field(description="Description of the detected object.")


class SegmentationResponse(BaseModel):
    boxes: List[SegmentationItem]


PROMPT = """
You are analyzing a railway OHE thermal inspection image.

Segment ONLY the physical railway overhead electrical components:

- contact wire
- jumper wire
- OHE connection
- electrical junction
- conducting wire
- clamp/connector physically part of the OHE connection

The output mask will be used by Python to select pixels and calculate
temperatures from the image's own thermal colour scale.

DO NOT include:
- sky
- trees
- buildings
- ground
- background
- thermal colour scale
- temperature labels or numbers
- P1/P2 labels
- date/time text
- camera UI
- battery/status icons
- borders
- unrelated objects

Do NOT calculate, estimate, infer, or report any temperature.

Return segmentation polygons for all visible OHE conducting
components/junctions. Use the exact JSON schema provided.
"""


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Streamlit secrets."
        )
    return genai.Client(api_key=api_key)


def _polygon_to_mask(item, height, width):
    if len(item.box_2d) != 4 or len(item.mask) < 3:
        return None

    ymin, xmin, ymax, xmax = [int(v) for v in item.box_2d]
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))

    if ymax <= ymin or xmax <= xmin:
        return None

    y1 = int(round(ymin / 1000.0 * height))
    x1 = int(round(xmin / 1000.0 * width))
    y2 = int(round(ymax / 1000.0 * height))
    x2 = int(round(xmax / 1000.0 * width))

    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)

    polygon = []
    for point in item.mask:
        if len(point) != 2:
            continue

        px = int(round(float(point[0]) / 1000.0 * box_w + x1))
        py = int(round(float(point[1]) / 1000.0 * box_h + y1))

        polygon.append([
            max(0, min(width - 1, px)),
            max(0, min(height - 1, py)),
        ])

    if len(polygon) < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask


def detect_ohe_mask(image_path):
    """
    Returns:
        mask: full-resolution uint8 mask containing OHE pixels.
        metadata: model name and accepted labels.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]
    client = _get_client()

    uploaded = client.files.upload(file=image_path)

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {"type": "text", "text": PROMPT},
            {
                "type": "image",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type or "image/jpeg",
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": SegmentationResponse.model_json_schema(),
        },
        generation_config={"thinking_level": "minimal"},
        store=False,
    )

    raw = interaction.output_text
    if not raw:
        raise RuntimeError("Gemini returned an empty segmentation response.")

    try:
        result = SegmentationResponse.model_validate_json(raw)
    except Exception as exc:
        raise RuntimeError(
            f"Gemini returned invalid segmentation JSON: {exc}"
        ) from exc

    combined = np.zeros((height, width), dtype=np.uint8)
    labels = []

    wanted_words = (
        "ohe", "wire", "jumper", "contact", "junction",
        "connector", "clamp", "conduct", "electrical"
    )

    for item in result.boxes:
        label = item.label.strip().lower()

        if not any(word in label for word in wanted_words):
            continue

        component_mask = _polygon_to_mask(item, height, width)
        if component_mask is None:
            continue

        combined = np.maximum(combined, component_mask)
        labels.append(item.label)

    # The right side is reserved for the thermal scale/UI.
    combined[:, int(width * 0.88):] = 0

    if int(combined.sum()) == 0:
        raise RuntimeError(
            "Gemini did not return a usable OHE wire/junction mask."
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined, {
        "model": MODEL_NAME,
        "labels": labels,
        "mask_pixels": int(combined.sum()),
    }
