"""
Chart data extraction helpers using Google's DePlot (Pix2Struct) model.

All code comments are written in English.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Dict, Optional

from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


logger = logging.getLogger(__name__)


def extract_chart_data_with_deplot(image_uri: str, pic_number: int) -> Optional[Dict]:
    """
    Extract numerical data from chart images using Google's DePlot model.

    Args:
        image_uri: Base64 encoded image data URI.
        pic_number: Picture number for identification.

    Returns:
        Dict with parsed chart data, or None if extraction fails.
    """
    try:
        if not image_uri or not image_uri.startswith("data:image"):
            logger.warning("Invalid image URI format for picture #%s", pic_number)
            return None

        header, encoded = image_uri.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))

        logger.info("Extracting chart data from picture #%s using DePlot...", pic_number)
        processor = Pix2StructProcessor.from_pretrained("google/deplot")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

        prompt = "Generate underlying data table of the figure below:"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        predictions = model.generate(**inputs, max_new_tokens=512)
        table_string = processor.decode(predictions[0], skip_special_tokens=True)

        # Try robust parser (ported from standalone script) first
        robust = parse_deplot_output(table_string)
        if robust and robust.get("total_points", 0) > 0:
            x_values = []
            for points in robust["datasets"].values():
                x_values.extend([x for x, _ in points])
            x_range = [min(x_values), max(x_values)] if x_values else None
            y_ranges = {
                name: [min(y for _, y in points), max(y for _, y in points)]
                for name, points in robust["datasets"].items()
                if points
            }
            result = {
                "chart_type": "line_chart",
                "x_axis_label": robust.get("headers", ["X"])[0] if robust.get("headers") else "X",
                "y_axis_labels": robust.get("headers", [])[1:] if robust.get("headers") else list(robust["datasets"].keys()),
                "datasets": robust["datasets"],
                "data_points_count": robust.get("total_points", 0),
                "x_range": x_range,
                "y_ranges": y_ranges,
                "raw_table": table_string,
                "extraction_method": "google/deplot",
                "parsing_success": True,
            }
            logger.info("Successfully extracted %s data points (robust parser)", result["data_points_count"])
            return result

        # Fallback to existing parser
        parsed_data = parse_deplot_table_output(table_string, pic_number)
        if parsed_data:
            logger.info("Successfully extracted %s data points", parsed_data.get("data_points_count", 0))
            return parsed_data

        # If parsing failed, still return raw table so caller may attempt LLM summarization
        logger.warning("Failed to parse DePlot output for picture #%s; returning raw table for LLM summarization", pic_number)
        return {
            "parsing_success": False,
            "raw_table": table_string,
            "extraction_method": "google/deplot",
        }
    except Exception as exc:
        logger.error("Chart data extraction failed for picture #%s: %s", pic_number, exc)
        return None


def parse_deplot_table_output(table_string: str, pic_number: int) -> Optional[Dict]:
    """
    Parse DePlot output text into a structured representation.
    """
    try:
        # DePlot uses <0x0A> as newline token in outputs frequently
        lines = [line.strip() for line in table_string.split("<0x0A>") if line.strip()]
        if len(lines) < 3:
            return None

        # Find a plausible header line that contains separators and keywords
        header_line = None
        for line in lines:
            if "|" in line and any(
                keyword in line.lower() for keyword in ["trend", "value", "time", "x", "y", "data"]
            ):
                header_line = line
                break
        if not header_line:
            return None

        headers = [h.strip() for h in header_line.split("|") if h.strip()]
        if len(headers) < 2:
            return None

        datasets: Dict[str, list[tuple[float, float]]] = {}
        x_values: list[float] = []

        data_lines = [line for line in lines if "|" in line and line != header_line and "TITLE" not in line.upper()]
        for line in data_lines:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= len(headers):
                x_val = extract_numeric_value(parts[0])
                if x_val is None:
                    continue
                x_values.append(x_val)
                for i, header in enumerate(headers[1:], 1):
                    if i < len(parts):
                        y_val = extract_numeric_value(parts[i])
                        if y_val is None:
                            continue
                        if header not in datasets:
                            datasets[header] = []
                        datasets[header].append((x_val, y_val))

        if not datasets:
            return None

        total_points = sum(len(series) for series in datasets.values())
        return {
            "chart_type": "line_chart",
            "x_axis_label": headers[0] if headers else "X",
            "y_axis_labels": headers[1:],
            "datasets": datasets,
            "data_points_count": total_points,
            "x_range": [min(x_values), max(x_values)] if x_values else None,
            "y_ranges": {
                name: [min(p[1] for p in series), max(p[1] for p in series)] for name, series in datasets.items() if series
            },
            "raw_table": table_string,
            "extraction_method": "google/deplot",
            "parsing_success": True,
        }
    except Exception as exc:
        logger.error("Failed to parse DePlot table: %s", exc)
        return None


def parse_deplot_output(table_string: str) -> Optional[Dict]:
    """
    Robust parser adapted to noisy DePlot outputs with inline tokens like '<0x0A>' and stray 'TITLE' cells.
    Builds datasets per column where the first column is treated as X (numeric if possible, otherwise index).

    Returns a dict with keys: headers (list[str]), datasets (dict[str, list[(x,y)]]), total_points (int),
    and optionally x_categories (list[str]) if X is categorical.
    """
    try:
        if not table_string:
            return None

        # Normalize into lines using the explicit newline token that DePlot often emits
        raw_lines = [seg.strip() for seg in table_string.split('<0x0A>') if seg.strip()]
        if not raw_lines:
            return None

        # Tokenize each line by '|', scrub empty tokens and noise
        def tokenize(line: str) -> list[str]:
            parts = [p.strip() for p in line.split('|')]
            # Remove empty tokens and obvious noise markers
            parts = [p for p in parts if p and p.upper() != 'TITLE']
            return parts

        token_lines: list[list[str]] = [tokenize(line) for line in raw_lines]
        token_lines = [row for row in token_lines if len(row) >= 2]
        if not token_lines:
            return None

        # Choose header row: first line with >=2 columns and at least one numeric-looking value in subsequent rows
        header_idx = 0
        headers = token_lines[header_idx]
        if len(headers) < 2:
            return None
        # If data rows have more columns than header names, extend headers with generic names
        try:
            max_cols = max(len(row) for row in token_lines)
            if len(headers) < max_cols:
                headers = headers + [f"col_{i}" for i in range(len(headers), max_cols)]
        except Exception:
            pass

        # If headers still contain obvious numeric-only strings, accept but will handle downstream
        # Build datasets
        datasets: Dict[str, list[tuple[float, float]]] = {name: [] for name in headers[1:]}

        # Track whether X is numeric; if not, fallback to index and record categories
        x_is_numeric = True
        x_categories: list[str] = []

        for row_idx, row in enumerate(token_lines[header_idx + 1 :], start=0):
            if len(row) < 2:
                continue
            # Pad row to header length to avoid index errors
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))

            x_cell = row[0]
            try:
                x_val = float(re.sub(r'[^\d.-]', '', x_cell)) if re.search(r'[\d]', x_cell) else None
            except Exception:
                x_val = None
            if x_val is None:
                x_is_numeric = False
                x_categories.append(x_cell)
                x_val = float(len(x_categories) - 1)

            # Parse Y values for each dataset column
            for col_idx, name in enumerate(headers[1:], start=1):
                cell = row[col_idx] if col_idx < len(row) else ''
                try:
                    y_val = float(re.sub(r'[^\d.-]', '', cell)) if re.search(r'[\d]', cell) else None
                except Exception:
                    y_val = None
                if y_val is None:
                    continue
                datasets.setdefault(name, []).append((x_val, y_val))

        # Remove empty datasets
        datasets = {k: v for k, v in datasets.items() if v}
        total_points = sum(len(points) for points in datasets.values())
        if total_points == 0:
            return None

        result: Dict[str, object] = {
            "headers": headers,
            "datasets": datasets,
            "total_points": total_points,
        }
        if not x_is_numeric and x_categories:
            result["x_categories"] = x_categories
        return result
    except Exception as exc:
        logger.error("Robust parse failed: %s", exc)
        return None


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract a numeric value from arbitrary cell text.
    """
    try:
        cleaned = re.sub(r"[^\d.-]", "", text)
        if cleaned:
            return float(cleaned)
    except Exception:
        return None
    return None


__all__ = [
    "extract_chart_data_with_deplot",
    "parse_deplot_table_output",
    "parse_deplot_output",
    "extract_numeric_value",
]


