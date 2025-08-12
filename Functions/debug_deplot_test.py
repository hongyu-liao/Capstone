"""
Debug helper to test DePlot recognition output structure without saving any images.

It will:
- Load images from 'Sample Line Chart/'
- Use Functions.chart_extraction.extract_chart_data_with_deplot to get structured results
- Print a concise JSON-like summary per image (no files generated)

All code comments are written in English.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from .chart_extraction import (
    extract_chart_data_with_deplot,
    parse_deplot_output,
)


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def image_file_to_data_uri(image_path: Path) -> str:
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def summarize_result(result: Dict) -> Dict:
    summary: Dict = {
        "parsing_success": bool(result.get("parsing_success")),
        "data_points_count": result.get("data_points_count", 0),
        "x_axis_label": result.get("x_axis_label"),
        "y_axis_labels": result.get("y_axis_labels"),
        "datasets": {},
    }
    datasets = result.get("datasets", {}) or {}
    for name, points in datasets.items():
        summary["datasets"][name] = points[:5]
    raw_table = (result.get("raw_table") or "")
    if raw_table:
        summary["raw_table_preview"] = raw_table[:500] + ("..." if len(raw_table) > 500 else "")
        robust = parse_deplot_output(raw_table)
        if robust:
            summary["headers"] = robust.get("headers")
            summary["total_points_robust"] = robust.get("total_points")
            if "x_categories" in robust:
                summary["x_categories"] = robust["x_categories"][:10]
    return summary


def main() -> None:
    setup_logging()
    sample_dir = Path("Sample Line Chart")
    if not sample_dir.exists():
        print(f"Sample Line Chart directory not found at: {sample_dir}")
        return

    candidates: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.gif"):
        candidates.extend(sample_dir.glob(ext))
    candidates = sorted(candidates)

    if not candidates:
        print(f"No images found in {sample_dir}")
        return

    print(f"Found {len(candidates)} images in directory: {sample_dir}")

    all_summaries: List[Tuple[str, Dict]] = []
    for idx, img_path in enumerate(candidates, start=1):
        print(f"\n=== Testing image {idx}/{len(candidates)}: {img_path.name} ===")
        try:
            data_uri = image_file_to_data_uri(img_path)
            result = extract_chart_data_with_deplot(data_uri, idx)
            if result is None:
                print("Result: None (parsing failed)")
            else:
                summary = summarize_result(result)
                print(json.dumps(summary, indent=2))
                all_summaries.append((img_path.name, summary))
        except Exception as exc:
            print(f"Error processing {img_path.name}: {exc}")

    print("\n=== Quick overall check ===")
    for name, summary in all_summaries:
        print(f"- {name}: points={summary.get('data_points_count')} datasets={list((summary.get('datasets') or {}).keys())}")


if __name__ == "__main__":
    main()
