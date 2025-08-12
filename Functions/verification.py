"""
Verification utilities for final NLP-ready JSON files.

Checks include:
- No residual image payloads present
- AI descriptions inserted and non-empty
- Web search context presence for conceptual images
- Chart data extraction presence for data visualizations

All code comments are written in English.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)


def _has_image_key(obj) -> bool:
    """
    Shallow check for 'image' key in a picture object.
    """
    if isinstance(obj, dict):
        return "image" in obj
    return False


def verify_final_json(
    json_file_path: str,
    *,
    require_no_images: bool = True,
    min_description_length: int = 20,
) -> Dict:
    """
    Verify that the final JSON has no images and contains expected AI-generated content.

    Rules:
    - For any picture with image_type == 'DATA_VISUALIZATION', expect chart_data_extraction.extraction_success == True
    - For any picture with image_type == 'CONCEPTUAL', expect web_context.ai_summary to be present and non-empty
    - For any informative picture, expect a non-empty description

    Args:
        json_file_path: Path to the NLP-ready JSON file.
        require_no_images: Whether to enforce zero images present.
        min_description_length: Minimum length for a description to be considered valid.

    Returns:
        Dict with verification summary, including per-picture issues.
    """
    json_path = Path(json_file_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read JSON for verification: %s", json_path)
        return {"ok": False, "error": "read_failed"}

    pictures = data.get("pictures", [])
    total = len(pictures)

    images_keys_found = 0
    with_ai_analysis = 0
    with_nonempty_description = 0
    with_web_context = 0
    with_chart_data = 0

    images_present_indices: List[int] = []
    per_picture_issues: List[Dict] = []

    for idx, pic in enumerate(pictures, start=1):
        picture_issues: List[str] = []

        if _has_image_key(pic):
            images_keys_found += 1
            images_present_indices.append(idx)
            picture_issues.append("residual_image_key_present")

        ai = pic.get("ai_analysis") if isinstance(pic, dict) else None
        if isinstance(ai, dict):
            with_ai_analysis += 1
            image_type = (ai.get("image_type") or "").strip().upper()
            desc = ai.get("description") or ai.get("detailed_description") or ""

            if isinstance(desc, str) and len(desc.strip()) >= min_description_length and "no AI description available" not in desc:
                with_nonempty_description += 1
            else:
                picture_issues.append("description_missing_or_too_short")

            has_web = bool(isinstance(ai.get("web_context"), dict) and ai["web_context"].get("ai_summary"))
            if has_web:
                with_web_context += 1

            has_chart = bool(
                isinstance(ai.get("chart_data_extraction"), dict)
                and ai["chart_data_extraction"].get("extraction_success")
            )
            if has_chart:
                with_chart_data += 1

            # Expectations based on type
            if image_type == "DATA_VISUALIZATION" and not has_chart:
                picture_issues.append("expected_chart_data_missing")
            if image_type == "CONCEPTUAL" and not has_web:
                picture_issues.append("expected_web_context_missing")

            if picture_issues:
                per_picture_issues.append(
                    {
                        "index": idx,
                        "image_type": image_type,
                        "issues": picture_issues,
                    }
                )
        else:
            per_picture_issues.append(
                {"index": idx, "image_type": "UNKNOWN", "issues": ["missing_ai_analysis"]}
            )

    no_images_ok = (images_keys_found == 0) if require_no_images else True

    # Define success heuristic: no images remain AND at least one non-empty description exists
    success = no_images_ok and (with_nonempty_description > 0)

    summary = {
        "ok": success,
        "file": str(json_path),
        "total_pictures": total,
        "no_images_ok": no_images_ok,
        "images_keys_found": images_keys_found,
        "with_ai_analysis": with_ai_analysis,
        "with_nonempty_description": with_nonempty_description,
        "with_web_context": with_web_context,
        "with_chart_data": with_chart_data,
        "images_present_indices": images_present_indices,
        "per_picture_issues": per_picture_issues,
    }
    return summary


__all__ = ["verify_final_json"]


