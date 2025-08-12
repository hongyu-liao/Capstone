"""
High-level pipeline steps for processing Docling JSON into an enhanced JSON and
then an NLP-ready JSON, closely matching the experimental notebook behavior.

All code comments are written in English.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chart_extraction import extract_chart_data_with_deplot
from .image_analysis import (
    analyze_single_image_with_lmstudio,
    enhance_analysis_with_chart_data,
    perform_web_search_and_summarize,
)


logger = logging.getLogger(__name__)


def step1_add_ai_descriptions_with_chart_extraction(
    json_path_str: str,
    *,
    output_json_path_str: Optional[str] = None,
    lm_studio_url: str = "http://localhost:1234/v1/chat/completions",
    model_name: str = "google/gemma-3-12b-it-gguf",
    enable_chart_extraction: bool = True,
    enable_web_search_for_conceptual: bool = True,
    sleep_between_images_s: float = 1.0,
) -> Tuple[bool, Optional[str]]:
    """
    Step 1: Add AI descriptions, optionally extract chart data using DePlot, and
    optionally augment conceptual images with web search summaries. Keeps the
    original images in the JSON (mirrors the notebook's enhanced step 1).

    Returns (success, output_json_path).
    """
    json_path = Path(json_path_str)
    if not json_path.is_file():
        logger.error("JSON file not found: %s", json_path)
        return False, None

    # Determine output path
    if output_json_path_str is None:
        output_json_path = json_path.parent / f"{json_path.stem}_with_descriptions_and_chart_data{json_path.suffix}"
    else:
        output_json_path = Path(output_json_path_str)

    # Load input JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read JSON file: %s", json_path)
        return False, None

    if "pictures" not in data or not data["pictures"]:
        logger.warning("No 'pictures' key found or list is empty in: %s", json_path)
        return False, None

    original_count = len(data["pictures"])
    processed_count = 0
    removed_count = 0
    chart_extracted_count = 0

    enhanced_pictures: List[Dict] = []

    for i, pic_data in enumerate(data["pictures"], start=1):
        logger.info("Processing picture %s/%s...", i, original_count)

        image_uri = (
            pic_data.get("image", {}).get("uri") if isinstance(pic_data.get("image"), dict) else None
        )
        if not image_uri or not isinstance(image_uri, str) or not image_uri.startswith("data:image"):
            logger.warning("Skipping picture #%s - no valid image data URI found.", i)
            enhanced_pictures.append(pic_data)
            continue

        ai_analysis = analyze_single_image_with_lmstudio(image_uri, lm_studio_url, model_name, i)
        if ai_analysis is None:
            logger.warning("AI analysis failed for picture #%s, keeping original", i)
            enhanced_pictures.append(pic_data)
            continue

        if ai_analysis.get("is_non_informative", False):
            logger.info("Marking non-informative image #%s for removal", i)
            removed_count += 1
            continue

        enhanced_pic = dict(pic_data)
        enhanced_pic["ai_analysis"] = {
            "image_type": ai_analysis.get("image_type", "UNKNOWN"),
            "description": ai_analysis.get("detailed_description", ""),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": model_name,
            "will_replace_image": True,
        }

        # Chart extraction for DATA_VISUALIZATION
        if enable_chart_extraction and ai_analysis.get("image_type") == "DATA_VISUALIZATION":
            logger.info("Attempting chart data extraction for picture #%s...", i)
            chart_data = extract_chart_data_with_deplot(image_uri, i)
            if chart_data:
                chart_extracted_count += 1
                enhanced_pic["ai_analysis"] = enhance_analysis_with_chart_data(
                    enhanced_pic["ai_analysis"], chart_data
                )
            else:
                # If DePlot failed but the image is likely a line chart, still insert a placeholder to mark expectation
                enhanced_pic["ai_analysis"].setdefault("chart_data_extraction", {})
                enhanced_pic["ai_analysis"]["chart_data_extraction"].update({
                    "extraction_success": False,
                    "extraction_method": "google/deplot",
                })

        # Web search for CONCEPTUAL images
        if enable_web_search_for_conceptual and ai_analysis.get("image_type") == "CONCEPTUAL":
            if ai_analysis.get("search_keywords"):
                enhanced_pic["ai_analysis"]["search_keywords"] = ai_analysis.get("search_keywords", [])
                web_summary = perform_web_search_and_summarize(
                    ai_analysis["search_keywords"], lm_studio_url, model_name
                )
                if web_summary:
                    enhanced_pic["ai_analysis"]["web_context"] = web_summary
                    original_desc = enhanced_pic["ai_analysis"]["description"]
                    enhanced_pic["ai_analysis"]["enriched_description"] = (
                        f"{original_desc} Additional context: {web_summary.get('ai_summary', '')}"
                    )

        enhanced_pictures.append(enhanced_pic)
        processed_count += 1

        if sleep_between_images_s > 0:
            try:
                time.sleep(sleep_between_images_s)
            except Exception:
                pass

    data["pictures"] = enhanced_pictures
    step1_metadata = {
        "original_picture_count": original_count,
        "analyzed_picture_count": processed_count,
        "removed_picture_count": removed_count,
        "chart_data_extracted_count": chart_extracted_count,
        "images_still_present": True,
        "step1_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ai_model": model_name,
        "chart_extraction_model": "google/deplot" if enable_chart_extraction else None,
        "step": "AI_descriptions_and_chart_data_extraction",
        "next_step": "Remove images using step2_remove_all_images()",
    }
    data["step1_metadata"] = step1_metadata

    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Enhanced Step 1 complete. Enhanced JSON saved to: %s", output_json_path)
        return True, str(output_json_path)
    except Exception:
        logger.exception("Failed to save enhanced JSON to: %s", output_json_path)
        return False, None


def step2_remove_all_images(
    json_with_descriptions_path: str, *, output_json_path_str: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Step 2: Remove all image payloads while preserving AI analysis fields,
    producing an NLP-ready JSON file.

    Returns (success, output_json_path).
    """
    json_path = Path(json_with_descriptions_path)
    if not json_path.is_file():
        logger.error("JSON file not found: %s", json_path)
        return False, None

    if output_json_path_str is None:
        base_name = json_path.stem
        if base_name.endswith("_with_descriptions_and_chart_data"):
            new_name = base_name.replace("_with_descriptions_and_chart_data", "") + "_nlp_ready"
        else:
            new_name = base_name + "_nlp_ready"
        output_json_path = json_path.parent / f"{new_name}{json_path.suffix}"
    else:
        output_json_path = Path(output_json_path_str)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read JSON file: %s", json_path)
        return False, None

    if "pictures" not in data:
        logger.warning("No 'pictures' key found in: %s", json_path)
        return False, None

    original_count = len(data["pictures"])
    nlp_ready_pictures: List[Dict] = []
    removed_images_count = 0

    for i, pic_data in enumerate(data["pictures"], start=1):
        nlp_pic_data: Dict = {}
        for key, value in pic_data.items():
            if key != "image":
                nlp_pic_data[key] = value
            else:
                removed_images_count += 1
        if "ai_analysis" not in nlp_pic_data:
            nlp_pic_data["ai_analysis"] = {
                "description": "Image was removed - no AI description available",
                "image_type": "REMOVED",
                "will_replace_image": True,
            }
        nlp_ready_pictures.append(nlp_pic_data)

    data["pictures"] = nlp_ready_pictures

    nlp_metadata = {
        "original_picture_count": original_count,
        "nlp_ready_picture_count": len(nlp_ready_pictures),
        "removed_images_count": removed_images_count,
        "images_completely_removed": True,
        "nlp_ready": True,
        "step2_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "NLP_ready_no_images",
    }
    data["nlp_ready_metadata"] = nlp_metadata

    if "step1_metadata" in data:
        try:
            del data["step1_metadata"]
        except Exception:
            pass

    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            "Step 2 complete. NLP-ready JSON saved to: %s (images removed: %s)",
            output_json_path,
            removed_images_count,
        )
        return True, str(output_json_path)
    except Exception:
        logger.exception("Failed to save NLP-ready JSON to: %s", output_json_path)
        return False, None


__all__ = [
    "step1_add_ai_descriptions_with_chart_extraction",
    "step2_remove_all_images",
]


