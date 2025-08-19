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
from typing import Dict, List, Optional, Tuple, Any

from .chart_extraction import extract_chart_data_with_deplot
from .image_analysis import (
    analyze_single_image_with_lmstudio,
    enhance_analysis_with_chart_data,
    perform_web_search_and_summarize,
    summarize_deplot_with_lmstudio,
    extract_chart_with_image_and_deplot_verification,
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

    # --------------------[ Fallback helpers when AI is unavailable ]--------------------
    def _safe_get_caption(pic: Dict) -> str:
        """Return a best-effort caption/title string from picture metadata."""
        try:
            # Common locations: caption (str), captions (list of str), title, alt
            if isinstance(pic.get("caption"), str) and pic["caption"].strip():
                return pic["caption"].strip()
            if isinstance(pic.get("captions"), list) and pic["captions"]:
                text = " ".join([c for c in pic["captions"] if isinstance(c, str)])
                if text.strip():
                    return text.strip()
            for key in ("title", "alt", "name"):
                val = pic.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
        return ""

    def _decode_image_dimensions(image_uri: Optional[str]) -> tuple[Optional[int], Optional[int]]:
        """Decode base64 image to get (width, height). Returns (None, None) on failure."""
        try:
            if not (isinstance(image_uri, str) and image_uri.startswith("data:image")):
                return None, None
            header, encoded = image_uri.split(",", 1)
            import base64, io
            from PIL import Image
            img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            return int(img.width), int(img.height)
        except Exception:
            return None, None

    def _estimate_image_bytes(image_uri: Optional[str]) -> Optional[int]:
        """Estimate bytes from base64 payload length (rough)."""
        try:
            if not (isinstance(image_uri, str) and "," in image_uri):
                return None
            encoded = image_uri.split(",", 1)[1]
            # Base64 expansion ~ 4/3
            return int(len(encoded) * 3 / 4)
        except Exception:
            return None

    def _fallback_is_non_informative(image_uri: Optional[str], pic: Dict) -> bool:
        """
        Heuristic for non-informative images when AI analysis is unavailable:
        - Very small dimensions (e.g., logos, icons)
        - Very small payload size
        - Empty/very short caption/title
        """
        caption = _safe_get_caption(pic)
        width, height = _decode_image_dimensions(image_uri)
        approx_bytes = _estimate_image_bytes(image_uri) or 0

        very_small_dims = (isinstance(width, int) and isinstance(height, int) and (width <= 96 or height <= 96))
        tiny_payload = approx_bytes > 0 and approx_bytes <= 10_000  # ~10 KB
        caption_missing = len((caption or "").strip()) < 5

        # Consider non-informative if at least two signals are present
        score = sum([1 if very_small_dims else 0, 1 if tiny_payload else 0, 1 if caption_missing else 0])
        return score >= 2

    def _remove_chart_insight_from_text(text: Optional[str]) -> Optional[str]:
        if not isinstance(text, str) or not text:
            return text
        # Remove common DePlot insight sentence if accidentally present in description
        # Example: "Chart contains 15 data points with X-axis range from -1.50 to 600.00"
        import re as _re
        pattern = _re.compile(r"Chart contains\s+\d+\s+data points(?:\s+with X-axis range from\s+[-\d\.]+\s+to\s+[-\d\.]+)?", _re.IGNORECASE)
        return pattern.sub("", text).strip()

    def _sanitize_ai_analysis(ai: Dict) -> Dict:
        if not isinstance(ai, dict):
            return ai
        image_type = (ai.get("image_type") or "").upper()
        if image_type == "DATA_VISUALIZATION":
            # Ensure description is never polluted by chart insight
            ai["description"] = _remove_chart_insight_from_text(ai.get("description")) or ai.get("description") or ""
            if "enriched_description" in ai:
                ai["enriched_description"] = _remove_chart_insight_from_text(ai.get("enriched_description")) or ai.get("enriched_description")
        return ai

    for i, pic_data in enumerate(data["pictures"], start=1):
        logger.info("Processing picture %s/%s...", i, original_count)

        image_uri = (
            pic_data.get("image", {}).get("uri") if isinstance(pic_data.get("image"), dict) else None
        )
        if not image_uri or not isinstance(image_uri, str) or not image_uri.startswith("data:image"):
            logger.warning("Skipping picture #%s - no valid image data URI found.", i)
            enhanced_pictures.append(pic_data)
            continue

        logger.info("Starting Gemma analysis for picture #%s...", i)
        ai_analysis = analyze_single_image_with_lmstudio(image_uri, lm_studio_url, model_name, i)
        if ai_analysis is None:
            # AI unavailable (e.g., LM Studio offline). Apply deterministic fallback.
            logger.warning("AI analysis failed for picture #%s. Applying fallback heuristics.", i)
            if _fallback_is_non_informative(image_uri, pic_data):
                logger.info("Fallback marked image #%s as NON-INFORMATIVE -> removing", i)
                removed_count += 1
                continue
            else:
                # Keep picture but attach a minimal ai_analysis stub to avoid REMOVED placeholders later
                enhanced_pic = dict(pic_data)
                enhanced_pic["ai_analysis"] = {
                    "image_type": "UNKNOWN",
                    "description": "AI analysis unavailable due to model connection; image retained for later analysis.",
                    "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": model_name,
                    "will_replace_image": True,
                }
                enhanced_pictures.append(enhanced_pic)
                processed_count += 1
                if sleep_between_images_s > 0:
                    try:
                        time.sleep(sleep_between_images_s)
                    except Exception:
                        pass
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

        # Chart extraction for DATA_VISUALIZATION with image+DePlot verification
        if enable_chart_extraction and ai_analysis.get("image_type") == "DATA_VISUALIZATION":
            logger.info("Attempting chart data extraction for picture #%s...", i)
            # Step 1: Get raw DePlot output
            chart_data = extract_chart_data_with_deplot(image_uri, i)
            
            if chart_data and chart_data.get("raw_table"):
                # Step 2: Use image + DePlot verification for better accuracy
                logger.info("Verifying chart data with image analysis for picture #%s...", i)
                verified_chart = extract_chart_with_image_and_deplot_verification(
                    image_uri, chart_data["raw_table"], lm_studio_url, model_name, i
                )
                
                if verified_chart and verified_chart.get("parsing_success"):
                    chart_extracted_count += 1
                    enhanced_pic["ai_analysis"] = enhance_analysis_with_chart_data(
                        enhanced_pic["ai_analysis"], verified_chart
                    )
                    logger.info("Successfully verified chart data with image analysis for picture #%s", i)
                else:
                    # Fallback to original parsing methods
                    logger.info("Image verification failed, trying standard parsing for picture #%s...", i)
                    if chart_data.get("parsing_success"):
                        chart_extracted_count += 1
                        enhanced_pic["ai_analysis"] = enhance_analysis_with_chart_data(
                            enhanced_pic["ai_analysis"], chart_data
                        )
                    elif chart_data.get("raw_table"):
                        # Try LLM summarization as final fallback
                        logger.info("Trying LLM summarization as fallback for picture #%s...", i)
                        llm_chart = summarize_deplot_with_lmstudio(chart_data["raw_table"], lm_studio_url, model_name)
                        if llm_chart and llm_chart.get("parsing_success"):
                            chart_extracted_count += 1
                            enhanced_pic["ai_analysis"] = enhance_analysis_with_chart_data(
                                enhanced_pic["ai_analysis"], llm_chart
                            )
                        else:
                            enhanced_pic["ai_analysis"].setdefault("chart_data_extraction", {})
                            enhanced_pic["ai_analysis"]["chart_data_extraction"].update({
                                "extraction_success": False,
                                "extraction_method": "deplot+fallbacks_failed",
                            })
                    else:
                        enhanced_pic["ai_analysis"].setdefault("chart_data_extraction", {})
                        enhanced_pic["ai_analysis"]["chart_data_extraction"].update({
                            "extraction_success": False,
                            "extraction_method": "deplot_no_raw_output",
                        })
            else:
                # DePlot extraction completely failed
                enhanced_pic["ai_analysis"].setdefault("chart_data_extraction", {})
                enhanced_pic["ai_analysis"]["chart_data_extraction"].update({
                    "extraction_success": False,
                    "extraction_method": "deplot_failed",
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

        # Final sanitation to guarantee DePlot insight never overwrites descriptions
        enhanced_pic["ai_analysis"] = _sanitize_ai_analysis(enhanced_pic.get("ai_analysis", {}))

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

    def remove_keys_recursive(obj: Any, keys_to_remove: List[str]) -> tuple[Any, int]:
        """
        Recursively remove any keys in keys_to_remove from nested dict/list structures.
        Returns (sanitized_obj, num_removed_keys)
        """
        removed = 0
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k in keys_to_remove:
                    removed += 1
                    continue
                sanitized_v, rem = remove_keys_recursive(v, keys_to_remove)
                removed += rem
                new_dict[k] = sanitized_v
            return new_dict, removed
        if isinstance(obj, list):
            new_list = []
            for item in obj:
                sanitized_item, rem = remove_keys_recursive(item, keys_to_remove)
                removed += rem
                new_list.append(sanitized_item)
            return new_list, removed
        return obj, 0

    def count_keys_recursive(obj: Any, key_name: str) -> int:
        if isinstance(obj, dict):
            return (1 if key_name in obj else 0) + sum(count_keys_recursive(v, key_name) for v in obj.values())
        if isinstance(obj, list):
            return sum(count_keys_recursive(x, key_name) for x in obj)
        return 0

    original_count = len(data["pictures"])
    nlp_ready_pictures: List[Dict] = []
    removed_images_count = 0

    keys_to_strip = [
        "image",           # Main Docling image container with uri field
        "image_data",      # Alternative image data field
        "imageData",       # CamelCase variant
        "thumbnail",       # Thumbnail images
        "page_image",      # Page-level image references
        "pageImage",       # CamelCase page image
        "uri",            # Direct URI fields (including base64 data URIs)
        "data",           # Raw image data fields
        "base64",         # Explicit base64 fields
        "picture",        # Alternative picture containers
        "img",            # Short image references
        "imageUri",       # URI-specific image fields
        "image_uri",      # Snake case URI fields
    ]

    for i, pic_data in enumerate(data["pictures"], start=1):
        # Remove nested image-related keys recursively
        sanitized_pic, removed = remove_keys_recursive(pic_data, keys_to_strip)
        removed_images_count += removed
        
        # Ensure AI analysis exists and mark images as removed
        if "ai_analysis" not in sanitized_pic:
            sanitized_pic["ai_analysis"] = {
                "description": "Image was removed - no AI description available",
                "image_type": "REMOVED",
                "will_replace_image": True,
                "images_removed_in_step2": True,
            }
        else:
            # Mark existing AI analysis to indicate images were removed
            sanitized_pic["ai_analysis"]["images_removed_in_step2"] = True
            sanitized_pic["ai_analysis"]["will_replace_image"] = True
            
            # Update description to note image removal if it doesn't mention it
            current_desc = sanitized_pic["ai_analysis"].get("description", "")
            if current_desc and "removed" not in current_desc.lower():
                sanitized_pic["ai_analysis"]["description"] = f"{current_desc} (Original image data removed for NLP processing)"
        
        nlp_ready_pictures.append(sanitized_pic)

    data["pictures"] = nlp_ready_pictures

    # Also remove root-level image containers if present
    root_keys_to_remove = [
        "resources", "page_images", "pageImages", "images", 
        "figures", "pictures_raw", "media", "assets",
        "embedded_images", "image_resources"
    ]
    for root_key in root_keys_to_remove:
        if root_key in data:
            try:
                del data[root_key]
                logger.info(f"Removed root-level image container: {root_key}")
            except Exception:
                pass

    # Remove any remaining base64 data URIs from the entire document
    def clean_base64_data_uris(obj: Any) -> Any:
        """Recursively find and clean base64 data URIs from any string fields"""
        if isinstance(obj, dict):
            cleaned_dict = {}
            for k, v in obj.items():
                cleaned_v = clean_base64_data_uris(v)
                # Check if this is a data URI field and replace with placeholder
                if isinstance(cleaned_v, str) and cleaned_v.startswith("data:image"):
                    cleaned_dict[k] = "data:image/placeholder;base64,REMOVED_FOR_NLP"
                else:
                    cleaned_dict[k] = cleaned_v
            return cleaned_dict
        elif isinstance(obj, list):
            return [clean_base64_data_uris(item) for item in obj]
        else:
            return obj
    
    # Apply base64 cleaning to the entire data structure
    data = clean_base64_data_uris(data)

    # Post-removal verification (deep)
    remaining_image_keys = count_keys_recursive(data.get("pictures", []), "image")
    
    # Check for any remaining base64 data URIs
    def count_base64_uris(obj: Any) -> int:
        """Count remaining base64 data URIs in the structure"""
        count = 0
        if isinstance(obj, dict):
            for v in obj.values():
                count += count_base64_uris(v)
        elif isinstance(obj, list):
            for item in obj:
                count += count_base64_uris(item)
        elif isinstance(obj, str) and obj.startswith("data:image") and "base64," in obj and len(obj) > 100:
            # Only count substantial base64 data, not our placeholders
            count += 1
        return count
    
    remaining_base64_uris = count_base64_uris(data)
    
    nlp_metadata = {
        "original_picture_count": original_count,
        "nlp_ready_picture_count": len(nlp_ready_pictures),
        "removed_images_count": removed_images_count,
        "removed_image_keys_count": removed_images_count,
        "images_completely_removed": True,
        "base64_data_uris_removed": True,
        "nlp_ready": True,
        "step2_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "NLP_ready_no_images_enhanced",
        "remaining_image_keys": remaining_image_keys,
        "remaining_base64_uris": remaining_base64_uris,
        "image_data_thoroughly_cleaned": remaining_base64_uris == 0,
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
            "Step 2 complete. NLP-ready JSON saved to: %s",
            output_json_path,
        )
        logger.info(
            "Image removal summary: removed %s image key occurrences, %s image keys remaining, %s base64 URIs remaining",
            removed_images_count,
            remaining_image_keys,
            remaining_base64_uris,
        )
        if remaining_base64_uris == 0:
            logger.info("✅ All image data successfully removed - JSON is fully NLP-ready")
        else:
            logger.warning("⚠️ Some base64 image data may still be present (%s URIs)", remaining_base64_uris)
        return True, str(output_json_path)
    except Exception:
        logger.exception("Failed to save NLP-ready JSON to: %s", output_json_path)
        return False, None


__all__ = [
    "step1_add_ai_descriptions_with_chart_extraction",
    "step2_remove_all_images",
]


