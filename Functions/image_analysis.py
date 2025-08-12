"""
Image analysis helpers using LM Studio-compatible chat completions API and
optional web search summarization using DuckDuckGo, plus enrichment utilities.

All code comments are written in English.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests
from duckduckgo_search import DDGS


logger = logging.getLogger(__name__)


def analyze_single_image_with_lmstudio(
    image_uri: str,
    lm_studio_url: str,
    model_name: str,
    pic_number: int,
    *,
    max_tokens: int = 700,
    temperature: float = 0.1,
) -> Optional[Dict]:
    """
    Analyze a single image using LM Studio chat completions style API.

    Returns a dict with keys: is_non_informative, image_type, detailed_description, search_keywords.
    """
    enhanced_prompt = (
        "You are an expert scientific analyst. Analyze the provided image and follow these steps:\n\n"
        "1. First, determine if this is:\n"
        "   - INFORMATIVE: A meaningful scientific element (graphs, charts, diagrams, maps, flowcharts, etc.)\n"
        "   - NON-INFORMATIVE: Publisher logos, watermarks, decorative icons, etc.\n\n"
        "2. If NON-INFORMATIVE, respond with exactly: \"N/A\"\n\n"
        "3. If INFORMATIVE, classify the image type:\n"
        "   - DATA_VISUALIZATION: Charts, graphs, plots with actual data points, statistical visualizations\n"
        "   - CONCEPTUAL: Flowcharts, process diagrams, maps, conceptual frameworks, schematic diagrams, methodological illustrations\n\n"
        "4. CRITICAL: Provide a comprehensive text description that can completely replace the image in a document. This description will be used instead of the image for NLP processing. Include all important details, relationships, data patterns, labels, and contextual information that a reader would need to understand what the image conveyed.\n\n"
        "5. If the image is CONCEPTUAL type, also provide 2-3 specific search keywords that would help find background information about the concepts, methods, or geographic locations shown in the image.\n\n"
        "Format your response as:\n"
        "TYPE: [DATA_VISUALIZATION/CONCEPTUAL]\n"
        "DETAILED_DESCRIPTION: [Your comprehensive replacement text description that captures all essential information from the image]\n"
        "SEARCH_KEYWORDS: [keyword1, keyword2, keyword3] (only if CONCEPTUAL type)"
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(lm_studio_url, json=payload, timeout=300)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("Failed to analyze image #%s: %s", pic_number, exc)
        return None

    if ai_response.strip() == "N/A":
        return {"is_non_informative": True}

    result: Dict = {
        "is_non_informative": False,
        "image_type": "UNKNOWN",
        "detailed_description": ai_response,
        "search_keywords": [],
    }

    try:
        lines = ai_response.split("\n")
        for line in lines:
            if line.startswith("TYPE:"):
                result["image_type"] = line.replace("TYPE:", "").strip()
            elif line.startswith("DETAILED_DESCRIPTION:"):
                result["detailed_description"] = line.replace("DETAILED_DESCRIPTION:", "").strip()
            elif line.startswith("SEARCH_KEYWORDS:"):
                keywords_text = line.replace("SEARCH_KEYWORDS:", "").strip().strip("[]")
                result["search_keywords"] = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    except Exception:
        logger.warning("Could not parse structured response for picture #%s", pic_number)

    return result


def perform_web_search_and_summarize(
    search_keywords: List[str],
    lm_studio_url: str,
    model_name: str,
    *,
    max_sources: int = 5,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> Optional[Dict]:
    """
    Perform DuckDuckGo search and summarize results via LM Studio text generation.
    """
    try:
        search_query = " ".join(search_keywords[:2]) if search_keywords else ""
        if not search_query:
            return None

        ddgs = DDGS()
        results = ddgs.text(search_query, max_results=max_sources)
        results = list(results) if results else []
        if not results:
            return None

        sources_text = ""
        sources_info = []
        for i, result in enumerate(results, 1):
            title = result.get("title", f"Source {i}")
            body = result.get("body", "")
            href = result.get("href", "")
            sources_text += f"\nSource {i} - {title}:\n{body}\n"
            if href:
                sources_text += f"URL: {href}\n"
            sources_text += "-" * 50 + "\n"
            sources_info.append({"title": title, "body": body, "url": href})

        summary_prompt = (
            f"Based on the following search results about \"{search_query}\", provide a concise one-paragraph summary "
            f"focusing on the essential concepts and definitions (maximum 6-8 sentences):\n\n{sources_text}\n"
            f"Please synthesize this information into a brief, coherent explanation.\n\nSummary:"
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": summary_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(lm_studio_url, json=payload, timeout=120)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()

        return {
            "search_query": search_query,
            "search_keywords": search_keywords,
            "sources_count": len(sources_info),
            "sources": sources_info,
            "ai_summary": summary,
            "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as exc:
        logger.error("Web search and summarization failed: %s", exc)
        return None


def enhance_analysis_with_chart_data(ai_analysis: Dict, chart_data: Dict) -> Dict:
    """
    Add chart data metadata into AI analysis dict, and enrich descriptions.
    """
    if not chart_data or not ai_analysis:
        return ai_analysis

    enhanced = dict(ai_analysis)
    enhanced["chart_data_extraction"] = {
        "extraction_success": True,
        "data_points_count": chart_data.get("data_points_count", 0),
        "chart_type": chart_data.get("chart_type", "unknown"),
        "x_axis_label": chart_data.get("x_axis_label", ""),
        "y_axis_labels": chart_data.get("y_axis_labels", []),
        "data_ranges": {
            "x_range": chart_data.get("x_range"),
            "y_ranges": chart_data.get("y_ranges", {}),
        },
        "extraction_method": chart_data.get("extraction_method", ""),
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if chart_data.get("data_points_count", 0) > 0:
        data_insight = f" Chart contains {chart_data['data_points_count']} data points"
        if chart_data.get("x_range"):
            x_min, x_max = chart_data["x_range"]
            data_insight += f" with X-axis range from {x_min:.2f} to {x_max:.2f}"
        enhanced["enriched_description"] = enhanced.get("enriched_description", "") + data_insight
        enhanced["description"] = enhanced.get("description", "") + data_insight

    return enhanced


__all__ = [
    "analyze_single_image_with_lmstudio",
    "perform_web_search_and_summarize",
    "enhance_analysis_with_chart_data",
]


