"""
Image analysis helpers using LM Studio-compatible chat completions API and
optional web search summarization using DuckDuckGo, plus enrichment utilities.

All code comments are written in English.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional
import json

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

    logger.info("[Gemma] ▶ Start image analysis | pic#%s | model='%s'", pic_number, model_name)

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
        logger.info(
            "[Gemma] ◀ Response received | pic#%s | chars=%s",
            pic_number,
            len(ai_response or ""),
        )
    except Exception as exc:
        logger.error("Failed to analyze image #%s: %s", pic_number, exc)
        return None

    if ai_response.strip() == "N/A":
        return {"is_non_informative": True}

    original_response = ai_response
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
                parsed_type = line.replace("TYPE:", "").strip()
                if parsed_type:
                    result["image_type"] = parsed_type
            elif line.startswith("DETAILED_DESCRIPTION:"):
                parsed_desc = line.replace("DETAILED_DESCRIPTION:", "").strip()
                if parsed_desc:
                    result["detailed_description"] = parsed_desc
            elif line.startswith("SEARCH_KEYWORDS:"):
                keywords_text = line.replace("SEARCH_KEYWORDS:", "").strip().strip("[]")
                result["search_keywords"] = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    except Exception:
        logger.warning("Could not parse structured response for picture #%s", pic_number)

    # Ensure we never lose the original description
    if not isinstance(result.get("detailed_description"), str) or len(result.get("detailed_description", "").strip()) == 0:
        result["detailed_description"] = original_response

    logger.info(
        "[Gemma] ✔ Parsed | pic#%s | type=%s | keywords=%s | desc_len=%s",
        pic_number,
        result.get("image_type"),
        len(result.get("search_keywords") or []),
        len(result.get("detailed_description") or ""),
    )

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

        logger.info("[Web] ▶ DDGS search start | query='%s'", search_query)
        ddgs = DDGS()
        results = ddgs.text(search_query, max_results=max_sources)
        results = list(results) if results else []
        logger.info("[Web] ◀ DDGS results | count=%s", len(results))
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

        logger.info("[Web] ▶ Summarize DDGS with LLM | model='%s'", model_name)
        response = requests.post(lm_studio_url, json=payload, timeout=120)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("[Web] ◀ Summary received | chars=%s", len(summary or ""))

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
    enhanced_chart: Dict[str, object] = {
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

    # Preserve datasets and auxiliary fields if available
    if chart_data.get("datasets"):
        enhanced_chart["datasets"] = chart_data["datasets"]
    if chart_data.get("raw_table"):
        enhanced_chart["raw_table"] = chart_data["raw_table"]
    if chart_data.get("x_categories"):
        enhanced_chart["x_categories"] = chart_data["x_categories"]

    enhanced["chart_data_extraction"] = enhanced_chart
    logger.info(
        "[Chart] ⛏ Merge chart data | series=%s | points=%s | method=%s",
        len((enhanced_chart.get("datasets") or {}).keys()),
        chart_data.get("data_points_count", 0),
        enhanced_chart.get("extraction_method"),
    )

    if chart_data.get("data_points_count", 0) > 0:
        data_insight = f"Chart contains {chart_data['data_points_count']} data points"
        if chart_data.get("x_range"):
            x_min, x_max = chart_data["x_range"]
            data_insight += f" with X-axis range from {x_min:.2f} to {x_max:.2f}"
        # Do NOT modify description. Store insight separately for UI to render under description.
        enhanced["chart_insight"] = data_insight

    return enhanced


__all__ = [
    "analyze_single_image_with_lmstudio",
    "perform_web_search_and_summarize",
    "enhance_analysis_with_chart_data",
]


def summarize_deplot_with_lmstudio(
    raw_table: str,
    lm_studio_url: str,
    model_name: str,
    *,
    max_tokens: int = 700,
    temperature: float = 0.1,
) -> Optional[Dict]:
    """
    Ask the LLM (e.g., Gemma via LM Studio) to convert DePlot's linearized table string
    into a structured JSON containing datasets. This mirrors the web search summarization flow.

    Expected JSON schema from the model:
    {
      "datasets": {"SeriesName": [[x, y], ...], ...},
      "x_label": "...",
      "y_labels": ["...", ...]
    }
    """
    if not raw_table:
        return None

    system_prompt = (
        "You are a data extraction assistant. Convert the following linearized table (from a chart) "
        "into a compact JSON with numeric datasets suitable for plotting. Use float numbers."
    )
    user_prompt = (
        "Linearized table (tokens '<0x0A>' denote line breaks, '|' denote columns):\n\n"
        f"{raw_table}\n\n"
        "Return ONLY JSON with keys: datasets (map of series name to list of [x,y] pairs), "
        "x_label (string), y_labels (list of strings)."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        logger.info("[Chart-LLM] ▶ Summarize DePlot output with LLM | model='%s'", model_name)
        response = requests.post(lm_studio_url, json=payload, timeout=180)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("[Chart-LLM] ◀ LLM content received | chars=%s", len(content or ""))
        # Attempt to locate JSON in the response
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
            logger.info("[Chart-LLM] ⎘ Extracted JSON substring | chars=%s", len(json_str or ""))
            parsed = json.loads(json_str)
            # Normalize to our structure
            datasets = parsed.get("datasets") or {}
            x_label = parsed.get("x_label") or "X"
            y_labels = parsed.get("y_labels") or list(datasets.keys())

            # Ensure numeric pairs
            normalized = {}
            total_points = 0
            for name, points in datasets.items():
                clean_points = []
                for p in points:
                    try:
                        x, y = float(p[0]), float(p[1])
                        clean_points.append([x, y])
                    except Exception:
                        continue
                if clean_points:
                    total_points += len(clean_points)
                    normalized[name] = [(float(x), float(y)) for x, y in clean_points]

            if not normalized:
                return None

            logger.info(
                "[Chart-LLM] ✔ Normalized datasets | series=%s | total_points=%s",
                len(normalized.keys()),
                total_points,
            )
            return {
                "chart_type": "line_chart",
                "x_axis_label": x_label,
                "y_axis_labels": y_labels,
                "datasets": normalized,
                "data_points_count": total_points,
                "extraction_method": "deplot+llm_summarization",
                "parsing_success": True,
            }
    except Exception as exc:
        logger.error("DePlot LLM summarization failed: %s", exc)
        return None



