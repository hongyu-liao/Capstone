"""
PDF to JSON conversion using Docling with LM Studio VLM pipeline.

All code comments are written in English.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


logger = logging.getLogger(__name__)


def convert_pdf_with_lmstudio(
    pdf_path_str: str,
    output_dir_str: str = "output_lmstudio_conversion",
    *,
    lm_studio_url: str = "http://localhost:1234/v1/chat/completions",
    model_identifier: str = "google/gemma-3-12b-it-gguf",
    prompt: str = "Parse the document.",
    max_tokens: int = 16384,
    generate_page_images: bool = True,
) -> Optional[str]:
    """
    Convert a PDF to Docling JSON using a model served by LM Studio.

    Args:
        pdf_path_str: Path to the source PDF file.
        output_dir_str: Directory to save the output files.
        lm_studio_url: LM Studio Chat Completions endpoint.
        model_identifier: Exact model name loaded in LM Studio.
        prompt: Prompt used by the VLM pipeline.
        max_tokens: Maximum tokens for generation.
        generate_page_images: Whether to generate page images in the JSON.

    Returns:
        Path to the saved JSON file as string, or None if conversion failed.
    """
    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_file():
        logger.error("File not found: %s", pdf_path)
        return None

    # Prepare output directory
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output will be saved in: %s", output_dir.resolve())

    # Configure the VLM Pipeline to use LM Studio
    logger.info("Configuring VLM Pipeline to use '%s' on LM Studio...", model_identifier)

    pipeline_options = VlmPipelineOptions(
        url=lm_studio_url,
        model=model_identifier,
        prompt=prompt,
        params={"max_tokens": max_tokens},
        generate_page_images=generate_page_images,
    )

    # Initialize the Document Converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Run conversion
    logger.info("Starting PDF conversion for: %s", pdf_path.name)
    try:
        result = converter.convert(pdf_path)
        document = result.document
        logger.info("PDF conversion complete.")
    except Exception:
        logger.exception(
            "A critical error occurred during conversion. Is the LM Studio server running with model '%s' loaded?",
            model_identifier,
        )
        return None

    # Save JSON output
    base_name = f"{pdf_path.stem}-{model_identifier.replace('/', '_')}"
    json_path = output_dir / f"{base_name}.json"
    try:
        document.save_as_json(json_path)
        logger.info("Saved JSON: %s", json_path.name)
        return str(json_path)
    except Exception:
        logger.exception("Failed to save JSON output to: %s", json_path)
        return None


__all__ = ["convert_pdf_with_lmstudio"]


