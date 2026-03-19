"""Extract text and image data from base64-encoded file attachments."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFile:
    filename: str
    text: str | None = None
    image_base64: str | None = None
    mime_type: str | None = None


def process_files(files: list[dict]) -> list[ExtractedFile]:
    """Process a list of file attachments from the competition payload.

    Returns extracted text for PDFs and base64 image data for images,
    ready to be included in the LLM context.
    """
    results: list[ExtractedFile] = []
    for f in files:
        filename = f.get("filename", "unknown")
        content_b64 = f.get("content_base64", "")
        mime = f.get("mime_type", "")

        try:
            raw = base64.b64decode(content_b64)
        except Exception:
            logger.warning("Failed to decode base64 for %s", filename)
            continue

        if mime == "application/pdf":
            results.append(_extract_pdf(filename, raw))
        elif mime.startswith("image/"):
            results.append(
                ExtractedFile(
                    filename=filename,
                    image_base64=content_b64,
                    mime_type=mime,
                )
            )
        else:
            text = raw.decode("utf-8", errors="replace")
            results.append(ExtractedFile(filename=filename, text=text))

    return results


def _extract_pdf(filename: str, raw_bytes: bytes) -> ExtractedFile:
    """Extract text from a PDF. Falls back to first-page image if text is sparse."""
    try:
        import fitz

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        text_parts: list[str] = []
        for page in doc:
            text_parts.append(page.get_text())
        text = "\n".join(text_parts).strip()

        if len(text) > 50:
            return ExtractedFile(filename=filename, text=text)

        # Sparse text — render first page as image for the LLM's vision
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        return ExtractedFile(
            filename=filename,
            text=text if text else None,
            image_base64=img_b64,
            mime_type="image/png",
        )
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", filename, exc)
        return ExtractedFile(filename=filename, text=f"[Could not extract PDF: {exc}]")


def build_file_context(extracted: list[ExtractedFile]) -> str:
    """Build a text summary of all extracted files for the LLM prompt."""
    if not extracted:
        return ""

    parts: list[str] = ["## Attached Files\n"]
    for ef in extracted:
        parts.append(f"### {ef.filename}")
        if ef.text:
            parts.append(ef.text)
        if ef.image_base64 and not ef.text:
            parts.append("[Image attached — see image content in this message]")
        parts.append("")

    return "\n".join(parts)
