import io
import zipfile

from docx import Document
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.core.config import settings

_MAX_EXTRACTED_TEXT_CHARS = 1_000_000
_MAX_PDF_PAGES = 100
_MAX_DOCX_ENTRIES = 2_000
_MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
_MAX_DOCX_EXPANSION_RATIO = 150


async def extract_text_from_file(file: UploadFile) -> str:
    import time as _time

    from app.core.event_log import log_event

    _started = _time.perf_counter()
    _fname = file.filename or ""
    _ext = _fname.rsplit(".", 1)[-1].lower() if "." in _fname else "unknown"

    try:
        text = await _extract_text_impl(file)
    except HTTPException:
        log_event(
            "parse",
            "parse_failed",
            file_type=_ext,
            filename=_fname,
            duration_ms=round((_time.perf_counter() - _started) * 1000, 1),
            error="document_validation_failed",
        )
        raise
    except Exception:
        log_event(
            "parse",
            "parse_failed",
            file_type=_ext,
            filename=_fname,
            duration_ms=round((_time.perf_counter() - _started) * 1000, 1),
            error="document_parse_failed",
        )
        raise

    log_event(
        "parse",
        "parse_completed",
        file_type=_ext,
        filename=_fname,
        chars_extracted=len(text or ""),
        empty=not bool((text or "").strip()),
        duration_ms=round((_time.perf_counter() - _started) * 1000, 1),
    )
    return text


async def _extract_text_impl(file: UploadFile) -> str:
    try:
        max_size = int(settings.MAX_FILE_SIZE)
        declared_size = getattr(file, "size", None)
        if declared_size is not None and int(declared_size) > max_size:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        content = await file.read(max_size + 1)
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        # Reset file pointer so other downstream functions can re-read if needed
        await file.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read upload stream") from e

    filename = (file.filename or "").lower()

    # 1. PDF Files
    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted or len(reader.pages) > _MAX_PDF_PAGES:
                raise ValueError("invalid PDF structure")
            pages_text = []
            extracted_chars = 0
            for page in reader.pages:
                extracted = page.extract_text() or ""
                extracted_chars += len(extracted)
                if extracted_chars > _MAX_EXTRACTED_TEXT_CHARS:
                    raise ValueError("extracted PDF text is too large")
                if extracted.strip():
                    pages_text.append(extracted.strip())
            return "\n\n".join(pages_text).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to parse PDF file") from exc

    # 2. DOCX Files (Paragraphs + Tables)
    elif filename.endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_DOCX_ENTRIES:
                    raise ValueError("DOCX archive has too many entries")
                total_uncompressed = sum(info.file_size for info in infos)
                if total_uncompressed > _MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise ValueError("DOCX archive expands beyond the limit")
                if total_uncompressed > max(1, len(content) * _MAX_DOCX_EXPANSION_RATIO):
                    raise ValueError("DOCX expansion ratio is too large")
                if any(info.flag_bits & 0x1 for info in infos):
                    raise ValueError("encrypted DOCX archives are not supported")

            doc = Document(io.BytesIO(content))
            extracted_blocks = []
            extracted_chars = 0

            # Extract standard paragraphs with a hard output bound.
            for para in doc.paragraphs:
                text = para.text.strip()
                extracted_chars += len(text)
                if extracted_chars > _MAX_EXTRACTED_TEXT_CHARS:
                    raise ValueError("extracted DOCX text is too large")
                if text:
                    extracted_blocks.append(text)

            # Extract content from tables if present.
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        extracted_chars += len(cell_text)
                        if extracted_chars > _MAX_EXTRACTED_TEXT_CHARS:
                            raise ValueError("extracted DOCX text is too large")
                        if cell_text and cell_text not in extracted_blocks:
                            extracted_blocks.append(cell_text)

            return "\n\n".join(extracted_blocks).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to parse DOCX file") from exc

    # 3. Plain Text Files
    elif filename.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to parse TXT file") from exc

    # 4. Fallback for Unsupported Types
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file.",
        )
