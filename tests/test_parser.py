import io
import pytest
from docx import Document
from fastapi import HTTPException, UploadFile
from pypdf import PdfWriter

from app.services.parser import extract_text_from_file


@pytest.mark.asyncio
async def test_extract_text_from_txt():
    content = b"Python developer with FastAPI experience."
    file = UploadFile(filename="resume.txt", file=io.BytesIO(content))

    text = await extract_text_from_file(file)
    assert "FastAPI experience" in text


@pytest.mark.asyncio
async def test_extract_text_from_docx():
    doc = Document()
    doc.add_paragraph("Experienced Docker and Kubernetes Engineer.")
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    file = UploadFile(filename="resume.docx", file=buffer)
    text = await extract_text_from_file(file)
    assert "Kubernetes Engineer" in text


@pytest.mark.asyncio
async def test_extract_text_from_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)

    file = UploadFile(filename="resume.pdf", file=buffer)
    text = await extract_text_from_file(file)
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_unsupported_file_extension():
    file = UploadFile(filename="resume.png", file=io.BytesIO(b"dummy image"))
    with pytest.raises(HTTPException) as exc_info:
        await extract_text_from_file(file)
    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in exc_info.value.detail
