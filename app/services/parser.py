import io
from pypdf import PdfReader
from docx import Document
from fastapi import UploadFile, HTTPException

async def extract_text_from_file(file: UploadFile) -> str:
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {str(e)}")

    elif filename.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(content))
            text = " ".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX file: {str(e)}")

    elif filename.endswith(".txt"):
        return content.decode("utf-8").strip()

    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file."
        )