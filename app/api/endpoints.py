from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import AnalysisResponse
from app.services.parser import extract_text_from_file
from app.services.analyzer import analyze_resume_content

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    job_description: str = Form(..., description="Target Job Description text"),
    resume: UploadFile = File(..., description="Resume document (.pdf, .docx, or .txt)")
):
    # Extract text from uploaded document
    resume_text = await extract_text_from_file(resume)
    
    if len(resume_text) < 50:
        raise HTTPException(
            status_code=400, 
            detail="Extracted resume text is too short or empty. Please check the document contents."
        )

    # Perform ATS analysis
    analysis_results = analyze_resume_content(resume_text, job_description)

    return AnalysisResponse(**analysis_results)