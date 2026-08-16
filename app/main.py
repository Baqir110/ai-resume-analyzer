from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="AI Resume & CV Analyzer API",
    version="1.0.0",
    description="Automated ATS compatibility scorer, skill gap extractor, and resume optimization engine."
)

app.include_router(api_router, prefix="/api/v1/resume", tags=["Resume Analyzer"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)