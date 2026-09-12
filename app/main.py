"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.api.new_features_endpoints import router as new_features_router
from app.api.streaming_endpoints import router as streaming_router

app = FastAPI(
    title="AI Resume & CV Analyzer API",
    version="1.0.0",
    description=(
        "Automated ATS compatibility scorer, skill gap extractor, "
        "and resume optimization engine."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    api_router,
    prefix="/api/v1/resume",
    tags=["Resume Analyzer"],
)

app.include_router(
    new_features_router,
    prefix="/api/v1/resume",
    tags=["New Features"],
)

app.include_router(
    streaming_router,
    prefix="/api/v1/resume",
    tags=["Streaming"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
