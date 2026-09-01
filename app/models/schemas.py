from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecommendationDetails(BaseModel):
    recommended_format: str = Field(..., json_schema_extra={"example": "german_latex"})
    label: str = Field(
        ..., json_schema_extra={"example": "🇩🇪 German Corporate LaTeX (.TEX / .PDF)"}
    )
    reason: str = Field(
        ...,
        json_schema_extra={
            "example": "Target posting is in German; structured LaTeX layout with custom macros provides precise formatting."
        },
    )


class AnalysisResponse(BaseModel):
    status: str = Field(default="success", json_schema_extra={"example": "success"})
    ats_match_score: float = Field(..., json_schema_extra={"example": 77.55})
    keyword_density_score: float = Field(..., json_schema_extra={"example": 100.0})
    matching_skills: List[str] = Field(
        ...,
        json_schema_extra={
            "example": ["Python", "FastAPI", "Docker", "PostgreSQL", "AWS", "Terraform"]
        },
    )
    missing_skills: List[str] = Field(
        ..., json_schema_extra={"example": ["Kubernetes", "Redis"]}
    )
    improvement_suggestions: List[str] = Field(
        ...,
        json_schema_extra={
            "example": [
                "Good key term match! Ensure keywords appear naturally inside achievements rather than just a standalone skills list."
            ]
        },
    )
    recommendation: Optional[RecommendationDetails] = Field(default=None)


class ErrorResponse(BaseModel):
    status: str = Field(default="error", json_schema_extra={"example": "error"})
    message: str = Field(
        ...,
        json_schema_extra={
            "example": "Could not extract readable text from resume file."
        },
    )
