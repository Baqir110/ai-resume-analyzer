from typing import List
from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    ats_match_score: float = Field(..., description="Calculated ATS compatibility score (0-100%)")
    matching_skills: List[str] = Field(..., description="Skills found in both CV and Job Description")
    missing_skills: List[str] = Field(..., description="Skills in Job Description missing from CV")
    keyword_density_score: float = Field(..., description="Keyword overlap density ratio")
    improvement_suggestions: List[str] = Field(..., description="Actionable CV optimization tips")