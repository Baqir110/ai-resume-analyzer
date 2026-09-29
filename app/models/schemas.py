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


class ATSCategoryBreakdown(BaseModel):
    """
    One category in the ATS score breakdown.

    ``score`` and ``not_measured`` go together: before a PDF exists the PDF
    Parsing category has nothing to measure, so its score is ``None`` and
    ``not_measured`` is True. Modelling that explicitly keeps an unmeasured
    category from being reported as a zero, which would read as "this document is
    unreadable" rather than "this has not been checked yet".
    """

    label: str = Field(default="", json_schema_extra={"example": "Keyword Match"})
    score: float | None = Field(default=None, json_schema_extra={"example": 85.0})
    max_score: float = Field(default=100.0)
    weight: float = Field(default=0.0, json_schema_extra={"example": 0.3})
    weighted_points: float = Field(default=0.0, json_schema_extra={"example": 25.5})
    points_lost: float = Field(default=0.0, json_schema_extra={"example": 15.0})
    not_measured: bool = Field(default=False)
    explanation: str = Field(default="")


class ATSBreakdown(BaseModel):
    """Full ATS score breakdown with category sub-scores."""

    ats_score: float = Field(..., json_schema_extra={"example": 87.5})
    max_score: float = Field(default=100.0)
    band: str = Field(default="", json_schema_extra={"example": "Good match"})
    total_points_lost: float = Field(..., json_schema_extra={"example": 12.5})
    summary: str = Field(default="")
    categories: dict[str, ATSCategoryBreakdown] = Field(default_factory=dict)
    improvement_areas: list[dict] = Field(default_factory=list)
    unfillable_gaps: dict = Field(default_factory=dict)
    pre_generation: bool = Field(default=False)
    is_honest_score: bool = Field(default=True)


class LayoutRecommendation(BaseModel):
    """Layout recommendation with reasons and alternatives."""

    recommended_layout: str = Field(..., json_schema_extra={"example": "international_ats"})
    recommended_name: str = Field(..., json_schema_extra={"example": "International ATS"})
    reason: str = Field(default="")
    ats_safety: str = Field(default="")
    ats_safety_score: float = Field(default=0.0)
    alternatives: list[dict] = Field(default_factory=list)
    page_guidance: dict = Field(default_factory=dict)
    detected_profile: dict = Field(default_factory=dict)


class ImprovementLoopResult(BaseModel):
    """Result of the ATS improvement loop."""

    initial_score: float = Field(..., json_schema_extra={"example": 72.0})
    final_score: float = Field(..., json_schema_extra={"example": 91.0})
    total_improvement: float = Field(..., json_schema_extra={"example": 19.0})
    target_score: float = Field(default=100.0)
    target_reached: bool = Field(default=False)
    rounds: list[dict] = Field(default_factory=list)
    total_rounds: int = Field(default=0)
    improvement_possible: bool = Field(default=True)
    explanation: str = Field(default="")


class AnalysisResponse(BaseModel):
    status: str = Field(default="success", json_schema_extra={"example": "success"})
    ats_match_score: float = Field(..., json_schema_extra={"example": 77.55})
    keyword_density_score: float = Field(..., json_schema_extra={"example": 100.0})
    matching_skills: list[str] = Field(
        ...,
        json_schema_extra={
            "example": ["Python", "FastAPI", "Docker", "PostgreSQL", "AWS", "Terraform"]
        },
    )
    missing_skills: list[str] = Field(..., json_schema_extra={"example": ["Kubernetes", "Redis"]})
    improvement_suggestions: list[str] = Field(
        ...,
        json_schema_extra={
            "example": [
                "Good key term match! Ensure keywords appear naturally inside achievements rather than just a standalone skills list."
            ]
        },
    )
    recommendation: RecommendationDetails | None = Field(default=None)

    # ------------------------------------------------------------
    # Parsed plain text of the uploaded resume, returned to the
    # caller so downstream services (auto-apply pipeline) can use it
    # without re-parsing the original PDF/DOCX.
    # ------------------------------------------------------------
    resume_text: str | None = Field(
        default=None,
        json_schema_extra={"example": "Example Candidate\nBerlin, Germany\n..."},
    )

    # ------------------------------------------------------------
    # New: transparent ATS breakdown
    # ------------------------------------------------------------
    ats_breakdown: ATSBreakdown | None = Field(default=None)

    # ------------------------------------------------------------
    # Plain-language findings, each with the action that resolves it
    # ------------------------------------------------------------
    ats_suggestions: list[dict] = Field(default_factory=list)

    # ------------------------------------------------------------
    # New: layout recommendation
    # ------------------------------------------------------------
    layout_recommendation: LayoutRecommendation | None = Field(default=None)

    # ------------------------------------------------------------
    # New: improvement loop result
    # ------------------------------------------------------------
    improvement_result: ImprovementLoopResult | None = Field(default=None)


class ErrorResponse(BaseModel):
    status: str = Field(default="error", json_schema_extra={"example": "error"})
    message: str = Field(
        ...,
        json_schema_extra={"example": "Could not extract readable text from resume file."},
    )
