"""API endpoints for new features."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.utils import parse_json_list as _parse_json_list
from app.core.security import require_api_key
from app.services.analysis.authenticity_checker import AuthenticityChecker
from app.services.analysis.market_insights import MarketInsightsEngine
from app.services.analysis.score_explainability import ScoreExplainabilityEngine
from app.services.analysis.skill_roadmap import SkillProgressionTracker
from app.services.career.interview_simulator import InterviewSimulator
from app.services.parsing.resume_parser import extract_text_from_file
from app.services.tracking.collaborative_feedback import CollaborativeFeedbackManager
from app.services.tracking.version_manager import ResumeVersionManager

router = APIRouter(dependencies=[Depends(require_api_key)])

scope_engine = ScoreExplainabilityEngine()
market_engine = MarketInsightsEngine()
skill_tracker = SkillProgressionTracker()
auth_checker = AuthenticityChecker()
version_manager = ResumeVersionManager()
feedback_manager = CollaborativeFeedbackManager()
interview_sim = InterviewSimulator()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# def _parse_json_list(raw: str | None, field_name: str) -> list[str]:
#     """Parse a JSON-encoded list from a form field.

#     Accepts either a JSON array (``["a","b"]``) or a comma-separated
#     string (``"a,b,c"``) as a fallback for convenience.
#     """
#     if raw is None or not str(raw).strip():
#         return []
#     text = str(raw).strip()
#     try:
#         parsed = json.loads(text)
#         if isinstance(parsed, list):
#             return [str(x).strip() for x in parsed if str(x).strip()]
#     except json.JSONDecodeError:
#         pass
#     # Fallback: comma-separated
#     return [item.strip() for item in text.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Score Explainability
# ---------------------------------------------------------------------------


@router.post("/score-breakdown")
async def score_breakdown(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
):
    """Get detailed score breakdown with explainability."""
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text")

    breakdown = scope_engine.explain_score(resume_text, job_description)
    return {"status": "success", "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Market Insights
# ---------------------------------------------------------------------------


@router.get("/market-insights")
async def market_insights(
    role: str = Query(..., min_length=1, max_length=300),
    location: str = Query(..., min_length=1, max_length=300),
    seniority: str = "mid",
    provider: str | None = None,
    route_mode: str | None = None,
):
    """Get AI-estimated job market intelligence for a role."""
    insights = await market_engine.get_market_insights(
        role,
        location,
        seniority,
        provider=provider,
        route_mode=route_mode,
    )
    return {"status": "success", "insights": insights}


# ---------------------------------------------------------------------------
# Skill Roadmap
# ---------------------------------------------------------------------------


@router.post("/skill-roadmap")
async def generate_skill_roadmap(
    current_skills: str = Form(
        ...,
        max_length=20_000,
        description='JSON array of skills, e.g. ["python", "sql"]',
    ),
    target_role: str = Form(..., max_length=300),
    months_available: int = Form(6, ge=1, le=120),
    provider: str | None = Form(None),
    route_mode: str | None = Form(None),
):
    """Generate a personalised, AI-driven skill development roadmap."""
    skills = _parse_json_list(current_skills, "current_skills")
    if not skills:
        raise HTTPException(
            status_code=400,
            detail="current_skills must contain at least one skill.",
        )

    roadmap = await skill_tracker.generate_learning_roadmap(
        skills,
        target_role,
        months_available,
        provider=provider,
        route_mode=route_mode,
    )
    return {"status": "success", "roadmap": roadmap}


# ---------------------------------------------------------------------------
# Resume Authenticity
# ---------------------------------------------------------------------------


@router.post("/check-authenticity")
async def check_authenticity(
    resume_file: UploadFile = File(...),
):
    """Check resume for plagiarism and authenticity."""
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract resume text")

    report = await auth_checker.check_authenticity(resume_text)
    return {"status": "success", "report": report}


# ---------------------------------------------------------------------------
# Resume Versioning
# ---------------------------------------------------------------------------


@router.post("/versions/save")
async def save_resume_version(
    user_id: str = Form(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$"),
    original_text: str = Form(..., min_length=1, max_length=200_000),
    optimized_text: str = Form(..., min_length=1, max_length=200_000),
    ats_score: int = Form(..., ge=0, le=100),
    job_id: str | None = Form(None, max_length=256),
    notes: str | None = Form(None, max_length=10_000),
):
    """Save a resume version."""
    version_id = version_manager.save_version(
        user_id,
        original_text,
        optimized_text,
        ats_score,
        job_id,
        notes=notes,
    )
    return {"status": "success", "version_id": version_id}


@router.get("/versions/compare")
async def compare_versions(
    version_id_1: int,
    version_id_2: int,
):
    """Compare two resume versions."""
    comparison = version_manager.compare_versions(version_id_1, version_id_2)
    if "error" in comparison:
        raise HTTPException(status_code=404, detail=comparison["error"])
    return {"status": "success", "comparison": comparison}


@router.get("/versions/{user_id}")
async def list_versions(user_id: str):
    """List user's resume versions."""
    versions = version_manager.get_versions(user_id)
    return {"status": "success", "versions": versions}


# ---------------------------------------------------------------------------
# Interview Simulator
# ---------------------------------------------------------------------------


@router.post("/interview/question")
async def generate_interview_question(
    family: str = Form(..., min_length=1, max_length=100),
    context: str = Form("Software Engineer", max_length=2_000),
):
    """Generate interview practice question."""
    question = await interview_sim.generate_interview_question(family, context)
    return {"status": "success", "question": question}


@router.post("/interview/evaluate")
async def evaluate_interview_answer(
    question_id: int = Form(..., ge=1),
    user_answer: str = Form(..., min_length=1, max_length=20_000),
    expected_topics: str = Form(
        ...,
        description='JSON array of expected topics, e.g. ["oop", "git"]',
    ),
):
    """Evaluate interview practice answer.

    ``expected_topics`` must be a JSON array string or comma-separated list.
    """
    topics = _parse_json_list(expected_topics, "expected_topics")
    feedback = await interview_sim.evaluate_answer(
        question_id,
        user_answer,
        topics,
    )
    return {"status": "success", "feedback": feedback}


# ---------------------------------------------------------------------------
# Collaborative Feedback
# ---------------------------------------------------------------------------


@router.post("/feedback/thread")
async def create_feedback_thread(
    resume_id: int = Form(..., ge=1),
    section: str = Form(..., min_length=1, max_length=200),
    author: str = Form(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$"),
    line_number: int | None = Form(None, ge=1, le=100_000),
):
    """Create feedback thread for resume section."""
    thread_id = feedback_manager.create_feedback_thread(
        resume_id,
        section,
        line_number,
        author,
    )
    return {"status": "success", "thread_id": thread_id}


@router.post("/feedback/comment")
async def add_feedback_comment(
    thread_id: int = Form(..., ge=1),
    author: str = Form(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$"),
    comment: str = Form(..., min_length=1, max_length=20_000),
    suggestion: str | None = Form(None, max_length=20_000),
    category: str = Form("clarity", max_length=64),
    severity: str = Form("medium", max_length=64),
):
    """Add comment to feedback thread."""
    comment_id = feedback_manager.add_comment(
        thread_id,
        author,
        comment,
        suggestion,
        category,
        severity,
    )
    return {"status": "success", "comment_id": comment_id}


@router.get("/feedback/{resume_id}")
async def get_resume_feedback(
    resume_id: int,
    unresolved_only: bool = False,
):
    """Get all feedback for a resume."""
    feedback = feedback_manager.get_resume_feedback(resume_id, unresolved_only)
    summary = feedback_manager.get_feedback_summary(resume_id)
    return {
        "status": "success",
        "feedback": feedback,
        "summary": summary,
    }
