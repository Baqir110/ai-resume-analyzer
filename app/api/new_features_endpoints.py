"""API endpoints for new features."""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import Optional, List

from app.services.analysis.score_explainability import ScoreExplainabilityEngine
from app.services.analysis.market_insights import MarketInsightsEngine
from app.services.analysis.skill_roadmap import SkillProgressionTracker
from app.services.analysis.authenticity_checker import AuthenticityChecker
from app.services.tracking.version_manager import ResumeVersionManager
from app.services.tracking.collaborative_feedback import CollaborativeFeedbackManager
from app.services.career.interview_simulator import InterviewSimulator
from app.services.parsing.resume_parser import extract_text_from_file

router = APIRouter()

scope_engine = ScoreExplainabilityEngine()
market_engine = MarketInsightsEngine()
skill_tracker = SkillProgressionTracker()
auth_checker = AuthenticityChecker()
version_manager = ResumeVersionManager()
feedback_manager = CollaborativeFeedbackManager()
interview_sim = InterviewSimulator()


# Score Explainability
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


# Market Insights
@router.get("/market-insights")
async def market_insights(
    role: str,
    location: str,
    seniority: str = "mid",
):
    """Get job market intelligence for a role."""
    insights = await market_engine.get_market_insights(role, location, seniority)
    return {"status": "success", "insights": insights}


# Skill Roadmap
@router.post("/skill-roadmap")
async def generate_skill_roadmap(
    current_skills: List[str] = Form(...),
    target_role: str = Form(...),
    months_available: int = Form(6),
):
    """Generate personalized skill development roadmap."""
    roadmap = await skill_tracker.generate_learning_roadmap(
        current_skills,
        target_role,
        months_available,
    )
    return {"status": "success", "roadmap": roadmap}


# Resume Authenticity
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


# Resume Versioning
@router.post("/versions/save")
async def save_resume_version(
    user_id: str = Form(...),
    original_text: str = Form(...),
    optimized_text: str = Form(...),
    ats_score: int = Form(...),
    job_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
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


@router.get("/versions/{user_id}")
async def list_versions(user_id: str):
    """List user's resume versions."""
    versions = version_manager.get_versions(user_id)
    return {"status": "success", "versions": versions}


@router.get("/versions/compare")
async def compare_versions(
    version_id_1: int,
    version_id_2: int,
):
    """Compare two resume versions."""
    comparison = version_manager.compare_versions(version_id_1, version_id_2)
    return {"status": "success", "comparison": comparison}


# Interview Simulator
@router.post("/interview/question")
async def generate_interview_question(
    family: str = Form(...),
    context: str = Form("Software Engineer"),
):
    """Generate interview practice question."""
    question = await interview_sim.generate_interview_question(family, context)
    return {"status": "success", "question": question}


@router.post("/interview/evaluate")
async def evaluate_interview_answer(
    question_id: int = Form(...),
    user_answer: str = Form(...),
    expected_topics: List[str] = Form(...),
):
    """Evaluate interview practice answer."""
    feedback = await interview_sim.evaluate_answer(
        question_id,
        user_answer,
        expected_topics,
    )
    return {"status": "success", "feedback": feedback}


# Collaborative Feedback
@router.post("/feedback/thread")
async def create_feedback_thread(
    resume_id: int = Form(...),
    section: str = Form(...),
    author: str = Form(...),
    line_number: Optional[int] = Form(None),
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
    thread_id: int = Form(...),
    author: str = Form(...),
    comment: str = Form(...),
    suggestion: Optional[str] = Form(None),
    category: str = Form("clarity"),
    severity: str = Form("medium"),
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
