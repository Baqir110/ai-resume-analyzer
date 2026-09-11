"""Streaming response support for real-time updates."""
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.core.event_log import log_event
from app.services.parsing.resume_parser import extract_text_from_file
from app.services.analysis.ats_analyzer import analyze_resume_content

router = APIRouter()


class StreamingAnalyzer:
    """Analyzer that emits events as it processes."""
    
    async def analyze_stream(
        self,
        resume_text: str,
        job_description: str,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream analysis results as they become available.
        
        Yields:
            JSON strings with progress updates
        """
        try:
            # Stage 1: Parsing
            yield self._format_event({
                'stage': 'parsing',
                'status': 'in_progress',
                'message': 'Extracting resume text...',
            })
            await asyncio.sleep(0.1)  # Simulate work
            
            yield self._format_event({
                'stage': 'parsing',
                'status': 'complete',
                'text_length': len(resume_text),
            })
            
            # Stage 2: ATS Scoring
            yield self._format_event({
                'stage': 'ats_scoring',
                'status': 'in_progress',
                'message': 'Computing ATS compatibility...',
            })
            
            analysis = analyze_resume_content(resume_text, job_description)
            
            yield self._format_event({
                'stage': 'ats_scoring',
                'status': 'complete',
                'ats_score': analysis.get('ats_match_score'),
                'keyword_density': analysis.get('keyword_density_score'),
            })
            
            # Stage 3: Skill Analysis
            yield self._format_event({
                'stage': 'skill_analysis',
                'status': 'complete',
                'matching_skills': analysis.get('matching_skills', []),
                'missing_skills': analysis.get('missing_skills', []),
            })
            
            # Stage 4: LLM Enhancement (optional)
            yield self._format_event({
                'stage': 'llm_enhancement',
                'status': 'in_progress',
                'message': 'Generating AI suggestions...',
            })
            
            await asyncio.sleep(1)  # Simulate LLM call
            
            yield self._format_event({
                'stage': 'llm_enhancement',
                'status': 'complete',
                'suggestions': analysis.get('improvement_suggestions', [])[:3],
            })
            
            # Final: Complete
            yield self._format_event({
                'stage': 'complete',
                'status': 'success',
                'message': 'Analysis complete',
                'full_result': analysis,
            })
        
        except Exception as e:
            yield self._format_event({
                'stage': 'error',
                'status': 'failed',
                'error': str(e),
            })
    
    def _format_event(self, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event (SSE)."""
        return f"data: {json.dumps(data)}\n\n"


@router.post("/analyze-stream")
async def analyze_resume_streaming(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
):
    """Stream resume analysis results in real-time.
    
    Uses Server-Sent Events (SSE) to emit progress updates.
    """
    from app.core.event_log import new_request_id, set_request_id
    
    resume_text = await extract_text_from_file(resume_file)
    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract resume text",
        )
    
    request_id = new_request_id()
    set_request_id(request_id)
    
    analyzer = StreamingAnalyzer()
    
    async def event_generator():
        async for event in analyzer.analyze_stream(
            resume_text,
            job_description,
            request_id,
        ):
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
