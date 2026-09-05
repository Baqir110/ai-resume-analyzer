import asyncio
import io
from typing import List, Dict, Any

from fastapi import UploadFile

from app.services.analyzer import analyze_resume_content
from app.services.parser import extract_text_from_file

class BulkAnalyzerService:
    @staticmethod
    async def analyze_single_cv(
        file_bytes: bytes, filename: str, job_description: str
    ) -> Dict[str, Any]:
        """
        Parses and evaluates a single candidate file against the job description.
        """
        # 1. Parse text from document stream
        upload = UploadFile(
            file=io.BytesIO(file_bytes),
            filename=filename,
        )
        parsed_text = await extract_text_from_file(upload)
        
        # 2. Run ATS scoring and keyword extraction
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None,
            analyze_resume_content,
            parsed_text,
            job_description,
        )
        
        return {
            "filename": filename,
            "ats_score": analysis.get("ats_match_score", 0),
            "keyword_density": analysis.get("keyword_density_score", 0),
            "matching_skills": analysis.get("matching_skills", []),
            "missing_skills": analysis.get("missing_skills", []),
            "parsed_character_count": len(parsed_text)
        }

    @classmethod
    async def process_batch(
        cls, files: List[tuple[bytes, str]], job_description: str
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel analysis across uploaded CV files and ranks results.
        """
        tasks = [
            cls.analyze_single_cv(content, filename, job_description)
            for content, filename in files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for res in results:
            if not isinstance(res, Exception):
                valid_results.append(res)
                
        # Sort candidates descending by ATS match score
        valid_results.sort(key=lambda x: x["ats_score"], reverse=True)
        return valid_results