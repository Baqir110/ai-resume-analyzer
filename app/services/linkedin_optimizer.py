import json
from typing import Dict, Any
from app.services.llm_provider import LLMService


class LinkedInOptimizerService:
    @classmethod
    def optimize_profile(
        cls,
        resume_text: str,
        target_role: str = "Software Engineer",
        provider: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Generates 3 punchy LinkedIn headlines and a 1st-person LinkedIn About section.
        """
        prompt = f"""
You are an executive LinkedIn personal branding consultant.
Analyze the provided RESUME to generate optimized LinkedIn profile content for a target role of: "{target_role}".

Requirements:
1. "headlines": List of 3 distinct, high-converting headlines (under 220 characters each) separated by '|'.
2. "about_section": A compelling, 1st-person "About" narrative (3 short paragraphs) emphasizing domain expertise, key tech stack, and passion for solving core industry problems. Include a "Specialties" bulleted list at the bottom.

Return valid JSON with keys "headlines" and "about_section".

RESUME:
{resume_text}
"""

        raw_response = LLMService.call_llm(
            prompt=prompt,
            provider=provider,
        )

        try:
            cleaned_json = raw_response.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:-3].strip()
            elif cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:-3].strip()

            return json.loads(cleaned_json)
        except Exception:
            return {
                "headlines": [
                    f"{target_role} | Systems Architecture & High-Performance Engineering",
                    f"Senior Engineer | Python, FastAPI, Cloud Infrastructure",
                    f"Data & Software Specialist | Building Scalable Services",
                ],
                "about_section": raw_response.strip(),
            }