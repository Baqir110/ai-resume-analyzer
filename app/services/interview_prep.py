import json
from typing import Dict, Any, List
from app.services.llm_provider import LLMService


class InterviewPrepService:
    @classmethod
    def generate_interview_prep(
        cls,
        resume_text: str,
        job_description: str,
        missing_skills: List[str],
        provider: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Generates role-specific technical/behavioral interview questions and missing-skill gap defense strategies.
        """
        prompt = f"""
You are a senior technical interviewer. Analyze the candidate's resume, the target job description, and the identified missing skills.

Missing Skills Identified: {', '.join(missing_skills) if missing_skills else 'None'}

Generate a structured interview preparation guide in valid JSON format with keys:
1. "technical_questions": List of 5 technical questions tailored to the job requirements.
2. "behavioral_questions": List of 3 behavioral STAR-method questions focusing on problem-solving and collaboration.
3. "gap_defenses": List of objects with "missing_skill", "strategic_answer", and "transferable_angle" explaining how the candidate can pivot when asked about skills they lack.

Return ONLY raw JSON.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""

        raw_response = LLMService.generate_response(
            prompt=prompt,
            provider=provider,
            temperature=0.2,
        )

        try:
            cleaned_json = raw_response.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:-3].strip()
            elif cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:-3].strip()

            data = json.loads(cleaned_json)
            return data
        except Exception as exc:
            return {
                "technical_questions": [
                    "Describe your core technical stack and how it fits this role."
                ],
                "behavioral_questions": [
                    "Tell me about a time you resolved an unexpected production issue."
                ],
                "gap_defenses": [
                    {
                        "missing_skill": skill,
                        "strategic_answer": f"Highlight foundational knowledge in related technologies and fast learning speed for {skill}.",
                        "transferable_angle": "Emphasize architectural concepts over specific tool syntax.",
                    }
                    for skill in missing_skills[:3]
                ],
                "error": f"JSON parsing fallback applied: {exc}",
            }
