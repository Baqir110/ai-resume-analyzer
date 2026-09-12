import json
from typing import Any

from app.services.career.interview_questions import get_family
from app.services.llm.provider import LLMService


class InterviewPrepService:
    @classmethod
    def generate_interview_prep(
        cls,
        resume_text: str,
        job_description: str,
        missing_skills: list[str],
        family: str = "technical",
        provider: str = "gemini",
    ) -> dict[str, Any]:
        """
        Generates role-specific interview questions and gap defenses.

        The `family` parameter selects the focus of the generated questions:
          - technical       — system design, coding, debugging, tooling
          - behavioral      — STAR-format conflict, ownership, failure
          - product         — product thinking, prioritization, user sense
          - leadership      — team growth, direction, hiring, incidents
          - mlops_devops    — CI/CD, observability, on-call, ML lifecycle
        """
        family_def = get_family(family)
        family_snippet = family_def["snippet"].strip()

        prompt = f"""
You are a senior technical interviewer. Analyze the candidate's resume, the
target job description, and the identified missing skills.

Missing Skills Identified: {", ".join(missing_skills) if missing_skills else "None"}

QUESTION FAMILY: {family_def["label"]}
{family_def["description"]}

QUESTION FAMILY INSTRUCTIONS
---------------------------
{family_snippet}

OUTPUT FORMAT
-------------
Return a single JSON object with exactly these keys:

1. "technical_questions"   — array of 5 strings tailored to the job requirements
                             and consistent with the question family above.
2. "behavioral_questions"  — array of 3 strings using the STAR method.
3. "gap_defenses"          — array of objects, each with:
                             - "missing_skill"     : the skill from Missing Skills
                             - "strategic_answer"  : how to pivot when asked about it
                             - "transferable_angle": the related expertise to lead with

Return ONLY the raw JSON object. No prose, no code fences, no explanation.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
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

            data = json.loads(cleaned_json)
            # Annotate the response with the family that was used, so the UI
            # can display which prompt the user selected.
            if isinstance(data, dict):
                data.setdefault("_meta", {})
                data["_meta"]["family"] = family
                data["_meta"]["family_label"] = family_def["label"]
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
                        "strategic_answer": (
                            f"Highlight foundational knowledge in related "
                            f"technologies and fast learning speed for {skill}."
                        ),
                        "transferable_angle": (
                            "Emphasize architectural concepts over specific " "tool syntax."
                        ),
                    }
                    for skill in missing_skills[:3]
                ],
                "_meta": {
                    "family": family,
                    "family_label": family_def["label"],
                    "error": f"JSON parsing fallback applied: {exc}",
                },
            }
