import re
from typing import Dict, Any, Optional
from app.services.llm_provider import LLMService


def _escape_latex(text: str) -> str:
    """Escapes special LaTeX characters in text."""
    if not text:
        return ""
    chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in chars.keys()))
    return pattern.sub(lambda match: chars[match.group(0)], text)


class CoverLetterService:
    @classmethod
    def generate_cover_letter_and_outreach(
        cls,
        resume_text: str,
        job_description: str,
        company_name: str = "Target Company",
        tone: str = "formal",
        provider: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Generates a 3-paragraph cover letter, a 150-word cold email, and raw LaTeX source.
        """
        tone_instructions = {
            "formal": "Professional, executive, and structured.",
            "startup": "Conversational, enthusiastic, outcome-driven, and engaging.",
            "technical": "Direct, deep technical emphasis on architecture, tools, and quantifiable achievements.",
        }

        selected_tone = tone_instructions.get(tone.lower(), tone_instructions["formal"])

        prompt = f"""
You are an expert executive resume writer and career coach.
Analyze the following RESUME and JOB DESCRIPTION to create tailored outreach materials for {company_name}.

Tone/Style Requirement: {selected_tone}

Task 1: COVER LETTER
Write a compelling 3-paragraph cover letter:
- Paragraph 1: High-impact opening connecting candidate experience to the target role at {company_name}.
- Paragraph 2: Core technical achievements and specific alignment with job requirements.
- Paragraph 3: Strategic closing statement with a confident call to action.

Task 2: COLD OUTREACH MESSAGE
Write a concise, 120-150 word LinkedIn message or cold email to the hiring manager. Focus on direct value creation.

CRITICAL OUTPUT FORMAT:
Return EXACTLY two sections separated by "---SECTION_BREAK---". Do not add introductory markdown text.

[COVER LETTER CONTENT]
---SECTION_BREAK---
[COLD OUTREACH CONTENT]

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""

        raw_response = LLMService.generate_response(
            prompt=prompt,
            provider=provider,
            temperature=0.3,
        )

        parts = raw_response.split("---SECTION_BREAK---")
        cover_letter_body = parts[0].strip() if len(parts) > 0 else raw_response.strip()
        cold_outreach_body = (
            parts[1].strip()
            if len(parts) > 1
            else "Outreach message generation unavailable."
        )

        # Compile matching LaTeX Source
        tex_source = cls._build_latex_cover_letter(cover_letter_body, company_name)

        return {
            "company_name": company_name,
            "tone": tone,
            "cover_letter": cover_letter_body,
            "cold_outreach": cold_outreach_body,
            "latex_source": tex_source,
        }

    @staticmethod
    def _build_latex_cover_letter(body_text: str, company_name: str) -> str:
        safe_body = _escape_latex(body_text).replace("\n\n", "\n\n\\vspace{0.8em}\n")
        safe_company = _escape_latex(company_name)

        return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{parskip}}

\definecolor{{primaryColor}}{{RGB}}{{37, 99, 235}}

\pagestyle{{empty}}

\begin{{document}}

{{\Large \textbf{{\color{{primaryColor}} Cover Letter -- Application for {safe_company}}}}}
\vspace{{1.5em}}

{safe_body}

\vspace{{2em}}
Sincerely,\\
\textbf{{Applicant}}

\end{{document}}
"""
