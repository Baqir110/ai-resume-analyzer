"""
Cover-letter templates.

Each template is a prompt snippet that shapes the letter's structure and
tone. The generator in cover_letter.py injects the snippet and lets the
LLM fill in the content from the candidate's resume and the JD.

Adding a template = adding an entry here. No other file needs to change.
"""

from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "classic_professional": {
        "label": "Classic Professional",
        "description": "Formal, structured, three-paragraph letter. Safe for finance, law, government, and traditional enterprises.",
        "snippet": """
Write a classic professional cover letter with this structure:

PARAGRAPH 1 — Opening (3 sentences):
  - State the exact role and company.
  - Say who you are in one clause (current title or degree).
  - Name one concrete reason this company specifically interests you.

PARAGRAPH 2 — Evidence (5–7 sentences):
  - Present 2–3 accomplishments pulled directly from the resume.
  - Each accomplishment must be quantifiable (%, $, time saved, scale).
  - Tie each accomplishment to a specific requirement from the job description.

PARAGRAPH 3 — Close (2–3 sentences):
  - Restate interest in the role without repeating paragraph 1.
  - Mention availability for interview.
  - Thank the reader.

TONE: Formal, measured. No exclamation points.
LENGTH: 250–320 words total.
""",
    },
    "modern_concise": {
        "label": "Modern Concise",
        "description": "Short, direct, tech-friendly. Fits startups, scale-ups, and engineering roles.",
        "snippet": """
Write a modern concise cover letter with this structure:

PARAGRAPH 1 — Hook (2 sentences):
  - Lead with the single strongest fact about the candidate relevant to this role.
  - State the target role and company in the second sentence.

PARAGRAPH 2 — Proof (4–5 sentences):
  - Two accomplishments with numbers.
  - One sentence connecting the candidate's stack to the company's stack.
  - One sentence on why this role, at this company, right now.

PARAGRAPH 3 — CTA (1–2 sentences):
  - Direct ask: "I would like to discuss how I can contribute to [team/product]."
  - Sign-off.

TONE: Confident, direct, no fluff. First-person.
LENGTH: 180–240 words total.
STYLE: Short paragraphs. Avoid the word "passionate".
""",
    },
    "story_driven": {
        "label": "Story-Driven",
        "description": "Narrative arc, opens with a specific moment. Fits product, design, and roles where communication matters.",
        "snippet": """
Write a story-driven cover letter with this structure:

PARAGRAPH 1 — Scene (3 sentences):
  - Open with a specific moment from the candidate's experience that
    illustrates a trait the job description asks for.
  - Anchor the moment: where, when, what was at stake.
  - Pivot to the target role and company in the last sentence.

PARAGRAPH 2 — Consequence (4–5 sentences):
  - What the candidate did in that moment and what resulted.
  - Two more supporting facts from the resume with numbers.
  - Name the specific overlap between the candidate's trajectory and this role.

PARAGRAPH 3 — Forward (2 sentences):
  - One sentence on what the candidate wants to build or solve next.
  - One sentence inviting a conversation.

TONE: Warm, specific, human. Avoid generic adjectives ("dynamic", "results-driven").
LENGTH: 260–330 words total.
STYLE: Prose over lists.
""",
    },
    "value_first": {
        "label": "Value-First",
        "description": "Opens with a metric and a promise. Fits senior ICs, leads, and consulting roles.",
        "snippet": """
Write a value-first cover letter with this structure:

PARAGRAPH 1 — Thesis (2 sentences):
  - Sentence 1: The single largest measurable outcome the candidate has delivered.
  - Sentence 2: What the candidate will deliver in this role within the first 90 days, framed as a hypothesis.

PARAGRAPH 2 — Backing (5–6 sentences):
  - Three supporting accomplishments with numbers.
  - Each one must echo a top-three requirement from the job description.
  - Include one sentence naming a technology or process the candidate has shipped end-to-end.

PARAGRAPH 3 — Close (2 sentences):
  - A specific, low-friction ask: a 20-minute call.
  - Thank the reader.

TONE: Senior, matter-of-fact, no hedging language.
LENGTH: 220–280 words total.
RULES: No "I am writing to apply". No "as you can see from my resume".
""",
    },
}


def get_template(template_id: str | None) -> dict[str, str]:
    if not template_id:
        return TEMPLATES["classic_professional"]
    return TEMPLATES.get(template_id, TEMPLATES["classic_professional"])


def list_templates() -> list[dict[str, str]]:
    return [
        {"id": key, "label": v["label"], "description": v["description"]}
        for key, v in TEMPLATES.items()
    ]


def default_template_id() -> str:
    return "classic_professional"


__all__ = ["TEMPLATES", "default_template_id", "get_template", "list_templates"]
