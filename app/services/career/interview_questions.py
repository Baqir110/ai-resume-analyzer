"""
Interview question families.

Each family is a curated set of themes the LLM uses as anchors when
generating role-specific questions. The service fills in candidate and
role context; these lists only provide the angle.

Adding a family = adding an entry here.
"""

from __future__ import annotations

from typing import Optional

FAMILIES: dict[str, dict] = {
    "technical": {
        "label": "Technical Deep-Dive",
        "description": "System design, coding, debugging, tooling questions.",
        "snippet": """
Generate interview questions that probe technical depth. Cover these angles:

- System design: draw a diagram of a system the candidate has built.
- Trade-offs: choose between two specific technologies named in the JD.
- Debugging: walk through a real bug they fixed and the diagnostic method.
- Code quality: how they test, review, and structure code in a codebase.
- Scale: what changes at 10× the current load.

For each question, include:
  - the question,
  - what the interviewer is really evaluating,
  - a strong answer pattern from this candidate's resume,
  - a red flag answer to avoid.
""",
    },
    "behavioral": {
        "label": "Behavioral (STAR)",
        "description": "Conflict, ownership, failure, influence, ambiguity.",
        "snippet": """
Generate behavioral interview questions. Use STAR framing
(Situation, Task, Action, Result). Cover these angles:

- A time the candidate disagreed with a teammate or manager.
- A project that failed and what they learned.
- A time they took ownership beyond their job description.
- A moment they had to influence without authority.
- An ambiguous problem they had to scope themselves.

For each question:
  - write the question in interviewer voice,
  - list the STAR elements a strong answer must contain,
  - suggest which of the candidate's real experiences to draw on.
""",
    },
    "product": {
        "label": "Product & User Sense",
        "description": "Product thinking, prioritization, user empathy.",
        "snippet": """
Generate product-sense interview questions. Cover these angles:

- How would you improve [a product named in the JD]?
- Prioritize three features given these constraints.
- Describe a user problem the candidate has solved.
- How would you measure success of [a feature from the JD]?
- What would you cut from the roadmap and why?

For each question, include the framework a strong answer would use
(e.g. RICE, HEART) and the specific evidence from the resume that
makes the candidate's answer credible.
""",
    },
    "leadership": {
        "label": "Leadership & Mentoring",
        "description": "Team growth, direction, hiring, incident leadership.",
        "snippet": """
Generate leadership-focused interview questions. Cover these angles:

- How do you run a technical design review?
- A time you mentored someone through a difficult project.
- How do you handle a teammate who is underperforming?
- A time you led an incident or outage.
- How do you decide what NOT to build?

For each question:
  - the question,
  - what signal the interviewer is looking for,
  - a concrete example from the resume that fits,
  - follow-up questions the interviewer will likely ask.
""",
    },
    "mlops_devops": {
        "label": "DevOps / MLOps / Reliability",
        "description": "CI/CD, observability, on-call, incident response, ML lifecycle.",
        "snippet": """
Generate operations-focused interview questions. Cover these angles:

- Walk through your CI/CD pipeline and what you would change.
- How do you monitor a production ML model for drift?
- Describe your ideal postmortem for a P1 incident.
- How do you handle secrets management and least-privilege access?
- What is your rollback strategy for a bad deploy?

For each question:
  - the question,
  - what a strong answer covers (tools, trade-offs, cadence),
  - how to answer using the candidate's actual stack from the resume,
  - one "gotcha" question the interviewer might follow up with.
""",
    },
}


def get_family(family_id: Optional[str]) -> dict:
    if not family_id:
        return FAMILIES["technical"]
    return FAMILIES.get(family_id, FAMILIES["technical"])


def list_families() -> list[dict[str, str]]:
    return [
        {
            "id": key,
            "label": value["label"],
            "description": value["description"],
        }
        for key, value in FAMILIES.items()
    ]


def default_family_id() -> str:
    return "technical"


__all__ = ["FAMILIES", "get_family", "list_families", "default_family_id"]
