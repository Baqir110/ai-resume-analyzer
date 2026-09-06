# AI Resume & CV Optimization Hub

This context covers the career-application documents the product analyzes and generates, with a focus on preserving the candidate's supplied career facts.

## Language

**German Minimal ATS**:
The selectable, opt-in German Lebenslauf layout identified by `german_minimal_ats`; it is exposed in the API and dashboard as “German Minimal ATS (Single-Column)”. It is a single-column, text-only format with German headings and no photo, tables, icons, sidebars, headers, or footers.
_Avoid_: German ATS, minimal German CV

**Factual validation gate**:
A generation check that compares normalized, immutable source career facts—companies, job titles, dates, and degrees—with generated LaTeX and rejects an output that omits or alters them. Failures return HTTP `422 Unprocessable Entity` with the affected invariants.
_Avoid_: best-effort validation, soft fact check

**Candidate header**:
The name and contact details rendered at the top of a generated CV, derived from the uploaded résumé. Fields that cannot be reliably extracted are omitted.
_Avoid_: template identity, hard-coded profile
