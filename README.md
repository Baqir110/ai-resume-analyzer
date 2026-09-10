# AI Resume & CV Optimization Hub

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

An AI-powered resume analysis, optimization, and career application platform built with FastAPI, Streamlit, and multiple LLM providers.

The platform analyzes resumes against job descriptions, identifies missing skills and keywords, generates job-tailored CVs (DOCX and LaTeX/PDF), supports bulk resume analysis, tracks LLM usage and provider quotas, and includes additional career workflow tools such as cover letters, interview preparation, LinkedIn optimization, application tracking, and resume audit analysis.

---

## Table of Contents

- [Overview](#overview)
- [What's New](#whats-new)
- [Use Cases](#use-cases)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [LLM Providers](#llm-providers)
- [Observability](#observability)
- [Document Generation](#document-generation)
- [Application Tracking](#application-tracking)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AI Resume & CV Optimization Hub is a resume intelligence and career application platform designed to evaluate candidate resumes against specific job descriptions and produce targeted improvements.

The system combines:

- TF-IDF and cosine similarity for deterministic ATS scoring
- Dynamic keyword and skill extraction
- Skill-gap analysis
- LLM-powered resume and CV optimization
- Multi-provider AI routing with automatic fallback
- Automatic layout selection based on job-description language
- Pre-flight resume translation for cross-language generation
- Bulk candidate analysis
- Word-level document diffing
- ATS-friendly DOCX generation
- German Lebenslauf PDF generation through LaTeX
- Per-provider quota and token usage tracking
- Structured pipeline event logging
- Career workflow tools (cover letters, interview prep, LinkedIn, audit, tracker)

The optimization workflow follows an additive approach: existing career history, employers, dates, education, projects, and other factual information are preserved while relevant job-specific terminology and improvements are incorporated.

---

## What's New

- **Automatic layout selection.** The backend detects the job description's language and picks the appropriate CV template (`international_ats` for English JDs, `german_corporate` for German JDs) without user intervention.
- **Pre-flight language normalization.** When generating a CV in a language different from the resume's source language, the resume text is translated before entering the generation pipeline. This avoids the LaTeX-level translation issues that occur when translating generated output.
- **Quota and token tracking.** Per-provider RPM/RPD/TPM usage is aggregated from the local processing log and surfaced in the dashboard with live progress bars. Supports declared limits per provider.
- **Pipeline event logging.** All stages — parsing, analysis, translation, LaTeX generation, PDF compilation, DOCX generation — emit structured JSONL events with a shared `request_id`. The dashboard exposes a filterable timeline.
- **Multi-provider fallback chain.** Provider failures automatically retry against the next available provider in a configurable fallback chain.
- **Fragment-scoped Streamlit dashboard.** Expensive sections (sidebar, CV generation, career suite, log panel) are isolated via `@st.fragment` so widget interactions re-execute only the affected section.
- **Cache layers.** Sidebar fetchers use `@st.cache_data` with short TTLs to prevent the classic Streamlit rerun storm.
- **Factual invariant validation.** The German Minimal ATS layout verifies that company names, dates, and degree titles survive into the generated document before the PDF is compiled.
- **Keyword-dump stripping.** Post-processing removes LLM-injected "Ergänzende Terminologie" / "Additional Keywords" sections that ATS parsers misread.
- **Domain-organized service layer.** `app/services/` is now split into subpackages (`parsing/`, `analysis/`, `llm/`, `cv/`, `career/`, `bulk/`, `tracking/`), so each module's responsibility is obvious from its import path.

---

## Use Cases

### Job Application Tailoring
Analyze a resume against a specific job description and identify ATS compatibility, matching skills, missing skills, missing keywords, and improvement opportunities.

### Resume Optimization
Generate a job-tailored version of an existing resume while preserving the candidate's original career history and factual information.

### Cross-Language CV Generation
Generate a CV in the target market's language even when the source resume is in another language, via pre-flight translation and automatic layout selection.

### Bulk Candidate Screening
Analyze multiple resumes against a single job description and rank candidates according to ATS compatibility and skill coverage.

### Resume Audit
Perform structured analysis of resume quality, content coverage, and job alignment.

### Career Workflow
Generate cover letters, prepare for interviews, optimize LinkedIn content, and track applications — all from the same workspace.

---

## Key Features

### Multi-Format Resume Parsing
Supports PDF, DOCX, and TXT. Content is extracted, normalized, and passed to the analysis pipeline. Every parse emits a `parse_completed` or `parse_failed` event with timing and character count.

### Hybrid ATS Scoring
Combines traditional NLP/ML techniques with AI-assisted processing:
- TF-IDF vectorization and cosine similarity
- Keyword extraction and density analysis
- Technical skill detection
- Matching-skill and missing-skill analysis
- Improvement suggestions

### Additive LLM Optimization
The optimization engine enriches existing resume content rather than rewriting it. Company names, job titles, employment dates, degree information, projects, and existing career history are preserved while relevant job-description terminology is incorporated.

### Multi-Provider LLM Architecture
Configurable AI providers:
- Google GenAI (native)
- Groq (native)
- OpenRouter (native)
- DeepSeek (native)
- OpenAI (native)
- Anthropic Claude (native or gateway)
- Experiential Labs Gateway
- Ollama (local)

The `LLMService` provides a uniform interface with automatic fallback: if a provider fails, the next one in the chain is tried. Every call is logged with token usage, cost estimate, and duration.

### Automatic Layout Selection
The backend inspects the job description's function-word profile to classify its language, then selects the appropriate CV template. `international_ats` for English job posts, `german_corporate` for German. Users can override the choice.

### Pre-flight Language Normalization
When the target layout requires a language different from the resume's, the resume text is translated before entering the generation pipeline. This avoids the LaTeX-mangling issues that arise when translating generated output.

### Keyword-Dump Stripping
Post-processing removes LLM-injected "Additional Keywords" sections that break ATS parsers. Two stages: whole-line section stripping and inline `\textbf{...}` block stripping.

### Factual Invariant Validation
The German Minimal ATS layout extracts role / company / date triples from the source resume and verifies each survives into the generated document. Generation is rejected with a 422 if any invariant is missing.

### Document Generation
- **DOCX**: ATS-friendly single-column Word documents
- **LaTeX → PDF**: Multiple CV templates (German Corporate, German Classic, German Modern, German Minimal ATS, International ATS, Standard, HR Executive Gold) compiled locally with `pdflatex`

### Bulk Analysis
Multiple resumes processed against one job description, ranked by ATS score.

### Career Workflow Tools
Dedicated services for resume auditing, cover-letter generation, interview preparation, LinkedIn optimization, and application tracking.

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard   │
                    │      User Interface     │
                    └────────────┬────────────┘
                                 │  (HTTP)
                                 ▼
                    ┌─────────────────────────┐
                    │      FastAPI API        │
                    │   REST Endpoint Layer   │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
  ┌───────────┐          ┌──────────────┐         ┌─────────────┐
  │  Parser   │          │   Analyzer   │         │   Tracker   │
  └─────┬─────┘          └──────┬───────┘         └─────────────┘
        │                       │
        │                       ▼
        │                ┌───────────────┐
        │                │ Skill / ATS   │
        │                │   Analysis    │
        │                └───────┬───────┘
        │                        │
        └────────────┬───────────┘
                     ▼
             ┌─────────────────┐        ┌──────────────────┐
             │   LLM Service   │◄───────│  Quota Tracker   │
             │ Provider Router │        │ (rate limits,    │
             └────────┬────────┘        │  token usage)    │
                      │                 └──────────────────┘
       ┌──────────────┼─────────────────────┐
       │              │                     │
       ▼              ▼                     ▼
   Native APIs   Gateway APIs           Ollama
       │              │                     │
       └──────────────┼─────────────────────┘
                      ▼
             ┌─────────────────┐
             │ Document / CV   │
             │   Generation    │
             └────────┬────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
         DOCX                  LaTeX
                                 │
                                 ▼
                               PDF

      All stages emit events to:
      ┌──────────────────────────────┐
      │  data/llm_processing.jsonl   │
      │  (shared request_id)         │
      └──────────────────────────────┘
```

---

## Technology Stack

| Category            | Technology                                                |
| ------------------- | --------------------------------------------------------- |
| Language            | Python 3.10+                                              |
| API Framework       | FastAPI                                                   |
| Frontend            | Streamlit (fragment-scoped)                               |
| NLP / ML            | scikit-learn (TF-IDF, cosine similarity)                  |
| PDF Parsing         | pypdf                                                     |
| DOCX Parsing        | python-docx                                               |
| DOCX Generation     | python-docx                                               |
| PDF Compilation     | pdflatex                                                  |
| Configuration       | `.env` environment variables                              |
| Database            | SQLite                                                    |
| AI Providers        | Google GenAI, Groq, OpenRouter, DeepSeek, OpenAI, Anthropic, Experiential Labs, Ollama |
| Testing             | pytest                                                    |
| API Server          | Uvicorn                                                   |

---

## Project Structure

The `services/` layer is organized by domain. Each subpackage groups modules that share a single responsibility.

```text
ai-resume-analyzer/
│
├── app/
│   ├── api/
│   │   └── endpoints.py                 # FastAPI routes + pipeline logging decorator
│   │
│   ├── core/
│   │   ├── config.py                    # Application settings
│   │   └── event_log.py                 # Shared JSONL pipeline event logger
│   │
│   ├── models/
│   │   └── schemas.py                   # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── parsing/
│   │   │   └── resume_parser.py         # PDF / DOCX / TXT extraction
│   │   │
│   │   ├── analysis/
│   │   │   ├── ats_analyzer.py          # TF-IDF + keyword + skill gap analysis
│   │   │   └── suggestions.py           # ATS improvement recommendations
│   │   │
│   │   ├── llm/
│   │   │   ├── provider.py              # Multi-provider LLM router + fallback
│   │   │   └── quota_tracker.py         # RPM/RPD/TPM usage aggregation
│   │   │
│   │   ├── cv/
│   │   │   ├── optimizer.py             # CV optimization + layout selection
│   │   │   ├── latex_generator.py       # LaTeX templates + PDF compilation
│   │   │   └── diff_preview.py          # Word-level bullet comparison
│   │   │
│   │   ├── career/
│   │   │   ├── cover_letter.py          # Cover letter + cold email generation
│   │   │   ├── interview_prep.py        # Interview question generation
│   │   │   ├── linkedin_optimizer.py    # LinkedIn profile optimization
│   │   │   └── audit_matrix.py          # Structured resume audit
│   │   │
│   │   ├── bulk/
│   │   │   └── bulk_analyzer.py         # Multi-resume batch screening
│   │   │
│   │   └── tracking/
│   │       └── tracker.py               # Application pipeline (SQLite)
│   │
│   ├── dashboard.py                     # Streamlit UI (fragment-scoped, cached)
│   └── main.py                          # FastAPI app assembly
│
├── data/
│   ├── applications.db                  # Application tracker (SQLite)
│   ├── llm_processing.jsonl             # Pipeline event log (JSONL)
│   └── skills.json                      # Local skill taxonomy
│
├── images/
│   └── arch.png
│
├── tests/
│   ├── check_endpoints.py
│   ├── test_analyzer.py
│   ├── test_api.py
│   ├── test_gateway.py
│   ├── test_hf_hub
│   └── test_parser.py
│
├── .env                                 # Local secrets (gitignored)
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.py                               # Starts FastAPI + Streamlit together
```

### Module responsibilities

| Subpackage | Owns |
| ---------- | ---- |
| `services/parsing/` | Converting raw files into normalized text |
| `services/analysis/` | Deterministic scoring (ATS, keywords, skill gaps) |
| `services/llm/` | Provider routing, quota tracking, token accounting |
| `services/cv/` | CV generation — Markdown, LaTeX, PDF, diffing |
| `services/career/` | Post-application outreach and preparation tools |
| `services/bulk/` | Batch operations over multiple resumes |
| `services/tracking/` | Persistent application state |
| `core/` | Cross-cutting infrastructure (config, logging) |

`__pycache__` and `.pytest_cache` are runtime artifacts and are not part of the source layout.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip
- Git
- LaTeX with `pdflatex` (only required for PDF generation)

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-extra
```

**macOS**:
```bash
brew install --cask mactex-no-gui
```

**Windows**: install MiKTeX or TinyTeX and ensure `pdflatex` is on `PATH`.

### Installation

```bash
git clone https://github.com/Baqir110/ai-resume-analyzer.git
cd ai-resume-analyzer

python -m venv venv
```

Activate:
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Install:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Configuration

Copy the example and fill in your keys:

```bash
# Windows
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Minimal `.env`:

```env
# LLM gateway (default route)
OPENAI_BASE_URL=https://api.experientiallabs.ai/v1
EXPERIENTIAL_ORG_KEY=your_experiential_org_key

# FastAPI URL (used by the dashboard)
DEFAULT_API_BASE=http://localhost:8000

# Optional native provider keys
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Default models per provider
GEMINI_MODEL=gemini-2.5-flash
GROQ_MODEL=qwen3.8-27b
OPENROUTER_MODEL=deepseek-v4-flash
DEEPSEEK_MODEL=deepseek-v4-flash
OPENAI_MODEL=gpt-5.6-luna
CLAUDE_MODEL=claude-fable-5
OLLAMA_MODEL=qwen3.8-27b

# Paths
LLM_PROCESSING_LOG=data/llm_processing.jsonl

# Optional Ollama
OLLAMA_BASE_URL=http://localhost:11434/api/generate
```

**Never commit `.env` or API keys.**

### Running the Application

Single command (starts both services):

```bash
python run.py
```

Or manually:

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard
streamlit run app/dashboard.py
```

- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## API Reference

Primary namespace: `/api/v1/resume/`

### Core endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/analyze` | ATS analysis + skill gap for one resume |
| `POST` | `/generate-full` | Tailored DOCX resume |
| `POST` | `/generate-german-cv` | German Lebenslauf PDF (LaTeX) |
| `POST` | `/generate-tex-cv` | Raw LaTeX source |

### Career workflow

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/diff-preview` | Word-level bullet comparison |
| `POST` | `/analyze-bulk` | Multi-resume batch screening |
| `POST` | `/audit-matrix` | Structured resume audit |
| `POST` | `/generate-cover-letter` | Cover letter + cold email |
| `POST` | `/interview-prep` | Interview questions + gap defenses |
| `POST` | `/linkedin-optimize` | LinkedIn content generation |

### Tracking & observability

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/tracker/applications` | List tracked applications |
| `POST` | `/tracker/applications` | Create application record |
| `PATCH` | `/tracker/applications/{id}` | Update status |
| `DELETE` | `/tracker/applications/{id}` | Delete record |
| `GET` | `/usage-summary` | Aggregated token usage |
| `GET` | `/model-catalog` | Available models with pricing metadata |
| `GET` | `/quota-status` | Per-provider rate-limit snapshot |
| `GET` | `/quota-events` | Recent rate-limit hits |
| `GET` | `/processing-log` | Recent pipeline events |
| `DELETE` | `/processing-log` | Clear the log |
| `GET` | `/backend-status` | Provider health + log file path |

Full schemas available at `/docs`.

---

## LLM Providers

The provider abstraction lives in `app/services/llm/provider.py`.

| Provider           | Integration      | Fallback Chain Position |
| ------------------ | ---------------- | ----------------------- |
| Experiential Labs  | Gateway          | 0 (default)             |
| Google GenAI       | Native API       | 1                       |
| OpenAI             | Native API       | 2                       |
| DeepSeek           | Native API       | 3                       |
| Groq               | Native API       | 4                       |
| OpenRouter         | Native API       | 5                       |
| Anthropic Claude   | Native / Gateway | 6                       |
| Ollama             | Local            | 7 (last resort)         |

If a provider fails, the next in the chain is tried automatically. The `experiential` synthetic provider always routes through the gateway and keeps the full fallback chain regardless of route mode.

---

## Observability

### Pipeline event log

All stages emit structured events to `data/llm_processing.jsonl`. Events share a `request_id` so a full pipeline run can be reconstructed.

```json
{"timestamp": "...", "kind": "pipeline", "event": "pipeline_started",   "request_id": "...", "operation": "analyze"}
{"timestamp": "...", "kind": "parse",    "event": "parse_completed",    "request_id": "...", "file_type": "pdf", "chars_extracted": 4312}
{"timestamp": "...", "kind": "analysis", "event": "analysis_completed", "request_id": "...", "ats_score": 72, "missing_count": 5}
{"timestamp": "...", "kind": "llm",      "event": "request_completed",  "request_id": "...", "provider": "experiential", "total_tokens": 1415, "estimated_cost_usd": 0.0}
{"timestamp": "...", "kind": "pdf",      "event": "pdf_compiled",       "request_id": "...", "pages": 1, "bytes": 84210}
{"timestamp": "...", "kind": "pipeline", "event": "pipeline_completed", "request_id": "...", "status": "success", "duration_ms": 16400}
```

Event kinds: `llm`, `parse`, `analysis`, `translate`, `latex`, `pdf`, `docx`, `pipeline`, `quota`, `cache`, `response`.

The dashboard surfaces this as a filterable table with a `Kind` column.

### Quota tracking

Per-provider RPM/RPD/TPM limits are declared in `.env`:

```env
GEMINI_QUOTA_RPM=15
GEMINI_QUOTA_RPD=1500
GEMINI_QUOTA_TPM=1000000
```

Usage is computed from the local event log. The dashboard shows percentage bars per provider per window (minute / 24h) with reset times.

> **Note:** Gemini does not expose a "remaining quota" endpoint for API-key access. Numbers reflect calls made from this app only, compared against declared limits. For account-wide readings, use the AI Studio dashboard.

---

## Document Generation

### DOCX
Generated from Markdown via `python-docx`. Single-column, standard headings, no tables — parser-friendly.

### LaTeX → PDF
Templates in `app/services/cv/latex_generator.py`, compiled locally with `pdflatex`. Available layouts:

| Layout | Purpose |
| ------ | ------- |
| `international_ats` | Compact single-page English CV |
| `standard` | Generic English ATS |
| `hr_executive_gold` | Executive English |
| `german_corporate` | Corporate German |
| `german_classic` | Traditional German |
| `german_modern` | Modern German |
| `german_minimal_ats` | Strict-invariant German single-column |

Preamble patches (`lmodern` font, `\sloppy`, `\emergencystretch`) are applied at runtime to prevent overflow and font-fallback artifacts.

---

## Application Tracking

`app/services/tracking/tracker.py` implements a local SQLite-backed application tracker at `data/applications.db`. The schema covers company, role, URL, ATS score, status, timestamps, and freeform notes. Operations are standard CRUD.

---

## Testing

```bash
python -m pytest -v                      # full suite
python -m pytest tests/test_analyzer.py  # analyzer only
python -m pytest tests/test_api.py       # API tests
python -m pytest tests/test_gateway.py   # provider routing
python -m pytest tests/test_parser.py    # resume parsing

python tests/check_endpoints.py          # ad-hoc endpoint smoke test
```

---

## Design Decisions

### Additive optimization
The pipeline enriches rather than rewrites. Factual data — employers, dates, degrees — is treated as immutable input.

### Hybrid ATS analysis
TF-IDF + cosine similarity give a deterministic baseline; LLM processing adds contextual enrichment. Neither replaces the other.

### Provider abstraction
All LLM access flows through `LLMService`. Switching providers is a config change, not a code change.

### Pre-flight language normalization
Cross-language CV generation translates the *input resume*, not the *generated output*. LaTeX never goes through a translation pass, so formatting is preserved.

### Automatic layout selection
Layout is a function of the job description's language. The user can override, but the default is deterministic.

### Local PDF compilation
`pdflatex` runs on the host. No third-party PDF API, no data egress.

### Stateless resume processing
Logs contain request metadata, not resume contents. The uploaded file exists only in memory during the request.

### Fragment-scoped dashboard
Expensive UI sections are wrapped in `@st.fragment` so a widget interaction re-executes only the affected section. Sidebar fetchers use `@st.cache_data` with short TTLs.

### Domain-organized service layer
`app/services/` is split into subpackages by domain (`parsing/`, `analysis/`, `llm/`, `cv/`, `career/`, `bulk/`, `tracking/`). Shared infrastructure lives in `app/core/`. This keeps imports self-documenting: `from app.services.cv.optimizer import ...` states intent.

---

## Roadmap

### Done
- [x] Multi-provider LLM router with fallback chain
- [x] Native integrations: Gemini, Groq, OpenRouter, DeepSeek, OpenAI, Anthropic
- [x] Experiential Labs gateway support
- [x] Ollama local execution
- [x] Streamlit dashboard with fragment scoping + cached fetchers
- [x] FastAPI backend with pipeline logging decorator
- [x] Resume parsing (PDF, DOCX, TXT)
- [x] Hybrid ATS scoring (TF-IDF + LLM)
- [x] Skill-gap and keyword-density analysis
- [x] Additive CV optimization
- [x] DOCX generation
- [x] LaTeX/PDF generation with 7 templates
- [x] Automatic layout selection from JD language
- [x] Pre-flight resume translation for cross-language generation
- [x] Factual invariant validation (German Minimal ATS)
- [x] Keyword-dump stripping
- [x] Per-provider quota and token tracking
- [x] Structured pipeline event logging
- [x] Word-level diff preview
- [x] Bulk resume analysis
- [x] Resume audit, cover letter, interview prep, LinkedIn services
- [x] Application tracker (SQLite)
- [x] Domain-organized service layer with subpackages

### Planned
- [ ] Streaming LLM responses in the dashboard
- [ ] Cover-letter and cold-email template library
- [ ] Role-specific interview question bank
- [ ] LinkedIn "About" section A/B comparison
- [ ] Analytics dashboard over the application tracker
- [ ] Additional CV templates (Academic, Technical Lead)
- [ ] Optional Postgres backend for the tracker
- [ ] Docker compose for one-command deploy
- [ ] CI: GitHub Actions running pytest on push

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/improvement`
3. Make changes
4. Run tests: `python -m pytest -v`
5. Commit: `git commit -m "Add feature"`
6. Push and open a PR

When adding a new service module, place it in the subpackage that matches its domain:

- New LLM provider → `services/llm/`
- New CV template or export format → `services/cv/`
- New resume analysis signal → `services/analysis/`
- New career/outreach tool → `services/career/`

---
