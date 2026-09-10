# AI Resume & CV Optimization Hub

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-compose-blue.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

AI Resume analysis, optimisation and career application based on FastAPI, Streamlit and various LLM providers.

The platform also uses AI to compare resumes with job descriptions, flags missing skills and keywords, creates job-specific CVs (DOCX, LaTeX/PDF), allows for bulk CV analysis, tracks the usage of LLMs and provider quotas, and offers further career workflow features like cover letters, interview preparation, LinkedIn optimization, application tracking, and resume audit analysis.
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
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [LLM Providers](#llm-providers)
- [Observability](#observability)
- [Document Generation](#document-generation)
- [Career Workflow](#career-workflow)
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
- German Lebenslauf PDF generation through LaTeX (10 templates)
- Cover-letter template library and interview-question families
- Per-provider quota and token usage tracking
- Structured pipeline event logging
- Analytics dashboard over the pipeline log and tracker DB

The optimization workflow follows an additive approach: existing career history, employers, dates, education, projects, and other factual information are preserved while relevant job-specific terminology and improvements are incorporated.

---

## What's New

- **Automatic layout selection.** The backend detects the job description's language and picks the appropriate CV template without user intervention.
- **Pre-flight language normalization.** Cross-language generation translates the *input resume*, not the *generated output*. LaTeX never goes through a translation pass.
- **Cover-letter template library.** Four distinct prompts — Classic Professional, Modern Concise, Story-Driven, Value-First — that shape the letter's structure and tone.
- **Interview-question families.** Five focus areas — Technical, Behavioral, Product, Leadership, MLOps/DevOps — that shape the generated question set.
- **Quota and token tracking.** Per-provider RPM/RPD/TPM usage aggregated from the local event log with live progress bars in the dashboard.
- **Structured pipeline logging.** Every stage — parse, analysis, translation, LLM call, LaTeX, PDF compile, DOCX build — emits a JSONL event with a shared `request_id`.
- **Multi-provider fallback chain.** Provider failures retry against the next available provider automatically.
- **Fragment-scoped dashboard.** Expensive UI sections wrapped in `@st.fragment` so widget interactions only re-execute the affected section. Sidebar fetchers use `@st.cache_data` with short TTLs.
- **Factual invariant validation.** The German Minimal ATS layout verifies that company names, dates, and degree titles survive into the generated document before PDF compilation.
- **Keyword-dump stripping.** Post-processing removes LLM-injected "Additional Keywords" sections that break ATS parsers.
- **Domain-organized service layer.** `app/services/` split into `parsing/`, `analysis/`, `llm/`, `cv/`, `career/`, `bulk/`, `tracking/`.
- **Docker compose deployment.** Two-service stack (backend + dashboard) with a shared data volume and healthchecks.
- **GitHub Actions CI.** Runs pytest on push against Python 3.11 and 3.12.

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

### Career Workflow
Generate cover letters in four distinct tones, prepare for interviews with five question families, optimize LinkedIn content, and track applications — all from the same workspace.

---

## Key Features

### Multi-Format Resume Parsing
Supports PDF, DOCX, and TXT. Every parse emits a `parse_completed` or `parse_failed` event with timing and character count.

### Hybrid ATS Scoring
Combines TF-IDF + cosine similarity with LLM-assisted interpretation:
- Keyword extraction and density analysis
- Technical skill detection
- Matching-skill and missing-skill analysis
- Improvement suggestions

### Additive LLM Optimization
The optimizer enriches rather than rewrites. Company names, job titles, employment dates, degree information, projects, and existing career history are preserved.

### Multi-Provider LLM Architecture
Configurable providers: Google GenAI, Groq, OpenRouter, DeepSeek, OpenAI, Anthropic Claude, Experiential Labs Gateway, Ollama. Automatic fallback on failure.

### Automatic Layout Selection
The backend inspects the JD's function-word profile and selects the appropriate CV template. Users can override.

### Pre-flight Language Normalization
When the target layout requires a different language, the resume text is translated **before** generation. LaTeX is never touched by translation.

### Ten CV Templates
`international_ats`, `academic`, `technical_lead`, `hr_executive_gold`, `standard`, `german_corporate`, `german_classic`, `german_modern`, `german_minimal_ats`, plus `auto` detection.

### Cover-Letter Template Library
Four prompt-shaped templates:
| ID | Style |
|----|-------|
| `classic_professional` | Formal, three-paragraph, safe for enterprises |
| `modern_concise` | Short, direct, tech-friendly |
| `story_driven` | Narrative arc, opens with a specific moment |
| `value_first` | Metric-first, senior IC framing |

### Interview-Question Families
Five focus areas:
| ID | Focus |
|----|-------|
| `technical` | System design, coding, debugging |
| `behavioral` | STAR: conflict, ownership, failure |
| `product` | Product thinking, prioritization |
| `leadership` | Team growth, direction, incidents |
| `mlops_devops` | CI/CD, observability, on-call |

### Keyword-Dump Stripping
Two-stage post-processing removes LLM-injected keyword sections that break ATS parsers.

### Factual Invariant Validation
The German Minimal ATS layout extracts role/company/date triples and verifies they survive into the output. Generation is rejected with 422 if any invariant is missing.

### Analytics Dashboard
Seven charts: tokens/day, cost/day, applications by status, ATS score distribution, top missing skills, provider mix, recent errors.

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
             │ Provider Router │        └──────────────────┘
             └────────┬────────┘
                      │
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
| Containerization    | Docker + Docker Compose                                   |
| CI                  | GitHub Actions                                            |

---

## Project Structure

```text
ai-resume-analyzer/
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # pytest on push (Python 3.11 + 3.12)
│
├── app/
│   ├── api/
│   │   └── endpoints.py                 # FastAPI routes + pipeline decorator
│   ├── core/
│   │   ├── config.py
│   │   └── event_log.py                 # Shared JSONL pipeline event logger
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── parsing/
│   │   │   └── resume_parser.py
│   │   ├── analysis/
│   │   │   ├── ats_analyzer.py
│   │   │   └── suggestions.py
│   │   ├── llm/
│   │   │   ├── provider.py              # Multi-provider router + fallback
│   │   │   └── quota_tracker.py         # RPM/RPD/TPM usage aggregation
│   │   ├── cv/
│   │   │   ├── optimizer.py             # CV optimization + layout selection
│   │   │   ├── latex_generator.py       # 10 LaTeX templates + PDF compilation
│   │   │   └── diff_preview.py
│   │   ├── career/
│   │   │   ├── cover_letter.py          # Service
│   │   │   ├── cover_letter_templates.py # 4 prompt templates
│   │   │   ├── interview_prep.py        # Service
│   │   │   ├── interview_questions.py   # 5 question families
│   │   │   ├── linkedin_optimizer.py
│   │   │   └── audit_matrix.py
│   │   ├── analytics/
│   │   │   └── dashboard_aggregator.py  # Aggregates log + tracker DB
│   │   ├── bulk/
│   │   │   └── bulk_analyzer.py
│   │   └── tracking/
│   │       └── tracker.py               # SQLite-backed application tracker
│   ├── dashboard.py                     # Streamlit UI (5 fragments)
│   └── main.py
│
├── data/
│   ├── applications.db                  # Application tracker (gitignored)
│   ├── llm_processing.jsonl             # Pipeline event log (gitignored)
│   └── skills.json
│
├── tests/
│   ├── check_endpoints.py
│   ├── test_analyzer.py
│   ├── test_api.py
│   ├── test_gateway.py
│   ├── test_german_minimal_ats.py
│   └── test_parser.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
└── run.py
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip
- Git
- LaTeX with `pdflatex` (only required for PDF generation)
- Docker Desktop (only required for the Docker workflow)

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

### Running Locally

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

## Running with Docker

Build the image:

```bash
docker compose build
```

Start both services:

```bash
docker compose up
```

- Dashboard: http://localhost:8501
- Backend docs: http://localhost:8000/docs

Stop:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f backend
docker compose logs -f dashboard
```

Rebuild after code changes:

```bash
docker compose up --build
```

**Note:** `docker compose up` will fail if the local `python run.py` is running, because both bind to ports 8000 and 8501. Stop the local instance first, or override the host port in `docker-compose.yml`.

---

## API Reference

Primary namespace: `/api/v1/resume/`

### Core

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
| `POST` | `/generate-cover-letter` | Cover letter (4 templates) + cold email |
| `POST` | `/interview-prep` | Interview questions (5 families) |
| `POST` | `/linkedin-optimize` | LinkedIn content generation |
| `GET`  | `/career-options` | Catalog of templates and families |

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
| `GET` | `/analytics/summary` | Aggregated analytics over log + tracker |
| `GET` | `/backend-status` | Provider health + log file path |

Full schemas available at `/docs`.

---

## LLM Providers

The provider abstraction lives in `app/services/llm/provider.py`.

| Provider           | Integration      | Fallback position |
| ------------------ | ---------------- | ----------------- |
| Experiential Labs  | Gateway          | 0 (default)       |
| Google GenAI       | Native API       | 1                 |
| OpenAI             | Native API       | 2                 |
| DeepSeek           | Native API       | 3                 |
| Groq               | Native API       | 4                 |
| OpenRouter         | Native API       | 5                 |
| Anthropic Claude   | Native / Gateway | 6                 |
| Ollama             | Local            | 7                 |

If a provider fails, the next in the chain is tried automatically.

---

## Observability

### Pipeline event log

All stages emit structured events to `data/llm_processing.jsonl`:

```json
{"timestamp": "...", "kind": "pipeline", "event": "pipeline_started",   "request_id": "...", "operation": "analyze"}
{"timestamp": "...", "kind": "parse",    "event": "parse_completed",    "request_id": "...", "file_type": "pdf", "chars_extracted": 4312}
{"timestamp": "...", "kind": "analysis", "event": "analysis_completed", "request_id": "...", "ats_score": 72, "missing_count": 5}
{"timestamp": "...", "kind": "llm",      "event": "request_completed",  "request_id": "...", "provider": "experiential", "total_tokens": 1415, "estimated_cost_usd": 0.0}
{"timestamp": "...", "kind": "pdf",      "event": "pdf_compiled",       "request_id": "...", "pages": 1, "bytes": 84210}
{"timestamp": "...", "kind": "pipeline", "event": "pipeline_completed", "request_id": "...", "status": "success", "duration_ms": 16400}
```

Event kinds: `llm`, `parse`, `analysis`, `translate`, `latex`, `pdf`, `docx`, `pipeline`, `quota`, `cache`, `response`.

### Quota tracking

Per-provider RPM/RPD/TPM limits can be declared in `.env`:

```env
GEMINI_QUOTA_RPM=15
GEMINI_QUOTA_RPD=1500
GEMINI_QUOTA_TPM=1000000
```

Usage is computed from the local event log. The dashboard shows percentage bars per provider with reset times.

> **Note:** Gemini does not expose a "remaining quota" endpoint for API-key access. Numbers reflect calls made from this app only, compared against declared limits. For account-wide readings, use the AI Studio dashboard.

### Analytics dashboard

`GET /analytics/summary?period=30d` returns aggregates over the log and tracker DB. The dashboard renders seven charts: tokens/day, cost/day, applications by status, ATS score distribution, top missing skills, provider mix, recent errors.

---

## Document Generation

### DOCX
Generated from Markdown via `python-docx`. Single-column, standard headings, no tables.

### LaTeX → PDF
Compiled locally with `pdflatex`. Available layouts:

| Layout | Purpose |
| ------ | ------- |
| `international_ats` | Compact single-page English CV |
| `academic` | Serif, Education-first, Research/Publications sections |
| `technical_lead` | Modern, Technical Summary + Open Source + Speaking sections |
| `standard` | Generic English ATS |
| `hr_executive_gold` | Executive English |
| `german_corporate` | Corporate German |
| `german_classic` | Traditional German |
| `german_modern` | Modern German |
| `german_minimal_ats` | Strict-invariant German single-column |

Preamble patches (`lmodern` font, `\sloppy`, `\emergencystretch`) applied at runtime to prevent overflow and font-fallback artifacts.

---

## Career Workflow

### Cover Letters
`CoverLetterService.generate_cover_letter_and_outreach()` accepts a `template` parameter (one of four IDs) and injects the matching prompt snippet. Response includes `template` and `template_label` in the metadata.

### Interview Prep
`InterviewPrepService.generate_interview_prep()` accepts a `family` parameter (one of five IDs) and injects the matching prompt snippet. Response includes `_meta.family` and `_meta.family_label`.

### LinkedIn Optimizer, Audit Matrix
Retained as-is. Each generates structured content for its domain.

---

## Application Tracking

`app/services/tracking/tracker.py` implements a SQLite-backed application tracker at `data/applications.db`. Schema covers company, role, URL, ATS score, status, timestamps, and freeform notes. Standard CRUD operations.

---

## Testing

```bash
python -m pytest -v                      # full suite — 17 tests
python -m pytest tests/test_analyzer.py
python -m pytest tests/test_api.py
python -m pytest tests/test_gateway.py
python -m pytest tests/test_parser.py
python -m pytest tests/test_german_minimal_ats.py

python tests/check_endpoints.py          # ad-hoc smoke test
```

CI runs the full suite on every push to `main`, `master`, or `develop` against Python 3.11 and 3.12.

---

## Design Decisions

### Additive optimization
The pipeline enriches rather than rewrites. Factual data — employers, dates, degrees — is treated as immutable.

### Hybrid ATS analysis
TF-IDF + cosine similarity gives a deterministic baseline; LLM processing adds contextual enrichment.

### Provider abstraction
All LLM access flows through `LLMService`. Switching providers is a config change, not a code change.

### Pre-flight language normalization
Cross-language CV generation translates the input resume, not the generated output. LaTeX is never touched by translation.

### Automatic layout selection
Layout is a function of the JD's language. The user can override, but the default is deterministic.

### Local PDF compilation
`pdflatex` runs on the host. No third-party PDF API.

### Stateless resume processing
Logs contain request metadata, not resume contents. Uploaded files exist only in memory during the request.

### Fragment-scoped dashboard
Expensive UI sections wrapped in `@st.fragment`. Sidebar fetchers use `@st.cache_data` with short TTLs.

### Domain-organized service layer
`app/services/` split by domain. Shared infrastructure in `app/core/`.

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
- [x] LaTeX/PDF generation with 10 templates
- [x] Automatic layout selection from JD language
- [x] Pre-flight resume translation for cross-language generation
- [x] Factual invariant validation (German Minimal ATS)
- [x] Keyword-dump stripping
- [x] Cover-letter template library (4 templates, wired)
- [x] Interview question families (5 families, wired)
- [x] Per-provider quota and token tracking
- [x] Structured pipeline event logging
- [x] Analytics dashboard over the pipeline log and tracker DB
- [x] Word-level diff preview
- [x] Bulk resume analysis
- [x] Resume audit, cover letter, interview prep, LinkedIn services
- [x] Application tracker (SQLite)
- [x] Domain-organized service layer with subpackages
- [x] Docker Compose deployment
- [x] GitHub Actions CI

### Deferred

- [ ] Streaming LLM responses in the dashboard (Streamlit's execution model makes this expensive; low benefit at single-user scale)
- [ ] LinkedIn "About" section A/B comparison (niche)
- [ ] Optional Postgres backend for the tracker (SQLite is sufficient for single-user; Postgres matters only at multi-user scale)

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
