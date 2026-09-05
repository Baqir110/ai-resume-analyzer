# AI Resume & CV Optimization Hub

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

An AI-powered resume analysis, optimization, and career application platform built with FastAPI, Streamlit, Python NLP/ML tooling, and multiple LLM providers.

The platform analyzes resumes against job descriptions, identifies missing skills and keywords, generates job-tailored CVs, provides document exports, supports bulk resume analysis, and includes additional career workflow tools such as cover letters, interview preparation, LinkedIn optimization, application tracking, and resume audit analysis.

---

## Table of Contents

* [Overview](#overview)
* [Use Cases](#use-cases)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Environment Configuration](#environment-configuration)
  * [Running the Application](#running-the-application)
* [API Reference](#api-reference)
* [Configuration](#configuration)
* [LLM Providers](#llm-providers)
* [Document Generation](#document-generation)
* [Application Tracking](#application-tracking)
* [Testing](#testing)
* [Design Decisions](#design-decisions)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

AI Resume & CV Optimization Hub is a resume intelligence and career application platform designed to evaluate candidate resumes against specific job descriptions and produce targeted improvements.

The system combines:

* TF-IDF and cosine similarity
* Dynamic keyword and skill extraction
* Skill-gap analysis
* LLM-powered resume optimization
* Multi-provider AI routing
* Bulk candidate analysis
* Word-level document diffing
* ATS-friendly DOCX generation
* German Lebenslauf PDF generation through LaTeX
* Resume audit analysis
* Cover-letter generation
* Interview preparation
* LinkedIn profile optimization
* Job application tracking

The optimization workflow follows an additive approach: existing career history, employers, dates, education, projects, and other factual information should be preserved while relevant job-specific terminology and improvements are incorporated.

---

## Use Cases

### Job Application Tailoring

Analyze a resume against a specific job description and identify:

* ATS compatibility
* Matching skills
* Missing skills
* Missing keywords
* Improvement opportunities

### Resume Optimization

Generate a job-tailored version of an existing resume while preserving the candidate's original career history and factual information.

### Bulk Candidate Screening

Analyze multiple resumes against a single job description and rank candidates according to ATS compatibility and skill coverage.

### Resume Audit

Perform structured analysis of resume quality, content coverage, and job alignment.

### LinkedIn Optimization

Use AI-assisted analysis and generation to improve LinkedIn-oriented professional content.

### Cover Letters

Generate job-specific cover letters based on candidate information and target job requirements.

### Interview Preparation

Generate role-specific interview preparation material based on the target position and identified skill gaps.

### Application Tracking

Maintain application-related information in the application's local database.

---

## Key Features

### Multi-Format Resume Parsing

The parser supports common resume formats including:

* PDF
* DOCX
* TXT

Resume content is extracted and normalized before being passed to the analysis pipeline.

### Hybrid ATS Scoring

The analyzer combines traditional NLP/ML techniques with AI-assisted processing.

Core analysis includes:

* TF-IDF vectorization
* Cosine similarity
* Keyword extraction
* Technical skill detection
* Matching-skill analysis
* Missing-skill analysis
* Keyword density
* Improvement suggestions

### Additive LLM Optimization

The optimization engine is designed to enrich existing resume content instead of unnecessarily rewriting it.

Important information such as:

* Company names
* Job titles
* Employment dates
* Degree information
* Projects
* Existing career history

is intended to remain intact while relevant job-description terminology is incorporated.

### Multi-Provider LLM Architecture

The application supports configurable AI providers, including:

* Google GenAI
* Groq
* OpenRouter
* DeepSeek
* OpenAI
* Anthropic Claude
* Experiential Labs Gateway
* Ollama

The LLM service provides provider abstraction and can use native provider integrations when configured, with gateway-based execution where supported.

### Visual Resume Diff

The diff-preview service compares original and optimized text at word level.

It can identify:

* Added text
* Removed text
* Changed text
* Similarity between versions

This allows users to inspect how AI optimization changed their original content.

### DOCX Resume Generation

The system can generate ATS-friendly Word documents using a clean, single-column structure.

### German Lebenslauf PDF Generation

German CVs can be generated through LaTeX and compiled locally with `pdflatex`.

Available layout styles include:

* Corporate Slate Navy
* Professional ATS
* Classic Conservative
* Modern Executive
* HR Executive Gold

### Bulk Analysis

Multiple resumes can be processed against one job description for candidate comparison and ranking.

### Career Workflow Tools

The current project also includes dedicated services for:

* Resume auditing
* Cover-letter generation
* Interview preparation
* LinkedIn optimization
* Application tracking

---

## System Architecture

![System Architecture](./images/arch.png)

The application is divided into several logical layers:

```text
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard   │
                    │      User Interface     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      FastAPI API        │
                    │   REST Endpoint Layer   │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌───────────┐      ┌────────────┐     ┌─────────────┐
        │  Parser   │      │  Analyzer  │     │   Tracker   │
        └─────┬─────┘      └─────┬──────┘     └─────────────┘
              │                  │
              │                  ▼
              │           ┌───────────────┐
              │           │ Skill / ATS   │
              │           │   Analysis    │
              │           └───────┬───────┘
              │                   │
              └──────────┬────────┘
                         ▼
                ┌─────────────────┐
                │   LLM Service   │
                │ Provider Router │
                └────────┬────────┘
                         │
          ┌──────────────┼─────────────────────┐
          │              │                     │
          ▼              ▼                     ▼
      Native APIs    Gateway APIs           Ollama
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
```

---

## Technology Stack

| Category            | Technology                                                          |
| ------------------- | ------------------------------------------------------------------- |
| Language            | Python 3.9+                                                         |
| API Framework       | FastAPI                                                             |
| Frontend            | Streamlit                                                           |
| NLP / ML            | Scikit-Learn, spaCy                                                 |
| PDF Parsing         | PyPDF                                                               |
| DOCX Parsing        | python-docx                                                         |
| Document Generation | python-docx                                                         |
| PDF Compilation     | pdflatex                                                            |
| Configuration       | `.env` environment variables                                        |
| Database            | SQLite                                                              |
| AI                  | Google GenAI, Groq, OpenRouter, DeepSeek, OpenAI, Anthropic, Ollama |
| Testing             | pytest                                                              |
| API Server          | Uvicorn                                                             |

---

## Project Structure

The current repository structure is:

```text
ai-resume-analyzer/
│
├── app/
│   │
│   ├── api/
│   │   └── endpoints.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   ├── services/
│   │   ├── analyzer.py
│   │   ├── audit_matrix.py
│   │   ├── bulk_analyzer.py
│   │   ├── cover_letter.py
│   │   ├── diff_preview.py
│   │   ├── interview_prep.py
│   │   ├── latex_generator.py
│   │   ├── linkedin_optimizer.py
│   │   ├── llm_provider.py
│   │   ├── optimizer.py
│   │   ├── parser.py
│   │   ├── suggestions.py
│   │   └── tracker.py
│   │
│   ├── dashboard.py
│   └── main.py
│
├── data/
│   ├── applications.db
│   ├── llm_processing.jsonl
│   └── skills.json
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
├── .env
├── .env.example
├── .gitignore
├── main.py
├── python3
├── README.md
├── requirements.txt
└── run.py
```

`__pycache__` and `.pytest_cache` directories are generated runtime/test artifacts and are not part of the application's source architecture.

---

## Getting Started

### Prerequisites

Install:

* Python 3.9 or newer
* pip
* Git
* LaTeX with `pdflatex` if PDF generation is required

For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-extra
```

For macOS:

```bash
brew install --cask mactex-no-gui
```

For Windows, install either MiKTeX or TinyTeX and ensure `pdflatex` is available in the system PATH.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Baqir110/ai-resume-analyzer.git
cd ai-resume-analyzer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a local environment file from the example:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Configure the required provider keys in `.env`.

Example:

```env
# Gateway
OPENAI_BASE_URL=https://api.experientiallabs.ai/v1
EXPERIENTIAL_ORG_KEY=your_experiential_org_key

# Local API
DEFAULT_API_BASE=http://localhost:8000

# Providers
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HF_TOKEN=your_huggingface_token

# Models
GEMINI_MODEL=gemini-2.5-flash
GROQ_MODEL=qwen3.8-27b
OPENROUTER_MODEL=deepseek-v4-flash
DEEPSEEK_MODEL=deepseek-v4-flash
OPENAI_MODEL=gpt-5.6-luna
CLAUDE_MODEL=claude-fable-5
OLLAMA_MODEL=qwen3.8-27b

# Paths
LLM_PROCESSING_LOG=data/llm_processing.jsonl

# Ollama
OLLAMA_BASE_URL=http://localhost:11434/api/generate
```

Do not commit `.env` or API keys to Git.

---

## Running the Application

### Start FastAPI

From the project root:

```bash
python run.py
```

Alternatively:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API should then be available at:

```text
http://localhost:8000
```

Swagger/OpenAPI documentation:

```text
http://localhost:8000/docs
```

### Start Streamlit

Open a second terminal with the virtual environment activated:

```bash
streamlit run app/dashboard.py
```

The dashboard is normally available at:

```text
http://localhost:8501
```

---

## API Reference

The primary API is exposed under:

```text
/api/v1/resume/
```

### Analyze Resume

```http
POST /api/v1/resume/analyze
```

Analyzes an uploaded resume against a target job description.

Typical request fields include:

* `resume_file`
* `job_description`
* `provider`

The analysis response can contain information such as:

* ATS match score
* Keyword density score
* Matching skills
* Missing skills
* Improvement suggestions

### Generate Full Resume

```http
POST /api/v1/resume/generate-full
```

Generates a job-tailored DOCX resume.

Request:

* Resume file
* Job description
* LLM provider

Response:

```text
application/vnd.openxmlformats-officedocument.wordprocessingml.document
```

### Generate German CV

```http
POST /api/v1/resume/generate-german-cv
```

Generates a German `Lebenslauf` PDF.

Request includes:

* Resume file
* Job description
* Layout style
* Provider

Response:

```text
application/pdf
```

### Generate LaTeX CV

```http
POST /api/v1/resume/generate-tex-cv
```

Returns the generated LaTeX source.

Request includes:

* Resume file
* Job description
* Layout style
* Provider

Response:

```text
text/plain
```

### Additional Workflow Endpoints

The application also contains services supporting:

* Bulk resume analysis
* Resume diff previews
* Resume auditing
* Cover letters
* Interview preparation
* LinkedIn optimization
* Application tracking

The exact routes and request schemas are exposed through the FastAPI Swagger documentation at `/docs`.

---

## LLM Providers

The provider abstraction is implemented in:

```text
app/services/llm_provider.py
```

Supported integrations include:

| Provider          | Integration   |
| ----------------- | ------------- |
| Google GenAI      | Native API    |
| Groq              | Native API    |
| OpenRouter        | API           |
| DeepSeek          | API           |
| OpenAI            | API           |
| Anthropic Claude  | API / Gateway |
| Experiential Labs | Gateway       |
| Ollama            | Local API     |

The architecture allows the application to select providers without coupling the rest of the application to a single LLM vendor.

Native provider execution can be used when credentials are available, while gateway routing can provide an alternative execution path where configured.

---

## Document Generation

### DOCX

The application generates ATS-oriented Word resumes using a clean, single-column format.

The generated document is intended to remain easy for both humans and Applicant Tracking Systems to parse.

### LaTeX PDF

German CV generation uses LaTeX templates and local `pdflatex` compilation.

Compilation occurs locally rather than requiring an external PDF-generation service.

This provides:

* Consistent layout
* Vector-based output
* Local compilation
* No external PDF-generation API requirement

---

## Application Tracking

Application tracking functionality is implemented through:

```text
app/services/tracker.py
```

The project currently contains a local SQLite database:

```text
data/applications.db
```

This database is used for application-related workflow data.

The application-tracking functionality can be extended independently from the resume analysis pipeline.

---

## Data and Logging

### Skills Taxonomy

```text
data/skills.json
```

Provides the local skill/keyword taxonomy used as a fallback during skill analysis.

### LLM Processing Log

```text
data/llm_processing.jsonl
```

Stores backend processing information in JSONL format.

### Application Database

```text
data/applications.db
```

Stores application-tracking data.

User-generated documents should not be committed to the repository.

---

## Testing

The project uses pytest.

Run the complete test suite:

```bash
python -m pytest -v
```

Run a specific test file:

```bash
python -m pytest tests/test_analyzer.py -v
```

API tests:

```bash
python -m pytest tests/test_api.py -v
```

Gateway tests:

```bash
python -m pytest tests/test_gateway.py -v
```

Parser tests:

```bash
python -m pytest tests/test_parser.py -v
```

Endpoint checks can also be performed with:

```bash
python tests/check_endpoints.py
```

---

## Design Decisions

### 1. Additive Optimization

The optimization pipeline is designed around enrichment rather than indiscriminate rewriting.

The objective is to improve job relevance while preserving factual candidate information.

### 2. Hybrid ATS Analysis

Traditional NLP/ML scoring is combined with LLM-assisted interpretation.

TF-IDF and cosine similarity provide a deterministic textual similarity layer, while LLM processing handles contextual enrichment and generation.

### 3. Provider Abstraction

LLM access is isolated behind a provider service.

This makes it possible to switch between providers without changing the higher-level resume-processing workflow.

### 4. Local PDF Compilation

LaTeX documents are compiled locally with `pdflatex`.

This avoids dependency on a third-party PDF-generation service.

### 5. Stateless Resume Processing

Resume processing is designed around input streams and generated outputs.

Processing logs contain request metadata rather than requiring permanent storage of the candidate's original resume.

### 6. Modular Career Workflow

Resume analysis, optimization, document generation, cover letters, interview preparation, LinkedIn optimization, auditing, and application tracking are separated into dedicated services.

This allows individual components to evolve without tightly coupling the entire application.

---

## Roadmap

### Completed

* [x] Multi-provider LLM backend
* [x] Google GenAI integration
* [x] Groq integration
* [x] OpenRouter integration
* [x] DeepSeek integration
* [x] OpenAI integration
* [x] Anthropic Claude support
* [x] Experiential Labs Gateway support
* [x] Ollama support
* [x] Native provider execution
* [x] Gateway fallback architecture
* [x] Streamlit dashboard
* [x] FastAPI backend
* [x] Resume parsing
* [x] TF-IDF ATS scoring
* [x] Keyword extraction
* [x] Skill-gap analysis
* [x] Additive CV optimization
* [x] DOCX generation
* [x] German Lebenslauf PDF generation
* [x] LaTeX template rendering
* [x] Visual diff preview
* [x] Bulk resume analysis
* [x] Resume audit service
* [x] Cover-letter service
* [x] Interview preparation service
* [x] LinkedIn optimization service
* [x] Application tracking
* [x] Local SQLite application database

### Planned

* [ ] Advanced cover-letter and cold-email workflows
* [ ] More role-specific interview preparation
* [ ] Expanded LinkedIn optimization
* [ ] Advanced application analytics
* [ ] Additional CV templates
* [ ] Additional ATS scoring signals
* [ ] Improved candidate comparison dashboards
* [ ] Production deployment configuration

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/improvement
```

3. Make your changes.
4. Run the test suite:

```bash
python -m pytest -v
```

5. Commit your changes:

```bash
git commit -m "Add new feature"
```

6. Push the branch:

```bash
git push origin feature/improvement
```

7. Open a Pull Request.

---

## License

Distributed under the MIT License.

See the project's `LICENSE` file for the complete license text.
