# AI Resume & CV Optimization Hub

[![Python 3.11â€“3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-compose-blue.svg)](https://docs.docker.com/compose/)

<!-- BEGIN:AUTO:STATS -->

| Metric | Value |
|--------|-------|
| API endpoints | 50 |
| Python files | 110 |
| Lines of code | 44,929 |
| Tests | 1332 |

<!-- END:AUTO:STATS -->

Compare a CV against a job posting, find out what is missing, and produce a
targeted CV as DOCX, LaTeX or PDF. There is also bulk screening, cover letters,
interview prep, a LinkedIn rewrite, and an application tracker.

The thing this project is careful about is not inventing experience. Tailoring
adds the posting's vocabulary to a CV that already supports it. Employers, job
titles, dates and degrees are carried through untouched, and the
[German Minimal ATS](CONTEXT.md) layout refuses to compile a document that
quietly dropped or altered any of them.

Runs entirely on your own machine. Ollama works with no account and no API key,
and `LLM_MODE=ollama` guarantees a CV never leaves the host.

- [What it does](#what-it-does)
- [Getting started](#getting-started)
- [Docker](#docker)
- [API reference](#api-reference)
- [LLM Providers](#llm-providers)
  - [Free / Local LLM Options](#free--local-llm-options)
  - [Switching models without a code change](#switching-models-without-a-code-change)
  - [Routing](#routing)
  - [Output ceilings](#output-ceilings)
  - [Model discovery](#model-discovery)
  - [Smoke testing](#smoke-testing)
  - [Troubleshooting](#troubleshooting)
- [What gets generated](#what-gets-generated)
- [Observability](#observability)
- [Testing](#testing)
- [Project structure](#project-structure)
- [How the scoring works](#how-the-scoring-works)
- [Security](#security)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)

---

## What it does

Resume parsing (PDF, DOCX, TXT) and a five-category ATS score that you can take
apart. Each category reports the points it won and lost, so the headline number
is a sum rather than an assertion.

Missing skills get reported as gaps. They are not quietly written into the CV.

Layout is recommended from the posting, with the reasoning shown. It stays a
recommendation; the choice is yours and nothing is locked.

Cross-language generation translates the *input* resume before generation, so
LaTeX never passes through a translation step.

Ten LaTeX layouts, four cover-letter tones, five interview-question families.

Optional job discovery and browser-driven applications, with auto-submit off.

---

## Getting started

### Prerequisites

- Python 3.11 or 3.12. Both are what CI runs and 3.12 is what the Docker image
  uses. 3.13 works too and the suite passes, but `python-jobspy` has no
  3.13-compatible release, so the JobSpy discovery source is marked
  `python_version < "3.13"` and is the one feature that drops out.
- A LaTeX installation, for the PDF output. Most of the application needs this.
- Docker Desktop, only for the Docker route.

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended lmodern

# macOS
brew install --cask mactex-no-gui
```

On Windows, install MiKTeX or TinyTeX and make sure `pdflatex` is on `PATH`.
Nothing here hard-codes a TeX path, so either works.

```bash
pdflatex --version    # check before you generate anything
```

### Install

```bash
git clone https://github.com/Baqir110/ai-resume-analyzer.git
cd ai-resume-analyzer
python -m venv .venv
```

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

The `-c constraints.txt` is not optional decoration. The numpy, regex and
scikit-learn pins below are mutually unsatisfiable without it, and CI and the
Dockerfile use this exact command so all three resolve the same way:

```bash
python -m pip install --upgrade pip
pip install -c constraints.txt -r requirements.txt
```

Contributors want the linters too:

```bash
pip install -c constraints.txt -r requirements-dev.txt
```

### Configure

```bash
# Windows
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

`.env.example` is the full commented reference: every provider with its base URL,
credential, model and output ceiling, plus the routing and retry switches.

The smallest useful `.env` keeps everything local:

```env
API_KEY=replace-with-a-long-random-secret
APPLICANT_PROFILE_PATH=data/applicant_profile.local.yaml
AUTOMATIC_APPLY=false
AUTOMATIC_SUBMIT=false

DEFAULT_API_BASE=http://127.0.0.1:8000
FASTAPI_API_BASE=http://127.0.0.1:8000

# Local, free, no account. "ollama" makes a failure an error instead of a
# silent send to a cloud provider.
LLM_MODE=ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen3:8b
OLLAMA_MAX_TOKENS=2048

LLM_RETRIES=0
LLM_PROCESSING_LOG=data/llm_processing.jsonl
```

Check the configuration before generating anything:

```bash
python -m scripts.llm_smoke
```

Never commit `.env`, API keys, browser profiles, or a filled-in applicant
profile. `data/applicant_profile.yaml` is a blank template; copy it to
`data/applicant_profile.local.yaml`, keep that one gitignored, and point
`APPLICANT_PROFILE_PATH` at it.

If credentials were ever committed in an earlier push, cleaning the working tree
is not enough. Rotate everything exposed and rewrite the history before
publishing again. See [Security](#security).

### Run

```bash
python run.py
```

That starts both services and waits for the backend to report healthy before it
opens the dashboard. Or run them yourself:

```bash
# Terminal 1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run app/dashboard/main.py
```

| | |
|---|---|
| Backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### One-Page CV Workflow

The dashboard is one workflow page with six stages, not six pages to click
through. A progress rail runs down the side and the sidebar says what to do next.

The stages, in order:

1. **Resume** — upload and parse
2. **Job description** — paste the posting
3. **ATS analysis** — score and gap report
4. **Improvement** — the actionable edits
5. **Generation** — DOCX, LaTeX and PDF
6. **Preview** — the rendered document

Everything else in the sidebar is deliberately not a stage. Overview is the
landing page, **layout** is a setting you choose once, and **diagnostics** is
where provider health and the pipeline log live.

Three words appear in these pages and they mean different things:

- **CONFIGURED** — a credential and a model are set. Says nothing about whether
  the provider works.
- **AVAILABLE** — the provider answered a listing request. Says nothing about
  whether it can generate.
- **PASSED** — a real request completed.

A provider is never presented as a success on the strength of the two weaker
states.

---

## Docker

```bash
docker compose build
docker compose up
```

Create `.env` with a strong `API_KEY` first. Compose refuses to start without
one.

- Dashboard on http://127.0.0.1:8501, docs on http://127.0.0.1:8000/docs
- Host ports bind to loopback only. Put an authenticated TLS proxy in front
  before exposing them.
- State lives in the `app_data` volume and is never baked into the image.
- The autonomous worker is opt-in: `docker compose --profile agent up agent`.

```bash
docker compose down             # stop
docker compose logs -f backend  # logs
docker compose up --build       # after code changes
```

`docker compose up` fails while a local `python run.py` is running, because both
want ports 8000 and 8501. Stop one or change the host port in
`docker-compose.yml`.

---

## API reference

Everything lives under `/api/v1/resume/`, except the job routes, which are
mounted at `/api/v1`.

<!-- BEGIN:AUTO:API -->

**Total endpoints: 51**

### Jobs, jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/jobs/answer-question` | Answer Question |
| `GET` | `/api/v1/jobs/applications` | List Applications |
| `POST` | `/api/v1/jobs/apply` | Apply To Job |
| `POST` | `/api/v1/jobs/apply-with-tailored-cv` | Apply With Tailored Cv |
| `POST` | `/api/v1/jobs/auto-discover-and-apply` | Auto Discover And Apply |
| `POST` | `/api/v1/jobs/fetch-jd` | Fetch Jd |

### New Features

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/resume/check-authenticity` | Check Authenticity |
| `POST` | `/api/v1/resume/feedback/comment` | Add Feedback Comment |
| `POST` | `/api/v1/resume/feedback/thread` | Create Feedback Thread |
| `GET` | `/api/v1/resume/feedback/{resume_id}` | Get Resume Feedback |
| `POST` | `/api/v1/resume/interview/evaluate` | Evaluate Interview Answer |
| `POST` | `/api/v1/resume/interview/question` | Generate Interview Question |
| `GET` | `/api/v1/resume/market-insights` | Market Insights |
| `POST` | `/api/v1/resume/score-breakdown` | Score Breakdown |
| `POST` | `/api/v1/resume/skill-roadmap` | Generate Skill Roadmap |
| `GET` | `/api/v1/resume/versions/compare` | Compare Versions |
| `POST` | `/api/v1/resume/versions/save` | Save Resume Version |
| `GET` | `/api/v1/resume/versions/{user_id}` | List Versions |

### Resume Analyzer

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/resume/analytics/summary` | Analytics Summary |
| `POST` | `/api/v1/resume/analyze` | Analyze Resume |
| `POST` | `/api/v1/resume/analyze-bulk` | Analyze Bulk |
| `POST` | `/api/v1/resume/ats-breakdown` | Ats Breakdown Endpoint |
| `POST` | `/api/v1/resume/audit-matrix` | Audit Matrix Endpoint |
| `GET` | `/api/v1/resume/backend-status` | Backend Status |
| `GET` | `/api/v1/resume/career-options` | Career Options |
| `POST` | `/api/v1/resume/diff-preview` | Diff Preview |
| `POST` | `/api/v1/resume/generate-cover-letter` | Generate Cover Letter Endpoint |
| `POST` | `/api/v1/resume/generate-cover-letter-pdf` | Generate Cover Letter Pdf Endpoint |
| `POST` | `/api/v1/resume/generate-full` | Generate Full Cv Endpoint |
| `POST` | `/api/v1/resume/generate-german-cv` | Generate German Cv Endpoint |
| `POST` | `/api/v1/resume/generate-tex-cv` | Generate Tex Cv Endpoint |
| `GET` | `/api/v1/resume/health` | Health Check |
| `POST` | `/api/v1/resume/improvement-loop` | Improvement Loop Endpoint |
| `POST` | `/api/v1/resume/interview-prep` | Interview Prep Endpoint |
| `POST` | `/api/v1/resume/layout-recommendation` | Layout Recommendation Endpoint |
| `POST` | `/api/v1/resume/linkedin-optimize` | Linkedin Optimize Endpoint |
| `GET` | `/api/v1/resume/model-catalog` | Model Catalog |
| `GET` | `/api/v1/resume/model-discovery` | Model Discovery |
| `GET` | `/api/v1/resume/pipeline-metrics` | Pipeline Metrics |
| `DELETE` | `/api/v1/resume/processing-log` | Clear Processing Log |
| `GET` | `/api/v1/resume/processing-log` | Processing Log |
| `GET` | `/api/v1/resume/quota-events` | Quota Events |
| `GET` | `/api/v1/resume/quota-status` | Quota Status |
| `GET` | `/api/v1/resume/tracker/applications` | List Applications Endpoint |
| `POST` | `/api/v1/resume/tracker/applications` | Create Application Endpoint |
| `DELETE` | `/api/v1/resume/tracker/applications/{app_id}` | Delete Application Endpoint |
| `PATCH` | `/api/v1/resume/tracker/applications/{app_id}` | Update Status Endpoint |
| `GET` | `/api/v1/resume/usage-summary` | Usage Summary |
| `POST` | `/api/v1/resume/validate-ats` | Validate Ats Endpoint |
| `POST` | `/api/v1/resume/validate-pdf` | Validate Pdf Endpoint |

### Streaming

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/resume/analyze-stream` | Analyze Resume Streaming |

<!-- END:AUTO:API -->

### Authentication

Every `/api/v1` route needs `X-API-Key` matching `API_KEY`. If the key is unset
the service fails closed rather than serving unauthenticated traffic.

### Request shapes

Generation endpoints take multipart form data, not JSON:

| Field | Required | Notes |
|---|---|---|
| `job_description` | yes | The posting, as plain text. |
| `resume_file` | yes | PDF, DOCX or TXT. |
| `layout_style` | no | A layout name, or `auto` to let the backend pick. |
| `template_style` | no | Overrides `layout_style` when both are sent. |
| `provider` | no | Pins a provider. Normally leave it to the configuration. |
| `model_name` | no | Pins a model id. |
| `route_mode` | no | `direct`, `automatic` or `experiential`. |
| `improvement_suggestions` | no | JSON array of strings from a prior audit. |

An unknown `layout_style` comes back as `400` with the list of layouts that do
exist, not as a 500.

---

## LLM Providers

Thirteen providers are registered in `app/services/llm/registry.py`, which is the
authoritative list. Each one gets its base URL, credential variable, model
variable and output ceiling from the registry, so adding a provider means adding
one entry there.

Only Anthropic and OpenAI are called through their vendor SDKs. Everything else,
Groq included, goes over plain HTTP against the provider's OpenAI-compatible
endpoint. One transport, one retry policy, one set of diagnostics.

### Free / Local LLM Options

Cost tiers, worst case first.

#### LOCAL FREE — no account, no key, nothing leaves your machine

| Provider | What it is | Cost |
|---|---|---|
| **Ollama** | Open-weight models on your own hardware. `ollama serve`, then `ollama pull <model>`. | Free. Hardware and electricity. |
| **OmniRoute** | A local gateway on your machine that fronts several of them. | Free. Hardware and electricity. |

#### FREE-TIER API — a key, but a quota that runs out

Groq, Gemini and Cerebras have free tiers with no card attached. The quota
resets, the model catalogue changes, and none of that is a promise this
repository makes. Treat the table as a starting point, not a guarantee.

#### PAID API — metered, needs a card

OpenAI, Anthropic, DeepSeek, Cloudflare, GitHub Models and Hugging Face. Same
routing, same fallback, different billing.

OmniRoute is a **gateway, not a model**. It has no weights of its own and decides
per request which backend answers, so it is configured with a URL and `auto`:

```env
LLM_PROVIDER=omniroute
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto
OMNIROUTE_MAX_TOKENS=2048
```

Leave `OMNIROUTE_API_KEY` blank and no `Authorization` header is sent at all, so
a gateway that runs without authentication is never handed a placeholder token.

### Switching models without a code change

Everything is configuration. No model id is hard-coded in the application.

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
```

```env
LLM_PROVIDER=omniroute
OMNIROUTE_MODEL=auto
```

```env
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

Restart the backend after editing. `run.py` watches `app/` for changes, but `.env`
is read at startup, so a reload will not pick up a new model id.

The registry in `app/services/llm/registry.py` is the single list that the router,
the dashboard and `.env.example` all read, so a new provider is one entry rather
than three edits in three places.

### Routing

```bash
LLM_MODE=auto       # local first, then the online chain
LLM_MODE=ollama     # local only, never leaves the machine
LLM_MODE=online     # online chain only, never touches a local provider
```

`LLM_MODE` wins over the older `LLM_PROVIDER` / `LLM_ROUTE_MODE` pair, which is
still read when `LLM_MODE` is unset.

| Variable | Default | Effect |
|---|---|---|
| `LLM_FALLBACK_ENABLED` | `true` | `false` means exactly one provider, whatever happens. |
| `LLM_MAX_PROVIDER_ATTEMPTS` | `4` | Cap on providers tried per request. Minimum 1. |
| `LLM_RETRIES` | `0` | Extra attempts against the *same* provider. `0` means one attempt. |
| `LLM_RETRY_BACKOFF_SECONDS` | `1.5` | Linear backoff per attempt. |

Fallback happens on retryable failures: timeouts, rate limits, 5xx, unreachable
endpoints. It is not triggered by a parsing bug in this application, and
`LLM_MAX_PROVIDER_ATTEMPTS` bounds it, so one request can never turn into an
unbounded number of billable ones.

Retries are narrower. A bad key, an unknown model, an exhausted balance, a
malformed body or an empty answer are not retried, because sending the identical
request again produces the identical result. A missing *local* model is the
interesting case: retrying the same provider cannot install it, so there is no
retry, but a different provider can serve the request, so the router does move
on. That is why the two decisions are separate flags rather than one boolean.

Provider ids go stale. That is expected and not a defect: when one dies, fix the
one line in `.env` and carry on.

### Output ceilings

Each provider has its own variable, read per request:

```bash
OLLAMA_MAX_TOKENS=2048          # OLLAMA_MAX_OUTPUT_TOKENS accepted as an alias
OMNIROUTE_MAX_TOKENS=2048
GEMINI_MAX_TOKENS=2048
GROQ_MAX_TOKENS=2048
OPENROUTER_MAX_TOKENS=2048
OPENAI_MAX_TOKENS=2048
CLAUDE_MAX_TOKENS=4096
DEEPSEEK_MAX_TOKENS=2048
LLM_MAX_TOKENS=2048             # fallback where a provider has none
```

A per-task budget is layered on top, because a full CV body needs more room than
a keyword extraction. A ceiling never collapses to zero, and on Ollama's `/v1`
shim it is reduced to what the shared window can actually hold.

### Model discovery

Discovery is optional. A provider with no listing still works with a hand-set
model id.

```bash
python -m scripts.llm_smoke --list              # every provider
python -m scripts.llm_smoke --list ollama       # one provider
curl http://localhost:8000/api/v1/resume/model-discovery
curl "http://localhost:8000/api/v1/resume/model-discovery?provider=groq"
```

Each row carries the provider, the model id, where it came from (`discovery` or
`configuration`), and any context length the provider actually stated. Nothing is
inferred, so an unstated context length stays absent rather than guessed from a
model name. The configured model is always listed, flagged if discovery did not
return it, which makes a typo in `.env` visible immediately.

### Smoke testing

```bash
python -m scripts.llm_smoke              # every configured provider
python -m scripts.llm_smoke ollama groq  # selected providers
python -m scripts.llm_smoke --json       # machine-readable
```

Per provider it reports reachability, model availability, whether a trivial
prompt succeeded, whether usable text came back, latency, and token usage where
the provider reports it. An unconfigured provider reports `NOT CONFIGURED` rather
than failing the run, so a half-configured machine still tells you something.

It sends one fixed harmless prompt (`Reply with exactly the word: acknowledged`)
and never sends your CV or a posting. It prints no credential, no prompt and no
response text, only lengths, counts and a failure category.

```text
PROVIDER      STATUS           MODEL                         SECS  DETAIL
ollama        OK               qwen3:8b                       6.5  Usable text returned.
gemini        OK               gemini-2.5-flash               3.9  Usable text returned.
cerebras      NOT CONFIGURED   -                                -  No credential. Set CEREBRAS_API_KEY.
deepseek      FAILED           deepseek-chat                  1.0  [insufficient_credits] Error code: 402 ...
```

### Verified status

A record of runs, not a capability list. Measured with
`python -m scripts.model_matrix`, in the environment this was developed in. A
failure here is a fact about that environment, an invalid key or an empty
balance, not a statement about the provider.

Three columns, three different claims. AVAILABLE means the provider answered a
listing request, which says nothing about whether it can generate. Smoke tested
means a trivial prompt completed. Full CV pipeline means a document came out the
other end. They are kept apart because collapsing them would let the table claim
more than it measured.

#### Local models

| Model | Size | AVAILABLE | Smoke tested | Full CV pipeline | Notes |
|---|---|---|---|---|---|
| `qwen3:8b` | 4.9 GB | PASS | PASS | PASS | The one this project is developed against. It completed the entire pipeline: 16.8 s, an 86 837-byte one-page PDF with 7 sections and 0 artifacts |
| `llama3.2` | 1.9 GB | PASS | PASS | PASS | Fastest, and it completed the entire pipeline. Intermittent failures remain, because a small model has a long tail |
| `qwen2.5:7b` | 4.4 GB | PASS | PASS | FAIL | Emits malformed links often enough to matter. They are dropped rather than fatal |
| `deepseek-r1:8b` | 4.9 GB | PASS | PASS | FAIL | Answers a one-word prompt with 774 characters of reasoning prose, and `think:false` does not suppress it |
| `glm-4.7-flash` | 17.7 GB | PASS | PASS | NOT TESTED | Smoke-tested only. Not run through the pipeline |
| `qwen3.6` | 22.3 GB | PASS | PASS | NOT TESTED | Smoke-tested only. Not run through the pipeline |
| `llama3.3:70b` | 39.6 GB | PASS | FAIL | NOT TESTED | Returns 500 from Ollama. On that machine 8 GB of VRAM against a 39.6 GB model, so this is a hardware limit rather than a defect |

#### API providers

| Provider | Configured | AVAILABLE | Smoke tested | Full CV pipeline | Result |
|---|---|---|---|---|---|
| `groq` | yes | PASS | PASS | PASS | 16.8 s, 3 LLM calls, 86 837-byte one-page PDF, 7 sections, 0 artifacts, 89.2% content retained. The first API provider measured past a smoke test |
| `omniroute` | yes | PASS | PASS | NOT TESTED | A gateway was listening on 20128 and answered. One machine at one moment, not a stable property |
| `gemini` | yes | PASS | FAIL | NOT TESTED | 429 `RESOURCE_EXHAUSTED`, the key's quota is spent. An earlier run of the same key passed, so this is billing state |
| `openai` | yes | PASS | FAIL | NOT TESTED | 401, the configured key is not valid |
| `claude` | yes | PASS | FAIL | NOT TESTED | 400, the key is not scoped to a workspace |
| `deepseek` | yes | PASS | FAIL | NOT TESTED | 402, insufficient balance, correctly classified as non-retryable |
| `openrouter` | yes | PASS | PASS | NOT TESTED | PASS once `OPENROUTER_MODEL` named a live id. The previously configured `nvidia/nemotron-3-ultra:free` was retired and answered 400 |
| `experiential` | yes | PASS | FAIL | NOT TESTED | 403, the key does not grant the configured model alias |
| `huggingface` | yes | PASS | NOT CONFIGURED | NOT TESTED | A credential is set, but no model is chosen, so nothing was sent |
| `cerebras` | no | NOT CONFIGURED | NOT CONFIGURED | NOT TESTED | No credential set |
| `cloudflare` | no | NOT CONFIGURED | NOT CONFIGURED | NOT TESTED | No credential set |
| `github` | no | NOT CONFIGURED | NOT CONFIGURED | NOT TESTED | No credential set |

The local models completed the entire pipeline. No API provider except `groq` did.
That is a statement about the credentials in this environment, not about the
providers.

The `openrouter` row is the clearest argument for the rule this project follows.
A model id that carried `:free` six months ago is gone, and the fix was one line
in `.env`. That failure also exposed a real defect, now fixed: a retired model id
arrives as `400 ... is not a valid model ID`, which used to be classified as a
malformed *request* rather than an unavailable *model*, so the router blamed the
prompt and gave up. It now walks that provider's remaining models instead of
failing on the first one.

### Troubleshooting

**"Ollama returned empty content"** The output budget ran out before any answer
was written. On a reasoning model the budget goes into the thinking channel,
leaving `content` empty with `finish_reason=length`. The application refuses to
return half a chain of thought as a CV, which is why it is an error rather than
a short answer. Raise the ceiling (`OLLAMA_MAX_TOKENS=4096`) or use the native
`/api/generate` endpoint, which honours `think:false`. The error reports the
finish reason, available message fields, reasoning length and token counts, never
the content.

**A local model is never used in `auto` mode** Look for `Skipping Ollama` in the
log. On the `/v1` shim the window is shared between prompt and completion, so a
long prompt can leave nothing useful. The log says which of the two reasons
applied. The native endpoint has no such limit.

**`NOT CONFIGURED` for a provider you set up** The message names the variable
that is missing, and it is one of the provider variables in
[`.env.example`](#configure). For a local gateway, confirm the port too:
`python -m scripts.llm_smoke --list omniroute` reports reachability.

**A model id stopped working** Provider catalogues change. Run
`python -m scripts.llm_smoke --list <provider>` for the current list and update
the one line in `.env`.

**`insufficient_credits`** A 402 is an empty balance, not a rate limit. It is not
retried, because retrying cannot add credit.

---

## What gets generated

### DOCX

ATS-friendly, single column, standard headings. No tables, no text boxes, no
sidebars.

### LaTeX and PDF

Ten layouts, compiled locally with `pdflatex`.

A layout counts as supported only if it was actually produced and
content-validated. There is a regression guard that fails if two templates come
out structurally identical, and another that fails if a template claims a
language it does not produce. Both are asserted at import, so a new layout that
silently duplicates an existing one cannot be merged.

Generation is refused rather than truncated in one case. A document that will not
fit on one page after four compaction levels raises a layout error naming the
page count. Content is never dropped quietly.

### PDF content validation

The finished PDF is read back and checked, because a document that compiles is
not the same as a document that came out right:

- sections survived extraction, not just the compile
- reading order, columns, overlapping and clipped text
- invisible text and font size
- per-field parseability

This is what catches a generator that silently loses the experience section.

### Factual validation

The German Minimal ATS layout extracts the immutable facts from the source CV,
companies, titles, dates and degrees, then compares them against the generated
LaTeX. A document that omitted or altered one is rejected with HTTP 422 and the
affected invariants listed. This is the check that makes the "we never invent
experience" claim enforceable rather than aspirational.

---

## Observability

### Pipeline event log

Every stage, parse, analysis, translation, LLM call, LaTeX, PDF compile, DOCX
build, emits a JSONL event sharing one `request_id`.

```bash
curl http://localhost:8000/api/v1/resume/processing-log?limit=50
```

### Quota tracking

Per-provider RPM, RPD and TPM aggregated from that log, with live progress bars
in the dashboard.

### Analytics

Pipeline counts, outcomes and latency over the event log and the tracker
database.

---

## Testing

```bash
python -m pytest -q
```

<!-- BEGIN:AUTO:TESTS -->

**Total tests: 1332** across 57 files.

| Test file | Count |
|-----------|-------|
| `tests/test_agent_core.py` | 24 |
| `tests/test_analyzer.py` | 3 |
| `tests/test_answer_engine.py` | 11 |
| `tests/test_api.py` | 9 |
| `tests/test_ats_improvement.py` | 36 |
| `tests/test_ats_scoring.py` | 65 |
| `tests/test_ats_scoring_helpers.py` | 0 |
| `tests/test_ats_structure_language.py` | 4 |
| `tests/test_audit_regressions.py` | 10 |
| `tests/test_automation_correctness.py` | 11 |
| `tests/test_browser_use_security.py` | 11 |
| `tests/test_compaction_escalation.py` | 10 |
| `tests/test_compiler_diagnostics.py` | 19 |
| `tests/test_context_budgeting.py` | 24 |
| `tests/test_cv_layout_api_errors.py` | 3 |
| `tests/test_cv_layouts.py` | 0 |
| `tests/test_dashboard_query_efficiency.py` | 2 |
| `tests/test_dashboard_workflow.py` | 29 |
| `tests/test_description_parser.py` | 12 |
| `tests/test_discovery.py` | 3 |
| `tests/test_document_wrapper_strip.py` | 10 |
| `tests/test_documentation_consistency.py` | 31 |
| `tests/test_e2e_pipeline.py` | 2 |
| `tests/test_error_recovery.py` | 8 |
| `tests/test_formatting_checks.py` | 47 |
| `tests/test_gateway.py` | 1 |
| `tests/test_generation_length_budget.py` | 6 |
| `tests/test_german_minimal_ats.py` | 7 |
| `tests/test_latex_escape.py` | 58 |
| `tests/test_latex_special_characters_compile.py` | 20 |
| `tests/test_layout_recommender.py` | 17 |
| `tests/test_link_handling.py` | 13 |
| `tests/test_live_status.py` | 3 |
| `tests/test_llm_routing.py` | 57 |
| `tests/test_mock_ats_form.py` | 12 |
| `tests/test_new_sources.py` | 14 |
| `tests/test_ollama_qwen3_generation.py` | 85 |
| `tests/test_orchestrator_retry.py` | 10 |
| `tests/test_package_validator.py` | 9 |
| `tests/test_page_fill.py` | 13 |
| `tests/test_parser.py` | 4 |
| `tests/test_pdf_compiler.py` | 20 |
| `tests/test_pdf_content_validation.py` | 27 |
| `tests/test_pdf_layouts.py` | 7 |
| `tests/test_preamble_fragment_strip.py` | 13 |
| `tests/test_provider_registry.py` | 53 |
| `tests/test_rate_limit_handling.py` | 18 |
| `tests/test_rate_limiting.py` | 21 |
| `tests/test_requirement_extraction.py` | 36 |
| `tests/test_retry_and_call_count.py` | 15 |
| `tests/test_review_regressions.py` | 11 |
| `tests/test_second_audit.py` | 8 |
| `tests/test_security.py` | 8 |
| `tests/test_structured_llm.py` | 3 |
| `tests/test_structured_logger.py` | 6 |
| `tests/test_variant_router.py` | 10 |
| `tests/test_verification.py` | 6 |

<!-- END:AUTO:TESTS -->

Linting is `ruff check app tests`, and it runs in CI.

Regenerate the auto-generated blocks in this file after changing the route table,
the tree or the test count:

```bash
python scripts/update_readme.py
```

---

## Project structure

<!-- BEGIN:AUTO:TREE -->

```text
ai-resume-analyzer/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   ├── jobs.py
│   │   ├── new_features_endpoints.py
│   │   ├── streaming_endpoints.py
│   │   └── utils.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── event_log.py
│   │   ├── network.py
│   │   ├── rate_limit.py
│   │   └── security.py
│   ├── dashboard/
│   │   ├── views/
│   │   │   ├── __init__.py
│   │   │   ├── advanced_tools.py
│   │   │   ├── agent_control.py
│   │   │   ├── analytics.py
│   │   │   ├── ats_step.py
│   │   │   ├── auto_apply.py
│   │   │   ├── career_suite.py
│   │   │   ├── cv_generator.py
│   │   │   ├── diagnostics.py
│   │   │   ├── inputs.py
│   │   │   ├── layout_picker.py
│   │   │   ├── llm_settings.py
│   │   │   ├── optimization.py
│   │   │   ├── overview.py
│   │   │   ├── pdf_preview.py
│   │   │   └── workflow_page.py
│   │   ├── __init__.py
│   │   ├── components.py
│   │   ├── helpers.py
│   │   ├── main.py
│   │   ├── theme.py
│   │   └── workflow.py
│   ├── data/
│   │   ├── llm_processing.jsonl
│   │   ├── resume_feedback.db
│   │   ├── resume_versions.db
│   │   └── skill_progression.db
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── analysis/
│   │   │   ├── ats_analyzer.py
│   │   │   ├── ats_improvement.py
│   │   │   ├── ats_scoring.py
│   │   │   ├── authenticity_checker.py
│   │   │   ├── formatting_checks.py
│   │   │   ├── layout_recommender.py
│   │   │   ├── market_insights.py
│   │   │   ├── requirement_extraction.py
│   │   │   ├── score_explainability.py
│   │   │   ├── skill_roadmap.py
│   │   │   └── suggestions.py
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   └── dashboard_aggregator.py
│   │   ├── bulk/
│   │   │   ├── __init__.py
│   │   │   └── bulk_analyzer.py
│   │   ├── career/
│   │   │   ├── __init__.py
│   │   │   ├── audit_matrix.py
│   │   │   ├── cover_letter.py
│   │   │   ├── cover_letter_pdf.py
│   │   │   ├── cover_letter_templates.py
│   │   │   ├── interview_prep.py
│   │   │   ├── interview_questions.py
│   │   │   ├── interview_simulator.py
│   │   │   └── linkedin_optimizer.py
│   │   ├── cv/
│   │   │   ├── __init__.py
│   │   │   ├── diff_preview.py
│   │   │   ├── latex_escape.py
│   │   │   ├── latex_generator.py
│   │   │   ├── optimizer.py
│   │   │   ├── pdf_compiler.py
│   │   │   ├── pdf_validation.py
│   │   │   └── variant_router.py
│   │   ├── jobs/
│   │   │   ├── __init__.py
│   │   │   ├── agent_schemas.py
│   │   │   ├── answer_engine.py
│   │   │   ├── apply_linkedin.py
│   │   │   ├── arbeitsagentur.py
│   │   │   ├── auto_runner.py
│   │   │   ├── backend_submitter.py
│   │   │   ├── browser_use_applier.py
│   │   │   ├── company_boards.py
│   │   │   ├── decision_engine.py
│   │   │   ├── dedupe.py
│   │   │   ├── description_parser.py
│   │   │   ├── discovery.py
│   │   │   ├── email_reader.py
│   │   │   ├── error_recovery.py
│   │   │   ├── finder.py
│   │   │   ├── full_pipeline.py
│   │   │   ├── jd_fetcher.py
│   │   │   ├── job_metadata.py
│   │   │   ├── orchestrator.py
│   │   │   ├── package_validator.py
│   │   │   ├── profile_manager.py
│   │   │   ├── scheduler.py
│   │   │   └── verification.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── provider.py
│   │   │   ├── quota_tracker.py
│   │   │   └── registry.py
│   │   ├── observability/
│   │   │   ├── __init__.py
│   │   │   └── structured_logger.py
│   │   ├── parsing/
│   │   │   ├── __init__.py
│   │   │   └── resume_parser.py
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── analytics.py
│   │   │   ├── collaborative_feedback.py
│   │   │   ├── live_status.py
│   │   │   ├── state_machine.py
│   │   │   ├── tracker.py
│   │   │   └── version_manager.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── fixtures/
│   │   ├── applicant_profile.yaml
│   │   ├── jd.txt
│   │   ├── mock_greenhouse_form.html
│   │   ├── output.docx
│   │   └── resume.txt
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agent_core.py
│   ├── test_analyzer.py
│   ├── test_answer_engine.py
│   ├── test_api.py
│   ├── test_ats_improvement.py
│   ├── test_ats_scoring.py
│   ├── test_ats_scoring_helpers.py
│   ├── test_ats_structure_language.py
│   ├── test_audit_regressions.py
│   ├── test_automation_correctness.py
│   ├── test_browser_use_security.py
│   ├── test_compaction_escalation.py
│   ├── test_compiler_diagnostics.py
│   ├── test_context_budgeting.py
│   ├── test_cv_layout_api_errors.py
│   ├── test_cv_layouts.py
│   ├── test_dashboard_query_efficiency.py
│   ├── test_dashboard_workflow.py
│   ├── test_description_parser.py
│   ├── test_discovery.py
│   ├── test_document_wrapper_strip.py
│   ├── test_documentation_consistency.py
│   ├── test_e2e_pipeline.py
│   ├── test_error_recovery.py
│   ├── test_formatting_checks.py
│   ├── test_gateway.py
│   ├── test_generation_length_budget.py
│   ├── test_german_minimal_ats.py
│   ├── test_latex_escape.py
│   ├── test_latex_special_characters_compile.py
│   ├── test_layout_recommender.py
│   ├── test_link_handling.py
│   ├── test_live_status.py
│   ├── test_llm_routing.py
│   ├── test_mock_ats_form.py
│   ├── test_new_sources.py
│   ├── test_ollama_qwen3_generation.py
│   ├── test_orchestrator_retry.py
│   ├── test_package_validator.py
│   ├── test_page_fill.py
│   ├── test_parser.py
│   ├── test_pdf_compiler.py
│   ├── test_pdf_content_validation.py
│   ├── test_pdf_layouts.py
│   ├── test_preamble_fragment_strip.py
│   ├── test_provider_registry.py
│   ├── test_rate_limit_handling.py
│   ├── test_rate_limiting.py
│   ├── test_requirement_extraction.py
│   ├── test_retry_and_call_count.py
│   ├── test_review_regressions.py
│   ├── test_second_audit.py
│   ├── test_security.py
│   ├── test_structured_llm.py
│   ├── test_structured_logger.py
│   ├── test_variant_router.py
│   └── test_verification.py
├── scripts/
│   ├── ai_fix.py
│   ├── check_endpoints.py
│   ├── check_hf_hub.py
│   ├── llm_smoke.py
│   ├── model_matrix.py
│   ├── run_smoke_tests.py
│   ├── setup_browser_profile.py
│   ├── start_brave_debug.bat
│   ├── start_edge_debug.bat
│   └── update_readme.py
├── app_watchdog.py
├── main.py
├── run.py
├── run_agent_continuous.py
├── run_job_agent.py
├── run_job_search.py
├── run_linkedin_indeed.py
├── test_linkedin_apply.py
```

<!-- END:AUTO:TREE -->

Service modules live in the subpackage matching their domain: LLM providers in
`services/llm/`, scoring in `services/analysis/`, document generation in
`services/cv/`.

---

## How the scoring works

Five weighted categories, each reporting its own points won and lost so the total
is reconstructible:

| Category | What it measures |
|---|---|
| Keyword match | Alignment with the posting's skills and role |
| Formatting | Structure, headings, contact details, density |
| Semantic relevance | Whether the document is about the same work |
| Experience relevance | Seniority, duration and recency |
| PDF parsing | Measured on the finished document only |

Before generation there is no document to parse, so that category reports
`not_measured`, its weight is redistributed across the rest, and the result is
labelled pre-generation. `POST /validate-ats` scores the finished PDF, and that is
the number to act on. The two are stored separately because they are different
measurements.

Requirement-level checks sit on top of keyword matching: degree level, named
certifications, language ability with CEFR levels, and industry terminology like
GDPR, HIPAA, SOX and PCI DSS. Both English and German wording is accepted, in the
posting and in the CV.

The factual invariant check is described above and is deliberately separate from
scoring. It gates output; it does not contribute to a number.

---

## Security

### Enforced

**API access.** Every `/api/v1` route requires `X-API-Key` matching `API_KEY`.
Compose refuses to start without it, and both published ports bind to `127.0.0.1`,
so neither service is reachable from another machine by default.

**SSRF.** `app/core/security.py` validates every outbound URL. `file://`,
`javascript:`, `data:`, credentials in the authority, and loopback, link-local,
private and reserved addresses are all rejected, including the decimal
(`http://2130706433/`) and octal (`http://0177.0.0.1/`) spellings of
`127.0.0.1`. DNS is resolved and the resulting address checked, so a hostname
pointing at a private address is rejected too, which closes DNS rebinding. Job
URLs get the same treatment before a browser navigates anywhere.

**Path traversal.** `validate_local_file` resolves the candidate path and
requires it inside `ALLOWED_FILE_ROOTS`. Generated PDFs get unique names rather
than reusing a caller-supplied one.

**LaTeX execution.** `pdflatex` runs with `-no-shell-escape`, never `shell=True`,
in a private temporary directory, under a bounded timeout, with a workspace size
cap. The source is validated before the compiler is invoked at all: a supported
A4 article class, balanced braces and environments.

**Compiler diagnostics.** A LaTeX error is the one place untrusted document text
could reach a log or an HTTP response, because `pdflatex` echoes the failing
source line. Three things are handled separately in
`app/services/cv/pdf_compiler.py`:

- the candidate's source line is withheld and reported by line number and length
- secret-named environment values are replaced wherever they appear, along with
  recognisable credential shapes (`sk-`, `AIza`, `ghp_`, `gsk_`, `hf_`, `xai-`,
  JWTs, PEM blocks) for secrets this deployment does not hold
- the temporary working directory is replaced, since its path contains the
  operator's username

The useful diagnostic is preserved in full: category, message, line numbers, file
names and the offending character. An earlier implementation replaced every run
of four or more letters with a length marker, which turned
`Unescaped LaTeX character '$'` into `<8 chars><1 bytes>` and made every error
unactionable.

**Local providers get no stray credential.** They use plain HTTP with no SDK, and
a provider with no configured key sends no `Authorization` header at all, so a
loopback gateway is never handed a placeholder token.

### Not enforced

- **Rate limiting is off by default.** `API_KEY` is a shared secret rather than a
  per-user identity, so a burst spends the provider's token budget and you get
  429s a moment later, which is what this project's own test runs did to a free
  tier. Set `RATE_LIMIT_ENABLED=true` for anything shared. The counters are in
  memory and per process, so with N workers the effective limit is the configured
  value times N. Do not expose the API to the internet regardless.
- **Logs are not a trusted sink.** A provider's error text can contain account
  identifiers or a partially masked key. This application redacts credentials it
  knows about; it cannot redact identifiers a remote service invents. Treat
  `data/llm_processing.jsonl` and the application log as sensitive.
- **Resume content lands in the database.** `data/applications.db` and
  `app/data/*.db` hold employment history and URLs. Both are gitignored and
  created on first use rather than shipped.
- **Prompt injection from a posting is contained, not solved.** A job description
  is untrusted input. The generator fences injected instructions and the factual
  check refuses output that dropped or altered a stated career fact, but a
  sufficiently adversarial posting is not a solved problem.

### Secrets

`.env` and `.env.*` are gitignored. `.env.example` is explicitly allowed back and
holds no values. `tests/test_documentation_consistency.py` fails the build if
`.env.example` grows anything shaped like a key. Remember that `git log -p` can
still surface a removed one: rotate any key that was ever committed.

---

## Known limitations

Each of these is measured, and each is left in place on purpose.

**A CV too long for one page is refused, not truncated.** After four compaction
levels it still raises a layout error naming the page count. Content is never
dropped silently, and the factual check would catch it if it were.

**A very short CV cannot be made to fill a page.** The fill pass opens up the
leading of a document that stops far too high, measured on the rendered page
rather than estimated, and takes a typical CV from 37% empty to 18%. The leading
is capped at 1.35x, so a sixteen-line CV still leaves about half the page empty.
Stretching a few lines over 200mm reads worse than the space does. The answer is
a longer CV.

**`llama3.2` is reliable but not dependable.** It passes far more often than it
used to, but it still fails intermittently, and a small model has a long tail.
`qwen2.5:7b` is unreliable for a different reason: it emits malformed links,
which are dropped rather than fatal, often enough to matter.

**Deep reasoning models do not work for CV generation here.**
`deepseek-r1:8b` answers a one-word prompt with 774 characters of reasoning
prose, and `think:false` does not suppress it. It passes the smoke test, which is
exactly why the smoke test and the pipeline are separate columns.

**Three large local models are smoke-tested only.** `glm-4.7-flash` (19 GB),
`qwen3.6` (24 GB) and `llama3.3:70b` (43 GB) were not run through the full
pipeline. The first two passed the smoke test; the third returns 500 from Ollama,
which on that machine is a hardware limit, an RTX 4060 with 8 GB of VRAM. It will
behave differently with more memory and nothing here has been measured there.

**No API provider except `groq` has been measured past a smoke test.** That is a
statement about credentials, not about the providers, but the honest position is
that remote generation through this application is largely unverified.

**One-page compaction has a floor.** Level 3 drops body text to 10pt with 0.5 cm
margins. Past that the document is refused rather than set in a smaller face,
because a CV at 8pt is not a CV.

**Semantic embeddings only install on Python 3.13.** The feature is verified
working, two paraphrases of one achievement scoring 0.574 cosine similarity
against 0.044 for unrelated text, but `sentence-transformers` pulls `transformers`,
which needs `regex>=2025.10.22`, and python-jobspy pins `regex<2025.0.0`. On 3.13
the jobspy entry is skipped and nothing conflicts. Either way the application is
unaffected: the import is guarded and it falls back to TF-IDF and says so.

**Python 3.13 loses one discovery source.** See
[Prerequisites](#prerequisites).

**The layout guard reads structure, not appearance.** It compares packages,
colours, macros, column machinery, heading treatment, margins, body type size and
measure. All ten templates are distinct under it. It cannot see kerning, or a
colour that renders almost identically, so it is a guard against convergence
rather than a description of the pages.

**The pre-generation score is not the final score.** Covered in
[How the scoring works](#how-the-scoring-works).

**The keyword-stuffing check is a threshold, and thresholds lie.** It fires when
a small set of content words exceeds 5% of the document, floor of 4 occurrences,
excluding a stopword list. Without that exclusion, "with", "team" and "work" are
the most repeated words in any CV and every CV gets accused of padding. It is
tuned quiet, because a false positive tells a candidate their CV is padded and
then recommends a rewrite producing the same document. This is the one check not
yet calibrated against a corpus of real CVs, and the one most likely to be wrong
in either direction.

**Requirement extraction is pattern-based and has blind spots.** Degree level,
certifications and languages are found by matching labelled sections and named
credentials, so an unusual phrasing or an alternative spelling is not recognised,
and a miss lowers the score without the candidate having done anything wrong. The
checks are additive to keyword matching, so a named skill the candidate did list
still counts.

**Streamlit's execution model makes some things expensive.** Streaming LLM
responses into the dashboard, and A/B testing the LinkedIn About section, are
deferred for that reason rather than overlooked.

---

## Contributing

```bash
git checkout -b feature/improvement
# make changes
python -m pytest -q
python -m ruff check app tests
git commit -m "Add feature"
```

New service modules go in the subpackage matching their domain. A new LLM
provider belongs in `services/llm/registry.py`, which is the single list the
router, the dashboard and `.env.example` all read.

Pre-commit runs on commit. `ruff` and `ruff-format` are deliberately not in the
hook set: they rewrote files during a commit, which pre-commit treats as a
failure, so the commit aborted and had to be retyped. Linting still runs in CI.

---

## License

MIT. See the repository for details.
