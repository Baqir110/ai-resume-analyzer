"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.api.jobs import router as jobs_router
from app.api.new_features_endpoints import router as new_features_router
from app.api.streaming_endpoints import router as streaming_router
from app.core.config import settings
from app.services.llm.provider import LLMService

app = FastAPI(
    title="AI Resume & CV Analyzer API",
    version="1.0.0",
    description=(
        "Automated ATS compatibility scorer, skill gap extractor, "
        "resume optimization engine, and AI-driven job application filler."
    ),
)


@app.get("/health")
def root_health():
    return {"status": "ok"}


@app.get("/v1/models")
def root_models():
    """
    OpenAI-compatible model list, so OpenCode can discover this backend.

    The response shape is fixed by the OpenAI convention and must stay
    ``{"data": [{"id", "object"}]}``.

    The contents are no longer hard-coded to one model name. This previously
    always answered ``qwen3:8b``, so pointing the application at a different
    provider or model left this endpoint lying about what was available. It now
    reports the configured model for every registered provider, which needs no
    network access and cannot fail.
    """
    return {"data": [{"id": model, "object": "model"} for model in configured_models()]}


def configured_models() -> list[str]:
    """Every model a configured provider would use, de-duplicated."""
    models: list[str] = []

    for provider in LLMService.SUPPORTED_PROVIDERS:
        if not LLMService._provider_is_configured(provider):
            continue
        try:
            model = LLMService.get_default_model(provider)
        except Exception:
            # A provider with no model configured is simply skipped; this
            # endpoint must not be able to fail.
            continue
        if model and model not in models:
            models.append(model)

    # An OpenAI-compatible client expects at least one entry, so fall back to
    # whatever the local model is configured as.
    if not models:
        fallback = LLMService.get_default_model("ollama")
        if fallback:
            models.append(fallback)

    return models


_cors_origins = [
    origin.strip() for origin in settings.CORS_ORIGINS if origin.strip() and origin.strip() != "*"
]
if not _cors_origins:
    _cors_origins = ["http://127.0.0.1:8501", "http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=bool(settings.CORS_ALLOW_CREDENTIALS),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key"],
)


@app.middleware("http")
async def limit_request_size(request, call_next):
    """Reject obviously oversized requests before multipart parsing."""
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > int(settings.MAX_REQUEST_BYTES):
        return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """Apply conservative browser headers to every API response."""

    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


app.include_router(
    api_router,
    prefix="/api/v1/resume",
    tags=["Resume Analyzer"],
)

app.include_router(
    new_features_router,
    prefix="/api/v1/resume",
    tags=["New Features"],
)

app.include_router(
    streaming_router,
    prefix="/api/v1/resume",
    tags=["Streaming"],
)

app.include_router(
    jobs_router,
    prefix="/api/v1",
    tags=["Jobs"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
