from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "AI Resume & CV Analyzer API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Override via env var SKILL_TAXONOMY_PATH if needed
    SKILL_TAXONOMY_PATH: str = "data/skills.json"
    APPLICANT_PROFILE_PATH: str = "data/applicant_profile.local.yaml"

    MIN_SKILL_CONFIDENCE: float = 0.5
    ENABLE_SEMANTIC_EMBEDDINGS: bool = False
    TOP_KEYWORDS_COUNT: int = 20
    ATS_THRESHOLD_LOW: float = 40.0
    ATS_THRESHOLD_MEDIUM: float = 65.0
    MAX_FILE_SIZE: int = 5 * 1024 * 1024
    MAX_PACKAGE_BYTES: int = 25 * 1024 * 1024
    MAX_REQUEST_BYTES: int = 30 * 1024 * 1024

    # Security / deployment controls.  Protected operations fail closed when
    # API_KEY is empty; the value must be supplied through the environment or
    # a secret manager, never committed to a profile or source file.
    API_KEY: str = ""
    CORS_ORIGINS: list[str] = [
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ]
    CORS_ALLOW_CREDENTIALS: bool = False
    ALLOWED_FILE_ROOTS: list[str] = ["data"]
    JOB_ALLOWED_HOSTS: list[str] = []

    # Rate limiting.
    #
    # Off by default. A limiter that starts refusing a single-user local
    # deployment is worse than no limiter, and this application is designed to be
    # run on one machine. Turn it on for any shared deployment.
    #
    # On by default, when enabled, is 60 requests a minute per client: generous
    # for a human driving the dashboard, and comfortably below a free tier's
    # per-minute token budget for the LLM-backed routes.
    #
    # Counters are in memory and per process, so with several workers the
    # effective limit is this value times the worker count. That is stated rather
    # than hidden, and a shared store would make the limiter a new thing that can
    # fail.
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: float = 60.0

    # Largest job description the API will accept.
    #
    # This is an input-validation limit, deliberately permissive: rejecting a
    # long posting outright is worse than analysing the part that carries the
    # requirements, which sits at the top of the text.
    #
    # The LLM prompt budget is smaller and lives in
    # app.services.cv.optimizer.MAX_JD_CHARS. Text past that budget does not
    # reach the model, so _truncate() logs exactly how much was dropped
    # instead of the request quietly succeeding on half a posting. Lower this
    # to fail fast on junk input, or raise it together with MAX_JD_CHARS after
    # checking the model's context window.
    MAX_JOB_DESCRIPTION_CHARS: int = 200_000
    MAX_QUESTION_CHARS: int = 2_000
    MAX_ANSWER_CHARS: int = 10_000

    # ------------------------------------------------------------------
    # Autonomous job agent configuration (Section 33). Env-overridable,
    # e.g. MAX_DAILY_APPLICATIONS=15 in .env. Nothing here is a secret;
    # credentials/cookies/tokens must go through .env / secrets manager,
    # never hard-coded (Section 34).
    # ------------------------------------------------------------------

    TARGET_ROLES: list[str] = ["DevOps Engineer", "MLOps Engineer", "IT Administrator"]
    TARGET_LOCATIONS: list[str] = ["Germany"]
    REMOTE_PREFERENCE: str = "hybrid_ok"  # remote_only | hybrid_ok | onsite_ok
    MINIMUM_SALARY: float | None = None
    SALARY_CURRENCY: str = "EUR"
    LANGUAGES: dict[str, str] = {"english": "c1", "german": "b1"}
    EMPLOYMENT_TYPES: list[str] = ["full_time"]

    MINIMUM_MATCH_SCORE: float = 60.0
    HIGH_PRIORITY_THRESHOLD: float = 85.0
    GOOD_MATCH_THRESHOLD: float = 70.0
    MAX_MISSING_REQUIRED_SKILLS: int = 2

    MAX_DAILY_APPLICATIONS: int = 10
    MAX_CONCURRENT_BROWSER_WORKERS: int = 2

    AUTOMATIC_APPLY: bool = False
    AUTOMATIC_SUBMIT: bool = False  # Section 18: default stops at READY_TO_SUBMIT
    COVER_LETTER_POLICY: str = "when_required"  # always | never | when_required

    FOLLOW_UP_DAYS: int = 14
    SOURCES_ENABLED: list[str] = [
        "jobspy",
        "ats_direct",
        "arbeitsagentur",
    ]  # add "web_search" once SERPAPI_KEY is set

    DISCOVER_TIMES: list[str] = ["08:00", "12:00", "16:00"]
    APPLY_TIMES: list[str] = ["10:00", "14:00"]

    # Company boards to poll directly (Section 4) -- add entries as
    # ("greenhouse"|"lever"|"ashby"|"workable"|"smartrecruiters"|"recruitee"|"personio", "<board-handle>", "Display Name")
    COMPANY_BOARDS: list[tuple[str, str, str]] = []

    # Workday boards to poll -- list of dicts:
    # [{"company": "siemens", "tenant": "siemens", "board": "Siemens", "instance": "wd3"}]
    WORKDAY_BOARDS: list[dict] = []

    # ------------------------------------------------------------------
    # Infrastructure: Postgres and Redis
    # Set DATABASE_URL to a postgres:// URI to use Postgres instead of
    # SQLite. Set REDIS_URL to enable Redis-backed caching/queuing.
    # ------------------------------------------------------------------
    DATABASE_URL: str = ""  # empty = use SQLite
    REDIS_URL: str = ""  # empty = skip Redis


settings = Settings()
