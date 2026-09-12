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

    MIN_SKILL_CONFIDENCE: float = 0.5
    TOP_KEYWORDS_COUNT: int = 20
    ATS_THRESHOLD_LOW: float = 40.0
    ATS_THRESHOLD_MEDIUM: float = 65.0
    MAX_FILE_SIZE: int = 5 * 1024 * 1024


settings = Settings()
