"""Safe runtime configuration loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator

from segro_evidence_extraction.models.base import StrictBaseModel


class Settings(StrictBaseModel):
    environment_name: str = "development"
    log_level: str = "INFO"
    hosted_llm_enabled: bool = False
    hosted_vlm_enabled: bool = False
    text_model_name: str | None = None
    vision_model_name: str | None = None
    pricing_config_path: Path | None = None
    local_cache_path: Path = Path("data/cache")
    local_index_path: Path = Path("data/indexes")
    openai_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("environment_name", "log_level")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            msg = "value must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("log_level")
    @classmethod
    def log_level_must_be_known(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = "log_level must be one of DEBUG, INFO, WARNING, ERROR or CRITICAL"
            raise ValueError(msg)
        return normalized


def load_settings(config_path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    raw: dict[str, Any] = {}
    if config_path is not None:
        raw.update(_read_yaml(config_path))
    source_env = env if env is not None else __import__("os").environ
    raw.update(_settings_from_env(source_env))
    return Settings.model_validate(raw)


def _read_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        msg = f"Configuration file does not exist: {config_path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _settings_from_env(env: dict[str, str]) -> dict[str, Any]:
    mapping = {
        "SEGRO_ENV": "environment_name",
        "SEGRO_LOG_LEVEL": "log_level",
        "ALLOW_HOSTED_LLM": "hosted_llm_enabled",
        "ALLOW_HOSTED_VLM": "hosted_vlm_enabled",
        "OPENAI_TEXT_MODEL": "text_model_name",
        "OPENAI_VISION_MODEL": "vision_model_name",
        "SEGRO_PRICING_CONFIG_PATH": "pricing_config_path",
        "SEGRO_LOCAL_CACHE_PATH": "local_cache_path",
        "SEGRO_LOCAL_INDEX_PATH": "local_index_path",
        "OPENAI_API_KEY": "openai_api_key",
    }
    result: dict[str, Any] = {}
    for env_name, setting_name in mapping.items():
        if env_name in env and env[env_name] != "":
            value: Any = env[env_name]
            if setting_name in {"hosted_llm_enabled", "hosted_vlm_enabled"}:
                value = env[env_name].lower() in {"1", "true", "yes", "on"}
            result[setting_name] = value
    return result
