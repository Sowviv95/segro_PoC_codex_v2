from pathlib import Path

import pytest
from pydantic import ValidationError

from segro_evidence_extraction.config import load_settings


def test_load_settings_from_file_and_env_without_real_dotenv() -> None:
    settings = load_settings(
        env={
            "SEGRO_ENV": "test",
            "SEGRO_LOG_LEVEL": "debug",
            "OPENAI_API_KEY": "secret-value",
            "ALLOW_HOSTED_LLM": "true",
        },
    )

    assert settings.environment_name == "test"
    assert settings.log_level == "DEBUG"
    assert settings.hosted_llm_enabled is True
    assert "secret-value" not in repr(settings)
    assert "openai_api_key" not in settings.model_dump()


def test_load_settings_from_config_file_without_real_dotenv() -> None:
    settings = load_settings(
        Path("configs/default.yaml"),
        env={
            "SEGRO_ENV": "test",
            "SEGRO_LOG_LEVEL": "debug",
            "OPENAI_API_KEY": "secret-value",
            "ALLOW_HOSTED_LLM": "true",
        },
    )

    assert settings.environment_name == "test"
    assert settings.log_level == "DEBUG"
    assert settings.hosted_llm_enabled is True
    assert "secret-value" not in repr(settings)
    assert "openai_api_key" not in settings.model_dump()


def test_invalid_log_level_fails_clearly() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        load_settings(env={"SEGRO_LOG_LEVEL": "LOUD"})
