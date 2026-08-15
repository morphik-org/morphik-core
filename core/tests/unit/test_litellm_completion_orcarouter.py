"""Unit tests for LiteLLMCompletion api_key_env resolution (used by the OrcaRouter model)."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.completion.litellm_completion import LiteLLMCompletionModel

ORCAROUTER_MODEL = {
    "model_name": "openai/orcarouter/auto",
    "api_base": "https://api.orcarouter.ai/v1",
    "api_key_env": "ORCAROUTER_API_KEY",
}


def _settings_with(registered_models):
    return SimpleNamespace(REGISTERED_MODELS=registered_models)


class TestApiKeyEnvResolution:
    def test_resolves_api_key_env_into_api_key(self):
        os.environ["ORCAROUTER_API_KEY"] = "sk-orca-test"
        try:
            with patch(
                "core.completion.litellm_completion.get_settings",
                return_value=_settings_with({"orcarouter_auto": dict(ORCAROUTER_MODEL)}),
            ):
                model = LiteLLMCompletionModel("orcarouter_auto")
            assert model.model_config["api_key"] == "sk-orca-test"
            assert "api_key_env" not in model.model_config
        finally:
            os.environ.pop("ORCAROUTER_API_KEY", None)

    def test_missing_env_var_warns_and_omits_api_key(self):
        os.environ.pop("ORCAROUTER_API_KEY", None)
        with patch(
            "core.completion.litellm_completion.get_settings",
            return_value=_settings_with({"orcarouter_auto": dict(ORCAROUTER_MODEL)}),
        ):
            model = LiteLLMCompletionModel("orcarouter_auto")
        assert "api_key" not in model.model_config
        assert "api_key_env" not in model.model_config

    def test_model_without_api_key_env_is_unchanged(self):
        with patch(
            "core.completion.litellm_completion.get_settings",
            return_value=_settings_with({"openai_gpt": {"model_name": "gpt-4.1"}}),
        ):
            model = LiteLLMCompletionModel("openai_gpt")
        assert model.model_config == {"model_name": "gpt-4.1"}

    def test_unknown_model_key_raises(self):
        with patch(
            "core.completion.litellm_completion.get_settings",
            return_value=_settings_with({}),
        ), pytest.raises(ValueError, match="not found in registered_models"):
            LiteLLMCompletionModel("does_not_exist")
