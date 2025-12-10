import os
from functools import lru_cache
from typing import List, Optional

import tomli
from pydantic import Field
from pydantic_settings import BaseSettings

from utils.env_loader import load_local_env

# Determine the root directory of the EE features (assuming this file is in ee/config.py)
EE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKEN_STORAGE_PATH = os.path.join(EE_ROOT_DIR, "connector_tokens")


class EESettings(BaseSettings):
    """Enterprise Edition specific configuration settings."""

    # Google Drive Connector Settings
    # Loaded ONLY from environment variables for security
    GOOGLE_CLIENT_ID: Optional[str] = Field(None, env="GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = Field(None, env="GOOGLE_CLIENT_SECRET")

    # Loaded from ee.toml or environment variables
    GOOGLE_REDIRECT_URI: Optional[str] = "http://localhost:8000/ee/connectors/google_drive/oauth2callback"
    GOOGLE_SCOPES: List[str] = ["https://www.googleapis.com/auth/drive.readonly"]
    GOOGLE_TOKEN_STORAGE_PATH: str = DEFAULT_TOKEN_STORAGE_PATH

    # GitHub Connector Settings
    # Loaded ONLY from environment variables for security
    GITHUB_CLIENT_ID: Optional[str] = Field(None, env="GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: Optional[str] = Field(None, env="GITHUB_CLIENT_SECRET")

    # Loaded from ee.toml or environment variables
    GITHUB_REDIRECT_URI: Optional[str] = "http://localhost:8000/ee/connectors/github/oauth2callback"
    GITHUB_SCOPES: List[str] = ["repo"]
    GITHUB_TOKEN_STORAGE_PATH: str = DEFAULT_TOKEN_STORAGE_PATH

    class Config:
        env_file = ".env"  # Load .env file if present
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from ee.toml


@lru_cache()
def get_ee_settings() -> EESettings:
    """Get cached EE settings instance.

    Loads settings from:
    1. Environment variables (highest precedence).
    2. ee.toml file.
    3. Default values defined in EESettings model.
    """
    load_local_env(override=True)

    ee_toml_path = os.path.join(EE_ROOT_DIR, "ee.toml")
    config_from_toml = {}
    if os.path.exists(ee_toml_path):
        with open(ee_toml_path, "rb") as f:
            toml_data = tomli.load(f)
            if "google_drive" in toml_data:
                config_from_toml = {
                    "GOOGLE_REDIRECT_URI": toml_data["google_drive"].get("redirect_uri"),
                    "GOOGLE_SCOPES": toml_data["google_drive"].get("scopes"),
                    "GOOGLE_TOKEN_STORAGE_PATH": toml_data["google_drive"].get("token_storage_path"),
                }
                # Ensure token storage path is absolute or resolved correctly if relative
                storage_path = config_from_toml.get("GOOGLE_TOKEN_STORAGE_PATH")
                if storage_path and not os.path.isabs(storage_path):
                    config_from_toml["GOOGLE_TOKEN_STORAGE_PATH"] = os.path.join(EE_ROOT_DIR, storage_path)

    # Environment variables will override TOML and defaults thanks to Pydantic-settings
    # We explicitly pass TOML values as defaults to the constructor if they exist.
    # Fields like GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are primarily driven by `env` in Field.
    settings_kwargs = {}
    if config_from_toml.get("GOOGLE_REDIRECT_URI"):
        settings_kwargs["GOOGLE_REDIRECT_URI"] = config_from_toml["GOOGLE_REDIRECT_URI"]
    if config_from_toml.get("GOOGLE_SCOPES"):
        settings_kwargs["GOOGLE_SCOPES"] = config_from_toml["GOOGLE_SCOPES"]
    if config_from_toml.get("GOOGLE_TOKEN_STORAGE_PATH"):
        settings_kwargs["GOOGLE_TOKEN_STORAGE_PATH"] = config_from_toml["GOOGLE_TOKEN_STORAGE_PATH"]

    # -------------------------------------------------------------------------
    # Derive a sensible default GOOGLE_REDIRECT_URI based on the *core* config.
    #   • If the value has already been provided via env or ee.toml we leave it
    #     untouched (handled above).
    #   • Otherwise we look at the global Morphik MODE setting.  In **cloud**
    #     mode the public API is expected to be served from the configured
    #     `API_DOMAIN` (default: api.morphik.ai).  For **self_hosted** mode we
    #     fall back to the traditional localhost development URI.
    # -------------------------------------------------------------------------
    if "GOOGLE_REDIRECT_URI" not in settings_kwargs:
        try:
            # Import lazily to avoid any potential circular dependency issues
            from core.config import get_settings  # pylint: disable=import-error

            core_settings = get_settings()

            if getattr(core_settings, "MODE", "self_hosted") == "cloud":
                api_domain = getattr(core_settings, "API_DOMAIN", "api.morphik.ai")
                derived_redirect = f"https://{api_domain}/ee/connectors/google_drive/oauth2callback"
            else:
                # Default for local/self-hosted development
                derived_redirect = "http://localhost:8000/ee/connectors/google_drive/oauth2callback"

            settings_kwargs["GOOGLE_REDIRECT_URI"] = derived_redirect
        except Exception:
            # In case the core settings cannot be loaded for any reason we keep
            # the safe localhost default to avoid breaking startup.
            settings_kwargs.setdefault(
                "GOOGLE_REDIRECT_URI",
                "http://localhost:8000/ee/connectors/google_drive/oauth2callback",
            )

    return EESettings(**settings_kwargs)
