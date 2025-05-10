import os
from functools import lru_cache
from typing import List, Optional

import tomli
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

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
    load_dotenv(override=True)

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

    return EESettings(**settings_kwargs)
