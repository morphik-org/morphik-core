from pathlib import Path
from typing import Any, Optional

import tomli
from dotenv import load_dotenv


def _secret_manager_from_toml() -> Optional[str]:
    """Peek at morphik.toml for the secret_manager setting."""
    toml_path = Path("morphik.toml")
    if not toml_path.exists():
        return None
    try:
        with toml_path.open("rb") as f:
            data = tomli.load(f)
        return data.get("morphik", {}).get("secret_manager")
    except Exception:
        return None


def should_use_dotenv() -> bool:
    """Return True when local .env files should be loaded."""
    toml_value = _secret_manager_from_toml()
    if toml_value:
        return toml_value.lower() == "env"

    # Default if nothing is specified
    return True


def load_local_env(*args: Any, **kwargs: Any) -> None:
    """
    Load a local .env file if the secret manager is set to 'env'.
    Accepts the same arguments as python-dotenv's load_dotenv.
    """
    if should_use_dotenv():
        load_dotenv(*args, **kwargs)
