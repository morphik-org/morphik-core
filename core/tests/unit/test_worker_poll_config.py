"""Configure queue latency without increasing document-processing concurrency."""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import get_settings

TEMPLATE = Path(__file__).resolve().parents[3] / "morphik.docker.toml"


@pytest.fixture
def configure(tmp_path, monkeypatch):
    template = TEMPLATE.read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JWT_SECRET_KEY", "worker-config-unit-test")
    monkeypatch.setenv("POSTGRES_URI", "postgresql+asyncpg://test:test@localhost/test")
    # Non-secret queue tuning comes from morphik.toml, including when the option is omitted.
    monkeypatch.setenv("ARQ_POLL_DELAY", "0.001")
    get_settings.cache_clear()

    def load(value=None, omit_worker=False):
        lines = template.splitlines()
        lines = [line for line in lines if not line.startswith("poll_delay =")]
        config = "\n".join(lines)
        if value is not None:
            config = config.replace("[worker]", f"[worker]\npoll_delay = {value}")
        if omit_worker:
            config = re.sub(r"\[worker\][^\[]*", "", config)
        (tmp_path / "morphik.toml").write_text(config)
        get_settings.cache_clear()
        return get_settings()

    yield load
    get_settings.cache_clear()


def test_existing_config_keeps_two_second_polling(configure):
    settings = configure()
    assert settings.ARQ_POLL_DELAY == 2.0
    assert settings.ARQ_MAX_JOBS == 1


def test_short_queue_interval_does_not_raise_job_concurrency(configure):
    settings = configure("0.1")
    assert settings.ARQ_POLL_DELAY == 0.1
    assert settings.ARQ_MAX_JOBS == 1


def test_config_without_worker_section_uses_compatible_defaults(configure):
    settings = configure(omit_worker=True)
    assert settings.ARQ_POLL_DELAY == 2.0
    assert settings.ARQ_MAX_JOBS == 1


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan"])
def test_invalid_poll_intervals_fail_at_startup(configure, value):
    with pytest.raises(ValidationError, match="ARQ_POLL_DELAY"):
        configure(value)
