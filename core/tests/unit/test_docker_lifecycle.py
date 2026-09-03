"""Static guards for the installer-generated production lifecycle scripts."""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _installer_start_script() -> str:
    installer = _read("install_docker.sh")
    marker = "cat > start-morphik.sh << 'EOF'\n"
    return installer.split(marker, 1)[1].split("\nEOF\nchmod +x start-morphik.sh", 1)[0]


def test_production_compose_persists_postgres_without_fixed_container_names():
    compose = _read("docker-compose.run.yml")

    assert "postgres_data:/var/lib/postgresql/data" in compose
    assert "container_name:" not in compose
    assert '"5432:5432"' not in compose
    assert "${MORPHIK_API_PORT:-8000}:${MORPHIK_API_PORT:-8000}" in compose
    assert '"${MORPHIK_ENV_FILE:-.env}"' in compose
    assert "required:" not in compose
    assert "# COMPOSE_PROJECT_NAME=morphik" in _read(".env.example")
    assert "LITELLM_LOCAL_MODEL_COST_MAP=${LITELLM_LOCAL_MODEL_COST_MAP:-True}" in compose


def test_normal_stop_preserves_volumes_and_stops_all_profiles():
    checked_in_stop = _read("stop-morphik.sh")
    unix_installer = _read("install_docker.sh")
    windows_installer = _read("install_docker.ps1")

    assert '--profile "*" down --remove-orphans' in checked_in_stop
    assert "down --volumes" not in checked_in_stop
    assert "down --volumes" not in unix_installer
    assert "'down','--volumes'" not in windows_installer
    assert "Persistent named volumes were preserved" in unix_installer
    assert "Persistent named volumes were preserved" in windows_installer


def test_persistence_test_uses_a_non_overridable_unique_project():
    script = _read("scripts/test_postgres_persistence.sh")

    assert 'TEST_PROJECT="morphik-persistence-test-$$-${RANDOM}"' in script
    assert "MORPHIK_PERSISTENCE_TEST_PROJECT" not in script
    assert "Refusing to reuse existing Docker resources" in script
    assert "\ncleanup\n" not in script


def test_start_is_repeatable_without_rewriting_compose():
    checked_in_start = _read("start-morphik.sh")
    unix_installer = _read("install_docker.sh")
    windows_installer = _read("install_docker.ps1")

    assert "export MORPHIK_API_PORT=" in checked_in_start
    assert "docker-compose.run.yml.tmp" not in checked_in_start
    assert "up -d --remove-orphans" in checked_in_start
    assert "export MORPHIK_API_PORT=" in unix_installer
    assert "`$env:MORPHIK_API_PORT = `$desired" in windows_installer


def test_docker_guide_uses_the_production_compose_path_consistently():
    guide = _read("DOCKER.md")

    assert "./install_docker.sh" in guide
    assert "docker compose up --build" not in guide
    assert "docker-compose.yml" not in guide
    assert "docker-compose.run.yml" in guide


@pytest.mark.parametrize(
    "script_text",
    [_read("start-morphik.sh"), _installer_start_script()],
    ids=["checked-in", "installer-generated"],
)
def test_start_defaults_to_latest_when_env_omits_version(tmp_path, script_text):
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    script = deployment / "start-morphik.sh"
    script.write_text(script_text, encoding="utf-8")
    script.chmod(0o755)
    (deployment / ".env").write_text("JWT_SECRET_KEY=test-only\n", encoding="utf-8")
    (deployment / "morphik.toml").write_text("[api]\nport = 8123\n", encoding="utf-8")
    (deployment / "docker-compose.run.yml").write_text("services: {}\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env bash\nprintf "%s|%s\\n" "$MORPHIK_VERSION" "$MORPHIK_API_PORT" > "$FAKE_DOCKER_OUTPUT"\n',
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    docker_output = tmp_path / "docker-output"

    env = os.environ.copy()
    env.pop("MORPHIK_VERSION", None)
    env.pop("COMPOSE_PROFILES", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_DOCKER_OUTPUT": str(docker_output),
        }
    )

    completed = subprocess.run(
        [str(script)],
        cwd=deployment,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Using Morphik version: latest" in completed.stdout
    assert docker_output.read_text(encoding="utf-8").strip() == "latest|8123"
