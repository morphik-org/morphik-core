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
    assert "trap cleanup EXIT" in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script
    assert "trap cleanup EXIT INT TERM" not in script


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


def _run_start_script(tmp_path, script_text, toml_text, with_backup_tool=True):
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    script = deployment / "start-morphik.sh"
    script.write_text(script_text, encoding="utf-8")
    script.chmod(0o755)
    (deployment / ".env").write_text("JWT_SECRET_KEY=test-only\n", encoding="utf-8")
    (deployment / "morphik.toml").write_text(toml_text, encoding="utf-8")
    (deployment / "docker-compose.run.yml").write_text("services: {}\n", encoding="utf-8")
    if with_backup_tool:
        (deployment / "morphik-backup.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env bash\nprintf "%s|%s\\n" "$*" "${MORPHIK_BACKUP_DIR:-}" > "$FAKE_DOCKER_OUTPUT"\n',
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    docker_output = tmp_path / "docker-output"

    env = os.environ.copy()
    for key in ("MORPHIK_VERSION", "COMPOSE_PROFILES", "MORPHIK_BACKUP_DIR"):
        env.pop(key, None)
    env.update({"PATH": f"{fake_bin}:{env['PATH']}", "FAKE_DOCKER_OUTPUT": str(docker_output)})

    completed = subprocess.run([str(script)], cwd=deployment, env=env, text=True, capture_output=True, check=False)
    docker_args = docker_output.read_text(encoding="utf-8").strip() if docker_output.exists() else ""
    return completed, docker_args, deployment


START_SCRIPTS = pytest.mark.parametrize(
    "script_text",
    [_read("start-morphik.sh"), _installer_start_script()],
    ids=["checked-in", "installer-generated"],
)


@START_SCRIPTS
def test_start_enables_backup_profiles_from_morphik_toml(tmp_path, script_text):
    toml_text = (
        '[api]\nport = 8123\n\n[backup]  # scheduled\nenabled = true  # on\ndirectory = "./nightly"\n'
        's3_uri = "s3://bucket/morphik"  # off-host\n\n[other]\nenabled = false\n'
    )

    completed, docker_args, deployment = _run_start_script(tmp_path, script_text, toml_text)

    assert completed.returncode == 0, completed.stderr
    args, backup_dir = docker_args.split("|")
    assert "--profile backup --profile backup-s3 up -d --remove-orphans" in args
    assert backup_dir == "./nightly"
    assert (deployment / "nightly").is_dir()
    assert oct((deployment / "nightly").stat().st_mode & 0o777) == "0o700"


@START_SCRIPTS
def test_start_leaves_backups_off_by_default(tmp_path, script_text):
    toml_text = '[api]\nport = 8123\n\n[backup]\nenabled = false\ns3_uri = "s3://bucket/morphik"\n'

    completed, docker_args, deployment = _run_start_script(tmp_path, script_text, toml_text)

    assert completed.returncode == 0, completed.stderr
    assert "--profile backup" not in docker_args
    assert "./morphik-backup.sh backup" in completed.stdout
    assert not (deployment / "backups").exists()


@START_SCRIPTS
def test_start_refuses_scheduled_backups_without_the_backup_tool(tmp_path, script_text):
    toml_text = "[backup]\nenabled = true\n"

    completed, docker_args, _ = _run_start_script(tmp_path, script_text, toml_text, with_backup_tool=False)

    assert completed.returncode != 0
    assert "morphik-backup.sh is missing" in completed.stderr
    assert docker_args == ""


def test_backup_services_are_optional_and_cannot_reach_the_docker_daemon():
    compose = _read("docker-compose.run.yml")
    backup = compose.split("\n  backup:\n", 1)[1].split("\n  backup-s3:\n", 1)[0]
    offsite = compose.split("\n  backup-s3:\n", 1)[1].split("\nnetworks:\n", 1)[0]

    for service in (backup, offsite):
        assert "profiles:" in service
        assert "./morphik-backup.sh:/opt/morphik/morphik-backup.sh:ro" in service
        assert "docker.sock" not in service
        assert "ports:" not in service
    assert "      - backup\n" in backup
    assert "./storage:/install/storage:ro" in backup
    assert "${MORPHIK_BACKUP_DIR:-./backups}:/backups" in backup
    assert "      - backup-s3\n" in offsite
    assert "${MORPHIK_BACKUP_DIR:-./backups}:/backups:ro" in offsite
    assert "storage" not in offsite


def test_backup_tool_is_installed_and_keeps_data_safe():
    tool = _read("morphik-backup.sh")
    unix_installer = _read("install_docker.sh")
    windows_installer = _read("install_docker.ps1")

    assert "umask 077" in tool
    assert "down --volumes" not in tool
    assert "--volumes" not in tool
    assert 'pg_restore -U "$PG_USER" -d "$PG_DB" --clean --if-exists --no-owner' in tool
    assert "$REPO_URL/morphik-backup.sh" in unix_installer
    assert "/app/morphik-backup.sh" in unix_installer
    assert "./morphik-backup.sh backup" in unix_installer
    assert "Ensure-BackupTool" in windows_installer
    assert "morphik-backup.ps1" in windows_installer
    assert "COPY morphik-backup.sh ./" in _read("dockerfile")
    assert "backups/" in _read(".gitignore")
    assert "**/backups" in _read(".dockerignore")
