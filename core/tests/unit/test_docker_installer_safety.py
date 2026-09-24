import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _generated_shell_script(installer: str, filename: str) -> str:
    match = re.search(rf"cat > {re.escape(filename)} << 'EOF'\n(.*?)\nEOF", installer, re.DOTALL)
    assert match, f"could not find generated {filename}"
    return match.group(1)


def _write_mock_docker(tmp_path: Path, *, container_project: str = "", volume_projects: dict[str, str] | None = None):
    volume_projects = volume_projects or {}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mock = bin_dir / "docker"

    volume_names = "\\n".join(volume_projects)
    volume_cases = "\n".join(
        f'    "{name}") printf \'%s\\n\' "{project}" ;;' for name, project in volume_projects.items()
    )
    container_response = f"printf '%s\\n' \"{container_project}\"" if container_project else "exit 1"
    volume_output = f"{volume_names}\n" if volume_names else ""
    mock.write_text(
        f"""#!/bin/bash
if [[ "$1" == "compose" ]]; then
    printf '%s\n' "$*" >> "$MOCK_DOCKER_LOG"
elif [[ "$1 $2" == "inspect morphik-postgres" ]]; then
    {container_response}
elif [[ "$1 $2" == "volume ls" ]]; then
    printf '%b' "{volume_output}"
elif [[ "$1 $2" == "volume inspect" ]]; then
    case "$3" in
{volume_cases}
        *) exit 1 ;;
    esac
else
    exit 1
fi
"""
    )
    mock.chmod(0o755)
    return bin_dir


def _resolve_project(tmp_path: Path, **mock_options) -> subprocess.CompletedProcess[str]:
    bin_dir = _write_mock_docker(tmp_path, **mock_options)
    env = os.environ.copy()
    env.pop("COMPOSE_PROJECT_NAME", None)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    helper = REPO_ROOT / "morphik-compose-project.sh"
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{helper}"; morphik_compose_resolve_existing_project || exit $?; printf "%s" "$COMPOSE_PROJECT_NAME"',
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generated_stop_scripts_preserve_named_volumes():
    shell_installer = _read("install_docker.sh")
    shell_stop = _generated_shell_script(shell_installer, "stop-morphik.sh")
    powershell_installer = _read("install_docker.ps1")

    assert "down --remove-orphans" in shell_stop
    assert "down --volumes" not in shell_stop
    assert "@('down','--remove-orphans')" in powershell_installer
    assert "@('down','--volumes'" not in powershell_installer
    assert "data volumes were preserved" in shell_stop
    assert "data volumes were preserved" in powershell_installer


def test_generated_scripts_load_project_recovery_helper():
    installer = _read("install_docker.sh")
    shell_start = _generated_shell_script(installer, "start-morphik.sh")
    shell_stop = _generated_shell_script(installer, "stop-morphik.sh")

    for script in (shell_start, shell_stop):
        assert 'source "$SCRIPT_DIR/morphik-compose-project.sh"' in script
        assert "morphik_compose_resolve_existing_project" in script

    powershell_installer = _read("install_docker.ps1")
    assert powershell_installer.count("Resolve-MorphikComposeProject") >= 3
    assert "morphik-compose-project.ps1" in powershell_installer


def test_generated_shell_scripts_parse():
    installer = _read("install_docker.sh")

    for filename in ("start-morphik.sh", "stop-morphik.sh"):
        result = subprocess.run(
            ["bash", "-n"],
            input=_generated_shell_script(installer, filename),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_generated_stop_script_runs_from_its_directory_without_deleting_volumes(tmp_path: Path):
    installer = _read("install_docker.sh")
    deployment_dir = tmp_path / "deployment"
    deployment_dir.mkdir()
    stop_script = deployment_dir / "stop-morphik.sh"
    stop_script.write_text(_generated_shell_script(installer, "stop-morphik.sh"))
    stop_script.chmod(0o755)
    (deployment_dir / "morphik-compose-project.sh").write_text(_read("morphik-compose-project.sh"))
    (deployment_dir / "docker-compose.run.yml").touch()
    (deployment_dir / ".env").write_text("COMPOSE_PROJECT_NAME=existing-install\n")

    bin_dir = _write_mock_docker(tmp_path)
    docker_log = tmp_path / "docker.log"
    env = os.environ.copy()
    env.pop("COMPOSE_PROJECT_NAME", None)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MOCK_DOCKER_LOG"] = str(docker_log)

    result = subprocess.run(
        [str(stop_script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = docker_log.read_text().strip()
    assert command == "compose -f docker-compose.run.yml down --remove-orphans"
    assert "--volumes" not in command


def test_postgres_storage_defaults_to_named_volume_and_allows_host_path():
    for compose_file in ("docker-compose.yml", "docker-compose.run.yml"):
        compose = _read(compose_file)
        assert '"${MORPHIK_POSTGRES_DATA_PATH:-postgres_data}:/var/lib/postgresql/data"' in compose

    docs = _read("DOCKER.md")
    assert "Do not add `--volumes` to a normal shutdown" in docs
    assert "MORPHIK_POSTGRES_DATA_PATH=./postgres-data" in docs


def test_project_recovery_prefers_existing_postgres_container(tmp_path: Path):
    result = _resolve_project(tmp_path, container_project="original-install")

    assert result.returncode == 0
    assert result.stdout.endswith("original-install")


def test_project_recovery_uses_only_existing_postgres_volume(tmp_path: Path):
    result = _resolve_project(tmp_path, volume_projects={"old_postgres_data": "original-install"})

    assert result.returncode == 0
    assert result.stdout.endswith("original-install")


def test_project_recovery_fails_closed_when_volumes_are_ambiguous(tmp_path: Path):
    result = _resolve_project(
        tmp_path,
        volume_projects={"first_postgres_data": "first-install", "second_postgres_data": "second-install"},
    )

    assert result.returncode != 0
    assert "Multiple Morphik Postgres volumes were found" in result.stderr
    assert not result.stdout.endswith("first-install")
    assert not result.stdout.endswith("second-install")
