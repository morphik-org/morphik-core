"""Regression tests for Docker installer auth-secret environment wiring."""

from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest
from fastapi import HTTPException

from core.local_uri import LOCAL_URI_PASSWORD_DISABLED_DETAIL, require_local_uri_password_configured


ROOT = Path(__file__).resolve().parents[3]


def _bash_function(source: str, name: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{.*?^\}}", source)
    assert match is not None
    return match.group(0)


def _powershell_executable() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell runtime is required for PowerShell installer parser tests")
    assert executable is not None
    return executable


def test_hosted_docker_installers_generate_session_secret():
    bash_installer = (ROOT / "install_docker.sh").read_text()
    powershell_installer = (ROOT / "install_docker.ps1").read_text()

    assert 'generate_auth_secret "morphik-jwt"' in bash_installer
    assert 'generate_auth_secret "morphik-session"' in bash_installer
    assert "openssl rand -hex 32" in bash_installer
    assert "Could not generate a secure auth secret" in bash_installer
    assert "$jwt = \"morphik-jwt-$(New-RandomHex 32)\"" in powershell_installer
    assert "$session = \"morphik-session-$(New-RandomHex 32)\"" in powershell_installer
    assert "\"SESSION_SECRET_KEY=$session\"" in powershell_installer


def test_compose_files_do_not_fallback_to_placeholder_auth_secrets():
    def service_block(compose_text: str, service: str) -> str:
        match = re.search(rf"(?ms)^  {service}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", compose_text)
        assert match is not None
        return match.group("body")

    for compose_file in ("docker-compose.yml", "docker-compose.run.yml"):
        compose_text = (ROOT / compose_file).read_text()
        assert "JWT_SECRET_KEY=${JWT_SECRET_KEY:-your-secret-key-here}" not in compose_text
        assert "JWT_SECRET_KEY=${JWT_SECRET_KEY:-}" in compose_text
        assert "SESSION_SECRET_KEY=${SESSION_SECRET_KEY:-}" in compose_text
        assert "LOCAL_URI_PASSWORD=${LOCAL_URI_PASSWORD:-}" in compose_text

        morphik_block = service_block(compose_text, "morphik")
        worker_block = service_block(compose_text, "worker")

        for block in (morphik_block, worker_block):
            assert "env_file:" in block
            assert "      - path: .env" in block
            assert "        required: false" in block
            assert "        format: raw" not in block
            assert "      - JWT_SECRET_KEY=${JWT_SECRET_KEY:-}" in block
            assert "      - SESSION_SECRET_KEY=${SESSION_SECRET_KEY:-}" in block
            assert "      - LOCAL_URI_PASSWORD=${LOCAL_URI_PASSWORD:-}" in block


def test_installers_require_compose_version_that_supports_optional_env_file():
    bash_installer = (ROOT / "install_docker.sh").read_text()
    powershell_installer = (ROOT / "install_docker.ps1").read_text()
    docker_docs = (ROOT / "DOCKER.md").read_text()

    assert 'MIN_COMPOSE_VERSION="2.24.0"' in bash_installer
    assert "compose_version_at_least" in bash_installer
    assert "optional env_file support" in bash_installer
    assert "$script:MinComposeVersion = [Version]'2.24.0'" in powershell_installer
    assert "optional env_file support" in powershell_installer
    assert "Docker and Docker Compose 2.24.0 or newer" in docker_docs


def test_bash_installer_compose_version_check_handles_boundaries():
    bash_installer = (ROOT / "install_docker.sh").read_text()
    function_def = _bash_function(bash_installer, "compose_version_at_least")
    command = f"""
    set -e
    {function_def}
    compose_version_at_least 2.24.0 2.24.0
    compose_version_at_least v2.24.1 2.24.0
    compose_version_at_least 3.0.0 2.24.0
    ! compose_version_at_least 2.23.9 2.24.0
    ! compose_version_at_least invalid 2.24.0
    """

    subprocess.run(["bash", "-c", command], check=True, text=True)


def test_bash_installer_stops_before_writing_env_when_secret_generation_fails(tmp_path):
    bash_installer = (ROOT / "install_docker.sh").read_text()
    function_defs = "\n\n".join(
        _bash_function(bash_installer, name)
        for name in ("print_error", "protect_env_file", "generate_auth_secret")
    )
    command = f"""
    set -e
    PATH=/no-such-command
    {function_defs}
    jwt_secret="$(generate_auth_secret "morphik-jwt")"
    cat > .env <<EOF
JWT_SECRET_KEY=${{jwt_secret}}
EOF
    """

    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "Could not generate a secure auth secret" in result.stderr
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.run.yml"])
def test_compose_files_load_auth_secrets_from_env_file(tmp_path, compose_file):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is required for env-file rendering test")

    source = (ROOT / compose_file).read_text()
    source = re.sub(r"(?m)^\\s*- \\./[^\\n]+\\n", "", source)
    (tmp_path / compose_file).write_text(source)
    (tmp_path / "morphik.toml").write_text("")
    env_values = {
        "JWT_SECRET_KEY": "jwt-secret-0123456789abcdef0123456789",
        "SESSION_SECRET_KEY": "session-secret-0123456789abcdef0123456789",
        "LOCAL_URI_PASSWORD": "local-uri-password-0123456789abcdef0123456789",
    }
    (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in env_values.items()) + "\n")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    for key, value in env_values.items():
        assert f"{key}: {value}" in result.stdout


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.run.yml"])
def test_compose_files_allow_shell_secret_injection_without_env_file(tmp_path, monkeypatch, compose_file):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is required for shell env rendering test")

    source = (ROOT / compose_file).read_text()
    source = re.sub(r"(?m)^\\s*- \\./[^\\n]+\\n", "", source)
    (tmp_path / compose_file).write_text(source)
    (tmp_path / "morphik.toml").write_text("")

    env = {
        "JWT_SECRET_KEY": "jwt-secret-0123456789abcdef0123456789",
        "SESSION_SECRET_KEY": "session-secret-0123456789abcdef0123456789",
        "LOCAL_URI_PASSWORD": "local-uri-password-0123456789abcdef0123456789",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    for key, value in env.items():
        assert f"{key}: {value}" in result.stdout


def test_installers_validate_local_uri_password_before_writing_env():
    bash_installer = (ROOT / "install_docker.sh").read_text()
    powershell_installer = (ROOT / "install_docker.ps1").read_text()

    assert 'validate_local_uri_password "$local_uri_password"' in bash_installer
    assert 'local_uri_password="$(normalize_auth_secret "$local_uri_password")"' in bash_installer
    assert "LOCAL_URI_PASSWORD must be at least ${AUTH_SECRET_MIN_LENGTH} characters" in bash_installer
    assert "LOCAL_URI_PASSWORD must not use an example or placeholder value" in bash_installer
    assert "/local/generate_uri endpoint" in bash_installer
    assert "/generate_local_uri endpoint" not in bash_installer
    assert "read -r -s -p" in bash_installer
    assert bash_installer.index('validate_local_uri_password "$local_uri_password"') < bash_installer.index(
        'set_env_value "LOCAL_URI_PASSWORD" "$local_uri_password"'
    )

    assert "$password = Assert-LocalUriPassword -Value $password" in powershell_installer
    assert "Read-Host -AsSecureString" in powershell_installer
    assert "function ConvertFrom-SecureInput" in powershell_installer
    assert "$normalized = Normalize-AuthSecret -Value $Value" in powershell_installer
    assert "LOCAL_URI_PASSWORD must be at least $script:AuthSecretMinLength characters" in powershell_installer
    assert "LOCAL_URI_PASSWORD must not use an example or placeholder value" in powershell_installer
    assert powershell_installer.index("$password = Assert-LocalUriPassword -Value $password") < powershell_installer.index(
        'Set-EnvValue -Key "LOCAL_URI_PASSWORD" -Value $password'
    )
    assert powershell_installer.index("$password = Normalize-AuthSecret -Value $password") < powershell_installer.index(
        "if ([string]::IsNullOrWhiteSpace($password))"
    )


def test_installers_restrict_env_file_permissions():
    bash_installer = (ROOT / "install_docker.sh").read_text()
    powershell_installer = (ROOT / "install_docker.ps1").read_text()

    assert "umask 077" in bash_installer
    assert "chmod 600 .env" in bash_installer
    assert "protect_env_file" in bash_installer
    assert bash_installer.index("umask 077") < bash_installer.index("cat > .env <<EOF")
    env_create_index = bash_installer.index("cat > .env <<EOF")
    assert env_create_index < bash_installer.index("protect_env_file", env_create_index)

    assert "function Protect-EnvFile" in powershell_installer
    assert "$acl.SetAccessRuleProtection($true, $false)" in powershell_installer
    assert "RemoveAccessRuleSpecific" in powershell_installer
    assert "System.Security.AccessControl.FileSystemAccessRule" in powershell_installer
    assert "Protect-EnvFile" in powershell_installer


@pytest.mark.parametrize(
    "local_uri_password",
    [
        "short-local-uri-password",
        "your-local-uri-password-here",
        "<replace-with-local-uri-password>",
        '"123456789012345678901234567890"',
    ],
)
def test_bash_installer_stops_before_writing_invalid_local_uri_password(tmp_path, local_uri_password):
    bash_installer = (ROOT / "install_docker.sh").read_text()
    function_defs = "\n\n".join(
        _bash_function(bash_installer, name)
        for name in ("print_error", "protect_env_file", "set_env_value", "normalize_auth_secret", "validate_local_uri_password")
    )
    command = f"""
    set -e
    AUTH_SECRET_MIN_LENGTH=32
    {function_defs}
    local_uri_password="$(normalize_auth_secret {shlex.quote(local_uri_password)})"
    validate_local_uri_password "$local_uri_password" || exit 1
    set_env_value "LOCAL_URI_PASSWORD" "$local_uri_password"
    """

    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "LOCAL_URI_PASSWORD" in result.stderr
    assert not (tmp_path / ".env").exists()


def test_local_uri_endpoint_has_explicit_disabled_response():
    api_source = (ROOT / "core/api.py").read_text()

    assert "password_token: Optional[str] = Form(None)" in api_source
    assert "require_local_uri_password_configured(settings.LOCAL_URI_PASSWORD)" in api_source

    with pytest.raises(HTTPException) as exc_info:
        require_local_uri_password_configured(None)

    exception = exc_info.value
    assert isinstance(exception, HTTPException)
    assert exception.status_code == 503
    assert exception.detail == LOCAL_URI_PASSWORD_DISABLED_DETAIL


def test_docker_docs_describe_session_secret_requirement():
    docker_docs = (ROOT / "DOCKER.md").read_text()

    assert "Create a `.env` file for Docker secrets" in docker_docs
    assert "docker compose up --build" in docker_docs
    assert docker_docs.index("Create a `.env` file for Docker secrets") < docker_docs.index("docker compose up --build")
    assert "umask 077" in docker_docs
    assert "umask 077" in docker_docs[: docker_docs.index("cat > .env <<EOF")]
    assert "JWT_SECRET_KEY=$(openssl rand -hex 32)" in docker_docs
    assert "SESSION_SECRET_KEY=$(openssl rand -hex 32)" in docker_docs
    assert "SESSION_SECRET_KEY=<32+-character-random-hex-secret>" in docker_docs
    assert "JWT_SECRET_KEY` and `SESSION_SECRET_KEY` must be non-empty" in docker_docs
    assert "`LOCAL_URI_PASSWORD` is unset or blank, `/local/generate_uri` is disabled" in docker_docs
    assert "authenticated Docker deployments must verify that `JWT_SECRET_KEY` and `SESSION_SECRET_KEY`" in docker_docs
    assert "through `.env` or shell-exported environment variables" in docker_docs
    assert "use hex values such as `openssl rand -hex 32`" in docker_docs
    assert "If startup fails with `LOCAL_URI_PASSWORD` validation errors" in docker_docs
    assert "returns HTTP `503` with `LOCAL_URI_PASSWORD is not configured; /local/generate_uri is disabled`" in docker_docs
    assert "If startup fails with `JWT_SECRET_KEY` or `SESSION_SECRET_KEY` validation errors" in docker_docs


@pytest.mark.parametrize("script_name", ["install_docker.ps1"])
def test_powershell_installer_parses_when_runtime_available(script_name):
    powershell = _powershell_executable()
    script_path = str(ROOT / script_name).replace("'", "''")
    command = f"""
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref] $tokens, [ref] $errors) | Out-Null
    if ($errors.Count -gt 0) {{
      $errors | ForEach-Object {{ Write-Error $_.Message }}
      exit 1
    }}
    """

    subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", command], check=True, text=True)
