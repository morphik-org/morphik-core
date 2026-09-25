"""Docker-level backup and restore test for the production Compose deployment."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCRIPT = REPO_ROOT / "scripts" / "test_backup_restore.sh"


@pytest.mark.integration
def test_backup_survives_volume_deletion_and_restores_identically():
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")

    daemon = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if daemon.returncode != 0:
        pytest.skip("Docker daemon is not running")

    subprocess.run([str(TEST_SCRIPT)], cwd=REPO_ROOT, check=True, timeout=900)
