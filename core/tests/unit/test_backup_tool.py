"""Unit tests for the manifest, compatibility, and retention logic in morphik-backup.sh.

The script keeps these rules in internal subcommands so they can run here without Docker.
They need bash and Perl's JSON::PP, which ship with the PostgreSQL image the script runs in.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "morphik-backup.sh"
SHA = "a" * 64


def _has_perl_json() -> bool:
    if shutil.which("bash") is None or shutil.which("perl") is None:
        return False
    return subprocess.run(["perl", "-MJSON::PP", "-e", "1"], capture_output=True, check=False).returncode == 0


pytestmark = pytest.mark.skipif(not _has_perl_json(), reason="bash and Perl JSON::PP are required")


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, MORPHIK_BACKUP_TOOL="1")
    result = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=60, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"{args} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def _db_stats(**overrides) -> dict:
    stats = {
        "postgres_version": "16.15",
        "extensions": {"plpgsql": "1.0", "vector": "0.8.0"},
        "schema_fingerprint": "087e266ef773666aafa1e061e324ef1e",
        "vector_dimensions": 1536,
        "documents_by_status": {"completed": 5, "processing": 1},
        "tables": {
            "documents": {"rows": 6, "checksum": "3139214768253929761"},
            "vector_embeddings": {"rows": 42, "checksum": "3449253678630010817"},
            "multi_vector_embeddings": {"rows": 7, "checksum": "11"},
            "folders": {"rows": 2, "checksum": "915408572173076530"},
        },
    }
    stats.update(overrides)
    return stats


def _facts(storage_provider: str = "local", model_name: str = "text-embedding-3-small") -> str:
    lines = [
        "s\tcreated_at\t2026-09-24T12:00:00Z",
        "s\tname\tmorphik-20260924T120000Z.backup",
        "s\tcreated_by.mode\tmanual",
        "b\tincludes_env\tfalse",
        "s\tcore.image\tghcr.io/morphik-org/morphik-core:latest",
        "z\tcore.image_id\t",
        'j\tcore.repo_digests\t["ghcr.io/morphik-org/morphik-core@sha256:abc"]',
        "s\tembedding.model\topenai_embedding",
        f"s\tembedding.model_name\t{model_name}",
        "n\tembedding.dimensions\t1536",
        "s\tvector_store.provider\tpgvector",
        "s\tmultivector_store.provider\tpostgres",
        f"s\tstorage.provider\t{storage_provider}",
        "p\tparts\tdatabase.dump\t" + SHA + "\t1000",
        "p\tparts\tconfig/morphik.toml\t" + SHA + "\t20",
    ]
    if storage_provider == "local":
        lines += [
            "p\tparts\tstorage.tar\t" + SHA + "\t2048",
            "n\tcounts.storage_files\t3",
            "n\tcounts.storage_bytes\t1234",
            "b\tstorage.included\ttrue",
        ]
    else:
        lines.append("b\tstorage.included\tfalse")
    return "\n".join(lines) + "\n"


def _build_manifest(tmp_path: Path, **kwargs) -> dict:
    facts = tmp_path / "facts.tsv"
    stats = tmp_path / "database.json"
    facts.write_text(_facts(**kwargs), encoding="utf-8")
    stats.write_text(json.dumps(_db_stats()), encoding="utf-8")
    return json.loads(_run("_manifest", str(facts), str(stats)).stdout)


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _toml(
    tmp_path: Path,
    *,
    model: str = "openai_embedding",
    model_name: str = "text-embedding-3-small",
    dimensions: str = "1536",
    storage: str = "local",
) -> Path:
    path = tmp_path / "morphik.toml"
    path.write_text(
        f"""
[registered_models]
{model} = {{ model_name = "{model_name}", api_base = "http://embeddings:8000" }}  # comment

[embedding]
model = "{model}"  # Reference to registered model
dimensions = {dimensions}

[vector_store]
provider = "pgvector"

[multivector_store]
provider = "postgres"  # "morphik" for Turbopuffer

[storage]
provider = "{storage}"
storage_path = "./storage"

# [storage]
# provider = "aws-s3"
""",
        encoding="utf-8",
    )
    return path


def test_manifest_records_configuration_counts_and_parts(tmp_path):
    manifest = _build_manifest(tmp_path)

    assert manifest["format"] == "morphik-core-backup"
    assert manifest["format_version"] == 1
    assert manifest["created_at"] == "2026-09-24T12:00:00Z"
    assert manifest["includes_env"] is False
    assert manifest["core"]["image_id"] is None
    assert manifest["core"]["repo_digests"] == ["ghcr.io/morphik-org/morphik-core@sha256:abc"]
    assert manifest["embedding"] == {
        "model": "openai_embedding",
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
    }
    assert manifest["counts"] == {
        "documents": 6,
        "documents_by_status": {"completed": 5, "processing": 1},
        "chunks": 42,
        "multivector_chunks": 7,
        "folders": 2,
        "storage_files": 3,
        "storage_bytes": 1234,
    }
    assert manifest["database"]["vector_dimensions"] == 1536
    assert manifest["database"]["schema"]["migration_tracking"] == "none"
    assert manifest["database"]["schema"]["fingerprint"] == "087e266ef773666aafa1e061e324ef1e"
    assert [part["name"] for part in manifest["parts"]] == ["config/morphik.toml", "database.dump", "storage.tar"]
    assert manifest["storage"]["included"] is True

    _run("_json", "check", str(_write_manifest(tmp_path, manifest)))


def test_manifest_for_s3_storage_has_no_storage_part(tmp_path):
    manifest = _build_manifest(tmp_path, storage_provider="aws-s3")

    assert manifest["storage"] == {"provider": "aws-s3", "included": False}
    assert "storage.tar" not in [part["name"] for part in manifest["parts"]]
    _run("_json", "check", str(_write_manifest(tmp_path, manifest)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(format="something-else"), "not a Morphik Core backup"),
        (lambda m: m.update(format_version=2), "newer than this script supports"),
        (lambda m: m.update(parts=[p for p in m["parts"] if p["name"] != "database.dump"]), "database.dump is missing"),
        (lambda m: m["parts"][0].update(sha256="nope"), "has no sha256"),
        (lambda m: m["parts"][0].update(name="../../etc/passwd"), "unsafe name"),
        (lambda m: m.update(parts=[p for p in m["parts"] if p["name"] != "storage.tar"]), "storage.tar is missing"),
        (lambda m: m.pop("created_at"), "created_at is missing"),
    ],
)
def test_manifest_check_rejects_invalid_manifests(tmp_path, mutate, message):
    manifest = _build_manifest(tmp_path)
    mutate(manifest)

    result = _run("_json", "check", str(_write_manifest(tmp_path, manifest)), check=False)

    assert result.returncode == 1
    assert message in result.stderr


def test_config_facts_read_the_shipped_docker_config():
    facts = _run("_config-facts", str(REPO_ROOT / "morphik.docker.toml")).stdout.splitlines()

    assert "s\tembedding.model\topenai_embedding" in facts
    assert "s\tembedding.model_name\ttext-embedding-3-small" in facts
    assert "n\tembedding.dimensions\t1536" in facts
    assert "s\tvector_store.provider\tpgvector" in facts
    assert "s\tmultivector_store.provider\tpostgres" in facts
    # The commented-out aws-s3 [storage] block must not win.
    assert "s\tstorage.provider\tlocal" in facts


def test_compat_accepts_a_matching_target(tmp_path):
    manifest = _write_manifest(tmp_path, _build_manifest(tmp_path))

    result = _run("_compat", str(manifest), str(_toml(tmp_path)))

    assert "INCOMPATIBLE" not in result.stdout


@pytest.mark.parametrize(
    ("toml_kwargs", "message"),
    [
        ({"dimensions": "768"}, "embedding dimensions differ: backup has 1536, target morphik.toml has 768"),
        ({"model_name": "text-embedding-3-large"}, "embedding model differs"),
        ({"storage": "aws-s3"}, "storage provider differs"),
        ({"dimensions": ""}, "no [embedding] dimensions"),
    ],
)
def test_compat_refuses_mismatched_targets(tmp_path, toml_kwargs, message):
    manifest = _write_manifest(tmp_path, _build_manifest(tmp_path))

    result = _run("_compat", str(manifest), str(_toml(tmp_path, **toml_kwargs)), check=False)

    assert result.returncode == 1
    assert message in result.stdout


def test_compat_allows_a_renamed_key_for_the_same_model(tmp_path):
    manifest = _write_manifest(tmp_path, _build_manifest(tmp_path))

    result = _run("_compat", str(manifest), str(_toml(tmp_path, model="my_embedding")))

    assert "INCOMPATIBLE" not in result.stdout
    assert "key changed from 'openai_embedding' to 'my_embedding'" in result.stdout


def test_compat_trusts_the_database_vector_width_over_the_saved_config(tmp_path):
    # The saved morphik.toml said 1536, but the vector column in the dump is 768 wide.
    manifest = _build_manifest(tmp_path)
    manifest["database"]["vector_dimensions"] = 768
    path = _write_manifest(tmp_path, manifest)

    refused = _run("_compat", str(path), str(_toml(tmp_path, dimensions="1536")), check=False)
    accepted = _run("_compat", str(path), str(_toml(tmp_path, dimensions="768")), check=False)

    assert refused.returncode == 1
    assert "backup has 768" in refused.stdout
    assert accepted.returncode == 0


def test_compare_detects_changed_rows_and_checksums(tmp_path):
    manifest = _write_manifest(tmp_path, _build_manifest(tmp_path))
    same = tmp_path / "same.json"
    same.write_text(json.dumps(_db_stats()), encoding="utf-8")
    changed = _db_stats()
    changed["tables"]["vector_embeddings"] = {"rows": 42, "checksum": "1"}
    changed["tables"]["documents"] = {"rows": 5, "checksum": "3139214768253929761"}
    changed["tables"]["chunk_v2"] = {"rows": 0, "checksum": "0"}
    changed["documents_by_status"] = {"completed": 5}
    different = tmp_path / "different.json"
    different.write_text(json.dumps(changed), encoding="utf-8")

    _run("_json", "compare", str(manifest), str(same))
    result = _run("_json", "compare", str(manifest), str(different), check=False)

    assert result.returncode == 1
    assert "table vector_embeddings content checksum differs" in result.stdout
    assert "table documents has 5 rows, expected 6" in result.stdout
    assert "unexpected table chunk_v2" in result.stdout
    assert "document status counts differ" in result.stdout


def test_missing_files_understands_local_storage_bucket_forms(tmp_path):
    refs = tmp_path / "refs.tsv"
    refs.write_text(
        "doc-a\tstorage\tingest_uploads/doc-a/a.txt\n"
        "doc-b\t\tdoc-b/spec ü.pdf\n"
        "doc-c\t/app/storage\t/app/storage/ingest_uploads/doc-c/c.txt\n"
        "doc-d\tstorage\tingest_uploads/doc-d/missing.txt\n",
        encoding="utf-8",
    )
    listing = tmp_path / "listing"
    listing.write_text(
        "./\n./ingest_uploads/\n./ingest_uploads/doc-a/a.txt\n./doc-b/spec ü.pdf\n./ingest_uploads/doc-c/c.txt\n",
        encoding="utf-8",
    )

    result = _run("_json", "missing-files", str(refs), str(listing), "./storage", check=False)

    assert result.returncode == 1
    assert result.stdout.splitlines() == ["doc-d\tingest_uploads/doc-d/missing.txt"]


def test_retention_deletes_only_the_oldest_scheduled_backups(tmp_path):
    names = [
        "morphik-20260901T000000Z-auto.backup",
        "morphik-20260902T000000Z-auto.backup",
        "morphik-20260903T000000Z-auto.backup",
        "morphik-20260904T000000Z-auto.backup",
        "morphik-20260801T000000Z.backup",
        "morphik-20260802T000000Z-pre-restore.backup",
        "notes.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("x", encoding="utf-8")

    _run("_retention", str(tmp_path), "2")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "morphik-20260801T000000Z.backup",
        "morphik-20260802T000000Z-pre-restore.backup",
        "morphik-20260903T000000Z-auto.backup",
        "morphik-20260904T000000Z-auto.backup",
        "notes.txt",
    ]


@pytest.mark.parametrize("keep", ["0", "-1", "two", ""])
def test_retention_rejects_invalid_keep(tmp_path, keep):
    (tmp_path / "morphik-20260901T000000Z-auto.backup").write_text("x", encoding="utf-8")

    result = _run("_retention", str(tmp_path), keep, check=False)

    assert result.returncode != 0
    assert (tmp_path / "morphik-20260901T000000Z-auto.backup").exists()


def test_requeue_body_lists_documents_for_the_requeue_endpoint(tmp_path):
    ids = tmp_path / "ids"
    ids.write_text("doc-1\ndoc-2\n", encoding="utf-8")

    body = json.loads(_run("_json", "requeue-body", str(ids)).stdout)

    assert body == {"jobs": [{"external_id": "doc-1"}, {"external_id": "doc-2"}]}
