"""End-to-end request and response test for the iQor retrieval verifier."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_iqor_retrieval.sh"


def _result(work_item_id: int, score: float, chunk_number: int = 0) -> dict:
    return {
        "content": f"QA backlog item {work_item_id}",
        "score": score,
        "document_id": f"doc-{work_item_id}",
        "chunk_number": chunk_number,
        "metadata": {
            "project": "QA",
            "work_item_type": "Product Backlog Item",
            "work_item_id": work_item_id,
        },
        "content_type": "text/plain",
        "filename": f"{work_item_id}.txt",
        "download_url": None,
        "is_padding": False,
    }


@pytest.mark.parametrize("explicit_response_path", [True, False], ids=["explicit-path", "temporary-path"])
def test_iqor_verifier_sends_contract_and_returns_five_distinct_items(tmp_path, explicit_response_path):
    if shutil.which("curl") is None or shutil.which("jq") is None:
        pytest.skip("curl and jq are required")

    captured = {}
    response = [
        _result(47501, 0.92),
        _result(47501, 0.81, chunk_number=1),
        _result(47502, 0.88),
        _result(47503, 0.84),
        _result(47504, 0.80),
        _result(47505, 0.76),
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    env = os.environ.copy()
    env.update(
        {
            "IQOR_QUERY": "find related QA backlog work",
            "MORPHIK_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "TMPDIR": str(tmp_path),
        }
    )
    response_path = tmp_path / "response.json"
    if explicit_response_path:
        env["IQOR_RESPONSE_FILE"] = str(response_path)
    else:
        env.pop("IQOR_RESPONSE_FILE", None)
    try:
        completed = subprocess.run(
            [str(VERIFY_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert captured["path"] == "/retrieve/chunks"
    assert captured["body"] == {
        "query": "find related QA backlog work",
        "k": 50,
        "min_score": 0,
        "use_colpali": False,
        "output_format": "text",
        "filters": {
            "$and": [
                {"project": {"$eq": "QA"}},
                {"work_item_type": {"$eq": "Product Backlog Item"}},
                {"work_item_id": {"$ne": 47490}},
            ]
        },
    }

    if not explicit_response_path:
        saved_line = next(line for line in completed.stderr.splitlines() if line.startswith("Raw response saved to "))
        response_path = Path(saved_line.removeprefix("Raw response saved to ").removesuffix("."))
        assert response_path.parent == tmp_path
        assert response_path.name.startswith("iqor-retrieval-response.")

    selected = json.loads(completed.stdout)
    assert len(selected) == 5
    assert len({item["metadata"]["work_item_id"] for item in selected}) == 5
    assert selected[0]["score"] == 0.92
    assert selected[0]["source_id"] == "doc-47501:0"
    assert json.loads(response_path.read_text()) == response


@pytest.mark.parametrize("invalid_result", ["missing-work-item-id", "below-min-score"])
def test_iqor_verifier_rejects_invalid_distinct_result(tmp_path, invalid_result):
    if shutil.which("curl") is None or shutil.which("jq") is None:
        pytest.skip("curl and jq are required")

    response = [_result(47501, 0.92), _result(47502, 0.88), _result(47503, 0.84), _result(47504, 0.80)]
    invalid = _result(47505, 0.76)
    if invalid_result == "missing-work-item-id":
        del invalid["metadata"]["work_item_id"]
    else:
        invalid["score"] = -0.01
    response.append(invalid)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    env = os.environ.copy()
    env.update(
        {
            "IQOR_QUERY": "find related QA backlog work",
            "MORPHIK_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "IQOR_RESPONSE_FILE": str(tmp_path / "response.json"),
        }
    )
    try:
        completed = subprocess.run(
            [str(VERIFY_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 1
    assert "violated the metadata, exclusion, score, or source-ID contract" in completed.stderr
