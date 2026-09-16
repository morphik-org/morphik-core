import httpx
import pytest


def _block_request(*args, **kwargs):
    """Fail fast if any unit test attempts an outbound HTTP request."""
    raise AssertionError("Network call attempted during unit test")


# Patch in conftest at import time so that imports at collection-time (LiteLLM) cannot hit the network.
_mp = pytest.MonkeyPatch()
_mp.setattr(httpx.Client, "request", _block_request, raising=True)
_mp.setattr(httpx.AsyncClient, "request", _block_request, raising=True)


def pytest_unconfigure(config):
    """Restore patched httpx methods when pytest shuts down."""
    _mp.undo()
