# Morphik SDK Tests

This directory contains tests and example code for the Morphik SDK.

## Test Types

- `test_sync.py` - Tests for the synchronous client
- `test_async.py` - Tests for the asynchronous client

### Test Data
- `test_docs/` - Sample text files for testing document ingestion

### Example Code
- `example_usage.py` - Example script demonstrating basic usage of the SDK

## Running Tests

These live tests mutate server state and may leave residual folders or scoped
artifacts behind. Run them only against a local or disposable test server or
tenant, never against production or shared data.

```bash
# Using default localhost:8000 server URI
pytest test_sync.py test_async.py -v

# Tests connect to localhost:8000 by default
# No need to specify a URI unless you want to test against a different server

# With a custom disposable test server URI (optional; applies to sync and async tests)
# Use direct http(s) endpoints inline; source credential-bearing morphik:// URIs
# from a local secret store or uncommitted env file instead of shell history or CI logs
MORPHIK_TEST_URI=http://custom-url:8000 pytest test_sync.py test_async.py -v
```

### Example Usage Script

The example script creates server data and only partially cleans it up. Run it
only against the same local or disposable test environment.

```bash
# Run synchronous example from this directory
PYTHONPATH=../.. uv run python example_usage.py

# Run asynchronous example from this directory
PYTHONPATH=../.. uv run python example_usage.py --run-async
```

## Environment Variables

- `MORPHIK_TEST_URI` - The Morphik server URI to use for tests; accepts direct `http(s)://` endpoints or authenticated `morphik://` URIs (default: http://localhost:8000)
- `SKIP_LIVE_TESTS` - Set to "1" to skip tests that require a running server
