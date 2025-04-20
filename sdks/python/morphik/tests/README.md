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

```bash
# Using default localhost:8000 URL
pytest test_sync.py test_async.py -v

# Tests connect to localhost:8000 by default
# No need to specify a URL unless you want to test against a different server

# With a custom server URL (optional)
MORPHIK_TEST_URL=http://custom-url:8000 pytest test_sync.py -v
```

### Example Usage Script
```bash
# Run synchronous example
python example_usage.py

# Run asynchronous example
python example_usage.py --async
```

## Environment Variables

- `MORPHIK_TEST_URL` - The URL of the Morphik server to use for tests (default: http://localhost:8000)
- `SKIP_LIVE_TESTS` - Set to "1" to skip tests that require a running server
