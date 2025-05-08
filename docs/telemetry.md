# Morphik Telemetry

Morphik includes an anonymous telemetry system to help us understand how the library is being used and to improve its functionality. We take privacy very seriously and ensure that no personally identifiable information (PII) is ever collected.

## What We Collect

The following anonymous data is collected:

- Installation ID (a randomly generated identifier, hashed from machine ID)
- Operation types (e.g., document ingestion, queries, retrievals)
- Operation durations
- Token usage statistics
- Error rates and types
- Basic metadata about operations (excluding any PII)

We explicitly DO NOT collect:

- File contents or queries
- API keys or credentials
- Personal information
- IP addresses or location data
- Any metadata fields containing sensitive information

## How to Opt Out

Telemetry is enabled by default but can be disabled by setting the environment variable:

```bash
export DATABRIDGE_TELEMETRY_ENABLED=0
```

Or in your Python code:

```python
import os
os.environ["DATABRIDGE_TELEMETRY_ENABLED"] = "0"
```

## Data Storage and Retention

All telemetry data is:
- Stored securely
- Automatically anonymized before transmission
- Used only for improving Morphik
- Never shared with third parties
- Retained for a maximum of 90 days

## Technical Details

The telemetry system uses OpenTelemetry to collect metrics and traces. In development mode, data is stored locally in `logs/telemetry/`. In production, data is sent to our secure collector endpoint.

You can inspect the telemetry data being collected by looking at the local log files in development mode:
- `logs/telemetry/traces.log`
- `logs/telemetry/metrics.log`

## Why We Collect Telemetry

This data helps us:
1. Understand how Morphik is used in real-world scenarios
2. Identify performance bottlenecks
3. Prioritize features and improvements
4. Fix bugs faster
5. Make data-driven decisions about the project's direction

## Questions or Concerns

If you have any questions or concerns about our telemetry collection, please:
1. Open an issue on our GitHub repository
2. Email us at founders@morphik.ai
3. Review our telemetry implementation in `core/services/telemetry.py`
