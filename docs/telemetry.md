# Morphik Telemetry

Morphik logs minimal operational metadata (operation name, status, duration, token counts) to `logs/telemetry/` so we can keep deployments healthy, then periodically uploads those JSONL files to `https://logs.morphik.ai` to avoid unbounded disk usage.

Telemetry is enabled by default; set `TELEMETRY=false` in the environment if you need to disable it locally, and contact founders@morphik.ai for additional compliance questions.
