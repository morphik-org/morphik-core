# Morphik Telemetry

Morphik logs minimal operational metadata (operation name, status, duration, token counts) to `logs/telemetry/` so we can keep deployments healthy, then periodically uploads those JSONL files to `https://logs.morphik.ai` to avoid unbounded disk usage.

Telemetry is enabled by default; set `TELEMETRY=false` in the environment if you need to disable it locally, and contact founders@morphik.ai for additional compliance questions.

## Local retention

Self-hosted deployments can configure `telemetry.max_local_bytes` in `morphik.toml` to bound local log
storage during telemetry uploader cycles. The retention scope depends on the uploader outcome:

- Successful uploads: Morphik truncates uploaded telemetry files, then enforces the configured budget
  across `logs/`, preserving the existing success-path cleanup behavior.
- Failed uploads: Morphik retries later and enforces the same byte budget only inside `logs/telemetry/`.
  This prevents proxy outages from growing local telemetry indefinitely without deleting unrelated
  application logs. Local telemetry files may be pruned oldest-first before they are uploaded if
  `logs/telemetry/` exceeds the configured budget.
- No events to upload: cycles with no telemetry files, or files that produce no uploadable bundle, do not
  run pruning.
