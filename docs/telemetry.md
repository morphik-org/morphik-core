# Morphik Telemetry

Morphik Core includes operational telemetry for self-hosted deployments. Telemetry is enabled by default.

## Quick Summary

- Telemetry is enabled by default for self-hosted deployments.
- When enabled, events are written locally and uploaded periodically by default; heartbeats are sent separately to `https://logs.morphik.ai`.
- To opt out, set `TELEMETRY=false` before starting Morphik services.
- Each API process startup can send an initial `first_start` heartbeat with its current `installation_id`; after the first successful heartbeat, later pings from that process are labeled `heartbeat`. Setting `TELEMETRY=false` and restarting services stops future telemetry, but it does not retract data already sent.
- Recorded data can include an installation identifier, selected metadata, and raw error text.

## Disable Telemetry

Set `TELEMETRY=false` in the environment for every Morphik process that should not emit telemetry. For Docker-based self-hosting, put the setting in the `.env` file used by Docker Compose before starting services:

```dotenv
TELEMETRY=false
```

For direct local runs, export the variable before starting the API or worker:

```bash
export TELEMETRY=false
```

`TELEMETRY=false` is the opt-out switch. The `[telemetry]` TOML section controls telemetry labels and uploader behavior, but it does not disable telemetry by itself.

If telemetry is enabled when an API process starts, Morphik can send an initial heartbeat labeled `first_start` immediately after startup. This label is per API process lifecycle, not a one-time installation marker; failed or rejected attempts can retry with the `first_start` label until one succeeds. Set `TELEMETRY=false` before the first API or worker process starts when the deployment should avoid sending telemetry from the beginning.

## What Is Recorded

When enabled, Morphik writes JSONL event files under `logs/telemetry/`. Event records can include operation type, status, duration, token counts, `installation_id`, `user_id`, `app_id`, trace identifiers, worker process ID, error messages for failed operations, and selected metadata.

Morphik reads or creates the installation identifier at `~/.databridge/installation_id`. It persists only as long as that path is preserved. The Docker compose files mount `storage`, `logs`, and `morphik.toml`, but they do not mount `~/.databridge`, so container recreation can rotate the identifier unless that path is mounted separately.

Metadata sanitization is exact-key based, not semantic. Metadata keys named `metadata`, `request_dump`, and `request_body` are dropped. Metadata keys named `query`, `folder_name`, `folder_path`, and `full_path` are redacted. Equivalent values may still be emitted when captured under other scalar keys, including filenames, document IDs, chat IDs, end-user IDs, folder names or paths under keys such as `name` or `folder_id`, and folder descriptions under `description`. String metadata values are truncated to 256 characters, and nested list/dict metadata values are dropped.

Failed operations may include raw exception text in the event `error` field. Error text is not covered by the metadata key redaction list or the 256-character metadata truncation rule, so provider errors, paths, IDs, or other request-derived details can appear there.

## Where It Goes

The API startup process starts a heartbeat job and a telemetry log uploader when telemetry is enabled.

- Heartbeats are sent to `https://logs.morphik.ai/api/heartbeat` with fields including `project_name`, `installation_id`, timestamp, version, event type, and signature.
- Telemetry uploads are sent to `https://logs.morphik.ai/api/events/upload`. The upload envelope includes fields such as `installationId`, `startedAt`, `finishedAt`, `eventCount`, `uploaderVersion`, `workerPids`, service/environment/project metadata, byte counts, signature, and a base64-encoded gzip payload containing the JSONL events.
- After a successful upload, Morphik truncates the uploaded `logs/telemetry/usage_events_worker_*.jsonl` files.
- The `/logs` API returns an empty response when the authenticated context has no `app_id`; the checked-in development `morphik.toml` uses auth bypass, so `/logs` stays empty there until auth bypass is disabled and requests carry a real app scope. Requests up to four hours read only local files filtered by both `user_id` and `app_id`, so they can omit recent events that were already uploaded and truncated. Requests over four hours query only `https://logs.morphik.ai/api/events/query` for uploaded events filtered by `app_id`; they do not merge local files, so larger windows can include other users' uploaded events for the same app and can omit recent events that have not uploaded yet.

## Configuration

Telemetry-related keys in the root `morphik.toml` sample configuration include:

```toml
[telemetry]
service_name = "databridge-core"
project_name = ""
upload_interval_hours = 4.0
max_local_bytes = 1073741824
```

- `service_name`: service label used in telemetry resource and upload metadata.
- `project_name`: optional project label used by heartbeat and upload metadata. Empty values fall back to the default OSS project label. The Docker-specific `morphik.docker.toml` template sets this to `oss_docker`; the installer and published Docker image expose that template as `morphik.toml`.
- `upload_interval_hours`: how often the log uploader attempts to send local telemetry files.
- `max_local_bytes`: total byte budget enforced across the `logs/` directory after successful uploads. If the directory exceeds this budget, the uploader removes the oldest files under `logs/`, not only files under `logs/telemetry/`. This budget is not enforced continuously; if uploads are disabled with `upload_interval_hours = 0` or uploads keep failing, local telemetry files can continue to grow.

Set `upload_interval_hours = 0` only if you want to stop the log uploader while leaving local telemetry and heartbeat behavior enabled. Use `TELEMETRY=false` when the deployment should opt out of telemetry.

For additional compliance questions, contact founders@morphik.ai.
