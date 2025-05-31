# Migration Guide: Honeycomb to Logfire

This guide helps you migrate your telemetry from Honeycomb to Pydantic Logfire.

## Overview

Logfire is built on OpenTelemetry just like our previous Honeycomb integration, making the migration straightforward. The main changes are:
- Different API endpoint
- Different authentication method
- No dataset concept (Logfire uses projects instead)

## Steps to Migrate

### 1. Get a Logfire Account and Token

1. Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev)
2. Create a new project for your databridge telemetry
3. Go to project settings and copy your API token

### 2. Update Environment Variables

Replace your Honeycomb API key with the Logfire token:

```bash
# Remove old Honeycomb variable
unset HONEYCOMB_API_KEY

# Add new Logfire token
export LOGFIRE_TOKEN=your_logfire_token_here
```

### 3. Update Configuration Files

The configuration files (`morphik.toml`) have been updated automatically:
- `honeycomb_enabled` → `logfire_enabled`
- `honeycomb_endpoint` → `logfire_endpoint`
- `honeycomb_proxy_endpoint` → `logfire_proxy_endpoint`

### 4. Deploy Updated Proxy

If you're running the otel-proxy on a service like Render:
1. Update the environment variable from `HONEYCOMB_API_KEY` to `LOGFIRE_TOKEN`
2. Deploy the updated proxy code

### 5. Restart Services

Restart your databridge services to pick up the new configuration:
- API server
- Worker processes
- Any other services using telemetry

## Key Differences

### Authentication
- **Honeycomb**: Uses `x-honeycomb-team` header
- **Logfire**: Uses `Authorization: Bearer <token>` header

### Data Organization
- **Honeycomb**: Uses datasets (like `databridge-core`)
- **Logfire**: Uses projects (create one for your application)

### Endpoints
- **Honeycomb**: `https://api.honeycomb.io`
- **Logfire**: `https://logfire-api.pydantic.dev`

## Viewing Data in Logfire

1. Log into your Logfire dashboard
2. Select your project
3. Use the Live view to see real-time traces
4. Use the Explorer to query historical data

## Rollback Plan

If you need to rollback to Honeycomb:
1. Revert the code changes
2. Update environment variables back to `HONEYCOMB_API_KEY`
3. Redeploy and restart services

## Benefits of Logfire

- Built by Pydantic team with Python-first design
- Excellent FastAPI integration
- Better handling of Pydantic models in traces
- More intuitive UI for Python developers
- SQL-based querying for advanced analysis

## Support

If you encounter any issues during migration:
1. Check the logs in `logs/telemetry/` for local debugging
2. Verify the proxy is receiving and forwarding requests
3. Check Logfire's status page for any service issues
4. Open an issue in the databridge repository
