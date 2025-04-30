web: uvicorn core.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --loop asyncio --http auto --ws auto --lifespan auto
worker: arq core.workers.ingestion_worker.WorkerSettings
