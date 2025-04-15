Okay, this is a great candidate for a background task queue. Here's a detailed plan designed for your coding LLM, breaking down the implementation of Solution 2 (Dedicated Task Queue using ARQ) step-by-step.

**Overall Goal:** Modify the file ingestion process so that the API endpoint (`/ingest/file`) quickly accepts the file, saves it, queues a background job for processing, and returns an immediate response. A separate worker process will pick up the job and perform the time-consuming parsing, embedding, and database insertion.

**Important Context for the LLM:**

*   **Read these files first:** Before starting, carefully read the contents of:
    *   `core/api.py`: This is the main API file containing the `/ingest/file` endpoint that needs modification. Pay attention to how services (`document_service`, `storage`, etc.) and dependencies (`AuthContext`) are used. Note the existing `startup` and `shutdown` event handlers.
    *   `core/services/document_service.py`: Understand the `ingest_file` method's logic, as this logic will be moved/adapted into the worker task. Note its dependencies (db, vector store, storage, parser, models).
    *   `core/storage/base_storage.py` and its implementations (e.g., `s3_storage.py`, `local_storage.py`): The worker will need to download the file using these.
    *   `core/parser/morphik_parser.py`: The worker will use this for parsing.
    *   `core/embedding/` (relevant models): The worker will use these for embedding.
    *   `core/vector_store/` (relevant stores): The worker will use these to store embeddings.
    *   `core/database/postgres_database.py`: The worker will use this to store document metadata.
    *   `morphik.toml` and potentially `.env`: Understand how configuration (like database URIs, API keys, model settings) is loaded via `core/config.py`. We will add Redis settings here.
*   **Downstream Changes:** Modifying `/ingest/file` will change its return value. Code *outside* this plan that calls this endpoint (like your SDK or test scripts) will need to be updated later to handle the new asynchronous nature (e.g., checking job status instead of getting the `Document` object immediately). This plan focuses *only* on the backend changes for now.
*   **Focus:** Adapt existing logic, don't reinvent it unless necessary for the task queue structure. The core parsing, embedding, and storage logic from `DocumentService.ingest_file` is still valid, it just needs to run in a different process.

---

**Detailed Plan:**

**Phase 1: Setup and Worker Definition**

**Task 1: Install Dependencies and Configure Redis**

1.  **Action:** Add `arq` and `redis` libraries to your project dependencies.
    *   If using `pyproject.toml`, add them under `[tool.poetry.dependencies]` or equivalent.
    *   If using `requirements.txt`, add `arq>=0.25` and `redis>=4.0`.
    *   Run `poetry install` or `pip install -r requirements.txt`.
2.  **Action:** Add Redis connection settings to your configuration.
    *   In your `.env` file (or preferred configuration method referenced by `core/config.py`), add settings for Redis:
        ```dotenv
        REDIS_HOST=localhost
        REDIS_PORT=6379
        # REDIS_PASSWORD=your_password # Optional
        # REDIS_DATABASE=0 # Optional
        ```
    *   **LLM Note:** You don't need to modify `core/config.py` yet, ARQ can read these environment variables directly via `arq.connections.RedisSettings()`.
3.  **Action (Manual):** Ensure a Redis server is running and accessible using the configured settings. For local development, you can use Docker: `docker run -d --name morphik-redis -p 6379:6379 redis`

**Task 2: Create Worker File and Define the Ingestion Task**

1.  **Action:** Create a new Python file: `core/workers/ingestion_worker.py`.
2.  **Action:** Define an `async` function within this file named `process_ingestion_job`.
    *   This function will perform the actual file processing.
    *   **LLM Instruction:** Define the function signature carefully. It needs to accept simple, serializable arguments passed from the API endpoint via the queue. Good arguments would be:
        *   `ctx`: The ARQ context dictionary (always the first argument).
        *   `file_key: str`: The unique key/path where the file is stored (e.g., S3 key or local path relative to storage root).
        *   `bucket: str`: The storage bucket name (e.g., S3 bucket or the base local storage path).
        *   `original_filename: str`: The original name of the uploaded file.
        *   `content_type: str`: The detected MIME type of the file.
        *   `metadata_json: str`: The user-provided metadata as a JSON string.
        *   `auth_dict: dict`: The `AuthContext` passed as a dictionary.
        *   `rules_list: list`: The list of rules (already converted to dictionaries).
        *   `use_colpali: bool`: Flag for ColPali usage.
        *   `folder_name: Optional[str]`: Folder scope.
        *   `end_user_id: Optional[str]`: User scope.
3.  **Action:** Implement the logic *inside* `process_ingestion_job`.
    *   **LLM Instruction:** **Adapt** the core logic currently found within `DocumentService.ingest_file` (in `core/services/document_service.py`). Do *not* rewrite the service methods themselves unless necessary for making them callable from the worker. The steps are:
        1.  Log the start of the job using `logging`.
        2.  Deserialize `metadata_json` and `auth_dict` back into Python objects (dict and `AuthContext`).
        3.  **Download** the file content from storage using `document_service.storage.download_file(bucket, file_key)`. Remember `storage` methods might be `async`.
        4.  Perform parsing: `additional_metadata, text = await document_service.parser.parse_file_to_text(file_content, original_filename)`
        5.  Apply rules: `rule_metadata, modified_text = await document_service.rules_processor.process_rules(text, rules_list)` (ensure `rules_processor` is available). Update metadata and text if rules modified them.
        6.  Create the `Document` object instance (similar to how it's done in `ingest_file`, passing the deserialized `auth` object). Add folder/user scope to `system_metadata`.
        7.  Split text: `chunks = await document_service.parser.split_text(text)`
        8.  Generate **regular** embeddings: `embeddings = await document_service.embedding_model.embed_for_ingestion(chunks)`
        9.  Create regular chunk objects: `chunk_objects = document_service._create_chunk_objects(...)`
        10. Generate **ColPali** embeddings *if* `use_colpali` is true and the models are available (check `document_service.colpali_embedding_model` etc.). This involves `_create_chunks_multivector` and `embed_for_ingestion`. Create `chunk_objects_multivector`.
        11. Store chunks and document metadata: `await document_service._store_chunks_and_doc(...)`. This requires access to `document_service.db`, `document_service.vector_store`, and `document_service.colpali_vector_store`.
        12. Log successful completion.
        13. (Optional) Return the `external_id` of the created document.
    *   **LLM Instruction:** Wrap the entire logic in a `try...except` block. Log errors clearly. If an error occurs, `raise` it so ARQ can handle retries/failure according to its configuration.
    *   **Downstream Impact Check:** This function now duplicates logic from `DocumentService.ingest_file`. Consider refactoring `DocumentService.ingest_file` in a later step to call smaller, reusable methods that this worker task can *also* call, reducing duplication. For now, focus on getting the worker task functional by adapting the existing logic.

**Task 3: Configure the ARQ Worker Settings**

1.  **Action:** In the *same* file (`core/workers/ingestion_worker.py`), define the ARQ `WorkerSettings` class.
    ```python
    # core/workers/ingestion_worker.py
    import arq
    from core.api import initialize_database, initialize_vector_store # Adjust imports as needed
    from core.services.document_service import DocumentService # Need access to init services
    # ... other necessary imports for service initialization ...

    # Your process_ingestion_job async function defined above...

    async def startup(ctx):
        """Worker startup: Initialize shared resources."""
        logger.info("Worker starting up. Initializing services...")
        # LLM: Adapt the initialization logic from core/api.py's startup events
        # Ensure database, vector store, models, and services are initialized ONCE per worker.
        # Make sure the DocumentService instance is available to the task function,
        # perhaps by storing it in the context `ctx`.
        # Example (needs adaptation based on your actual init functions):
        # await initialize_database()
        # await initialize_vector_store()
        # Load models (embedding, completion, colpali, reranker)
        # Create service instances (storage, parser, document_service etc.)
        # Store services in ctx for tasks to use:
        # ctx['document_service'] = initialized_document_service
        # You'll need to import and instantiate all required components here.
        # Pay attention to how models are loaded to avoid reloading per job.
        pass # Replace with actual initialization

    async def shutdown(ctx):
        """Worker shutdown: Clean up resources."""
        logger.info("Worker shutting down.")
        # Example: Close database connections if necessary
        # await core.api.database.engine.dispose() # Needs access to the engine instance
        pass # Replace with actual cleanup

    # ARQ Worker Settings
    class WorkerSettings:
        functions = [process_ingestion_job] # List the task function
        on_startup = startup
        on_shutdown = shutdown
        # ARQ will automatically use REDIS_SETTINGS from env vars if not specified here
        # keep_result_ms = 3600 * 1000 # Keep job results for 1 hour (optional)
        # max_jobs = 10 # Max concurrent jobs per worker (adjust based on resources)
    ```
2.  **LLM Instruction:** Implement the `startup` function. It *must* initialize all necessary services and models (`DocumentService`, database connections, vector stores, embedding models, parser, storage client) needed by `process_ingestion_job`. Store these initialized services in the `ctx` dictionary so the task function can access them (e.g., `document_service = ctx['document_service']`). Look at `core/api.py` for how these are currently initialized globally or during API startup. Replicate that initialization logic here, ensuring models are loaded only once per worker startup.
3.  **LLM Instruction:** Implement the `shutdown` function for any necessary cleanup (e.g., closing database connections).


**Task 5: Update API Startup/Shutdown for ARQ Pool**

1.  **Action:** In `core/api.py`, modify the `startup` and `shutdown` event handlers.
2.  **LLM Instruction:**
    *   Import `arq`.
    *   Define a global variable `redis_pool = None` at the top level.
    *   In the `startup_event` function (or the main `@app.on_event("startup")`):
        *   Add `global redis_pool`.
        *   Create the ARQ Redis pool: `redis_settings = arq.connections.RedisSettings()` (this reads from env vars).
        *   `redis_pool = await arq.create_pool(redis_settings)`.
    *   In the `shutdown_event` function (or `@app.on_event("shutdown")`):
        *   Add `global redis_pool`.
        *   Check if `redis_pool` exists and close it: `if redis_pool: await redis_pool.close()`.

**Phase 3: Running and Verification**

**Task 6: Run the System**

1.  **Action (Manual):** Open two separate terminals.
2.  **Terminal 1:** Start the FastAPI server as usual: `uvicorn core.api:app --reload` (or your normal start command).
3.  **Terminal 2:** Start the ARQ worker: `arq core.workers.ingestion_worker.WorkerSettings` (adjust path if needed). Watch the worker logs for startup messages and job processing.

**Task 7: Verification Plan**

1.  **Action (Manual):** Use a client (like `curl`, Postman, or your Python SDK) to call the `/ingest/file` endpoint with a test file (e.g., the `colpali_example.pdf`).
2.  **Check 1:** Verify the API response is *fast* and returns a JSON like `{"status": "queued", "filename": "...", "job_id": "..."}`.
3.  **Check 2:** Immediately after uploading, try making other API requests (e.g., `/health`, `/query` with existing data). Verify they are *responsive* and not blocked.
4.  **Check 3:** Monitor the **worker terminal logs**. You should see messages indicating it picked up the job (`Processing job for file key...`) and eventually completes (`Successfully processed job...` or an error).
5.  **Check 4:** After the worker logs indicate completion, verify the data was actually ingested:
    *   Use `db.get_document_by_filename(...)` or `db.list_documents(...)` to find the document metadata.
    *   Use `db.retrieve_chunks(...)` with a relevant query to see if chunks were created and stored correctly.
6.  **Check 5:** Test with ColPali enabled and disabled (`use_colpali` flag) to ensure both paths work through the queue.
7.  **Check 6:** Test error handling: Upload an invalid file or trigger an error within the worker task (e.g., disconnect the database temporarily) and check the worker logs for failure messages.

**Phase 4: Follow-up (Optional)**

**Task 8: Implement Job Status Tracking (Separate Task)**

1.  **Action (Manual/LLM):** Recognize that the client now only knows the job was *queued*. Implement a way to check the status.
    *   **Option A (Simple):** Add a status field (e.g., `pending`, `processing`, `completed`, `failed`) to your `Document` model in the database. The worker updates this field. Create a new API endpoint like `/job/status/{job_id}` or `/document/status/{document_id}` that clients can poll.
    *   **Option B (ARQ Results):** Configure ARQ to keep job results (`keep_result_ms` in `WorkerSettings`) and use `arq.job.Job(job_id, redis=redis_pool).result()` in a new API endpoint to fetch the result/status.
    *   **Option C (WebSockets):** For real-time updates, implement WebSocket communication where the worker pushes status updates back to connected clients. (More complex).
    *   **LLM Instruction:** Ask the LLM to implement *one* of these status tracking mechanisms as a separate follow-up task. Option A is often the most straightforward.

This detailed plan provides clear steps, context, and downstream considerations for the LLM to implement the ARQ task queue system effectively. Remember to review the generated code carefully, especially the worker task logic and resource initialization.