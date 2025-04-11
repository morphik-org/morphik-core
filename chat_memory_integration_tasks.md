Okay, here's a breakdown of tasks for your intern to implement chat history and the User Knowledge Graph (UKG) memory feature. Each task is designed to be relatively self-contained, building upon the previous one, with specific checkpoints for verification.

**Important Preliminaries for the Intern:**

1.  **Familiarize with Core Components:** Spend time understanding the basic flow of the existing `/query` endpoint in `core/api.py`. Trace how it uses `DocumentService`, `BaseDatabase`, `BaseVectorStore`, `BaseCompletionModel`, and `GraphService`.
2.  **Understand Models:** Look at the Pydantic models in `core/models/`, especially `Document`, `ChunkResult`, `CompletionResponse`, `Graph`, `Entity`, `Relationship`.
3.  **Environment Setup:** Ensure they have a working local setup where they can run the Morphik server and execute tests.
4.  **Testing Framework:** Understand how tests are run (e.g., using `pytest`).

---

**Task Breakdown:**

**Phase 1: Implementing the Basic Chat API**

**Task 1: Create the Chat Completion API Endpoint Stub**

*   **Goal:** Set up the basic structure for the new chat API endpoint without complex logic.
*   **Steps:**
    1.  Define new Pydantic models in `core/models/request.py` (or a new `chat.py`):
        *   `ChatMessage`: Fields `role` (Literal['user', 'assistant', 'system']) and `content` (str).
        *   `ChatCompletionRequest`: Fields `messages` (List[ChatMessage]), `conversation_id` (Optional[str]), `remember` (Optional[bool]), plus existing RAG/completion params like `filters`, `k`, `temperature`, `max_tokens`, `use_colpali`.
    2.  In `core/api.py`, create a new POST endpoint `/chat/completions`.
    3.  Use `ChatCompletionRequest` as the input model and the existing `CompletionResponse` as the output model.
    4.  Include the `auth: AuthContext = Depends(verify_token)` dependency.
    5.  **Initial Logic:** For now, just log the received messages and return a *fixed, dummy* `CompletionResponse` (e.g., `completion="Received chat request"`). Don't call any services yet.
*   **Checkpoint / Verification:**
    *   Run the server locally.
    *   Use `curl` or a tool like Postman/Insomnia to send a POST request to `/chat/completions` with a sample `ChatCompletionRequest` body (including a list of messages).
    *   **Verify:**
        *   The server returns a `200 OK` status.
        *   The server returns the dummy `CompletionResponse`.
        *   Server logs show the received messages.
        *   Requests without proper authentication fail (`401 Unauthorized`).
        *   Requests with missing required fields (like `messages`) fail (`422 Unprocessable Entity`).

**Task 2: Integrate Basic RAG into Chat Endpoint**

*   **Goal:** Make the chat endpoint perform basic RAG based on the *last* user message, ignoring history for now.
*   **Steps:**
    1.  In the `/chat/completions` handler:
        *   Identify the *last* message in the `messages` list. Assume it's the user's current query. Extract its `content`.
        *   Call `document_service.retrieve_chunks` using this extracted content as the query, passing relevant parameters (`k`, `filters`, `use_colpali`, etc.) from the request.
        *   Create a simple `CompletionRequest` for `document_service.completion_model.complete`:
            *   Use the extracted content as the `query`.
            *   Use the retrieved chunks' `content` for `context_chunks`.
            *   Pass `max_tokens`, `temperature`.
        *   Call `document_service.completion_model.complete`.
        *   Populate the `sources` field in the `CompletionResponse` based on the retrieved chunks.
        *   Return the actual `CompletionResponse` from the completion model.
*   **Checkpoint / Verification:**
    *   Ingest a known document (e.g., using `db.ingest_text` in a test script or the shell).
    *   Send a request to `/chat/completions` with a `messages` list where the last message asks a question about the ingested document.
    *   **Verify:**
        *   The endpoint returns a `200 OK` status.
        *   The `CompletionResponse` contains a relevant answer based on the ingested document.
        *   The `sources` field in the response lists chunks from the ingested document.
        *   Test with different `k` values and `filters` to ensure they are passed correctly to `retrieve_chunks`.

**Task 3: Incorporate Chat History into LLM Prompt**

*   **Goal:** Pass the conversation history (along with RAG context) to the LLM.
*   **Steps:**
    1.  Modify the `/chat/completions` handler *before* calling `document_service.completion_model.complete`.
    2.  Define a strategy for how much history to include (e.g., last 5 messages, or messages within a token limit). Start simple (e.g., fixed number of messages).
    3.  Modify the construction of the `messages` list passed to `completion_model.complete`:
        *   It should now include:
            *   A system prompt.
            *   The selected recent chat history messages (user/assistant turns).
            *   The RAG context (maybe formatted as a special system/user message like "Context documents: ...").
            *   The *very last* user message.
        *   Ensure the message roles (`user`, `assistant`, `system`) are correct.
*   **Checkpoint / Verification:**
    *   Design a simple multi-turn test conversation:
        *   Turn 1 (User): "What is Morphik?"
        *   Turn 2 (Assistant): [Server responds based on RAG]
        *   Turn 3 (User): "How is it different from a standard vector DB?" (This requires context from Turn 1/2).
    *   Send the full message history (`[msg1, msg2, msg3]`) to `/chat/completions`.
    *   **Verify:**
        *   The LLM response to Turn 3 correctly uses information from the previous turns (implicitly provided in the history). The answer should acknowledge "Morphik" mentioned earlier.
        *   (Optional, requires logging): Add logging inside the completion model call (or just before) to print the exact `messages` list being sent to the LLM. Verify it contains the history, RAG context, and the last query in the expected format.

---

This breakdown provides incremental steps with clear verification points, reducing risk and allowing you to monitor the intern's progress effectively. Remember to review the code and test results thoroughly after each task.


Okay, that's a crucial distinction! Morphik is for developers building apps, and *those apps* have end-users whose context/memory needs to be tracked. This changes the scope and ownership of the User Knowledge Graph (UKG).

Here's the revised plan, starting from Task 4 (since you've completed 1-3), incorporating the concept of end-user-specific UKGs owned by the developer account.

**Prerequisite Check (Based on your completed Tasks 1-3):**

*   Does your `ChatCompletionRequest` model (or equivalent input for `/chat/completions`) currently have a way to specify the `end_user_id` for whom the chat is intended?
    *   **If YES:** Great, proceed to Task 4.
    *   **If NO:**
        *   **Action Required (Task 1 Revision):** Modify the `ChatCompletionRequest` Pydantic model to include a *mandatory* field like `end_user_id: str`. Update the `/chat/completions` endpoint signature in `core/api.py` to accept this new request model.
        *   **Checkpoint:** Verify that requests to `/chat/completions` *without* `end_user_id` now fail validation (422), and requests *with* it are accepted (returning the dummy response or basic RAG response from your completed tasks).

---

**Revised Task Breakdown (Starting from Task 4):**

**Phase 2: Implementing End-User Specific UKGs**

**Task 4: Adapt Database & Services for End-User UKGs**

*   **Goal:** Modify database interactions and service logic to handle UKGs named and scoped by both the developer and the end-user.
*   **Steps:**
    1.  **UKG Naming Convention:** Decide on and document the naming scheme. Recommendation: `ukg_{hashed_developer_entity_id}_{hashed_end_user_id}`. Hashing both IDs is good practice for consistency and avoiding potential PII in names. Implement a helper function for consistent hashing (e.g., `sha256(id.encode()).hexdigest()[:16]`).
    2.  **Graph Storage (`GraphModel` / `db` methods):**
        *   When storing/updating a UKG:
            *   The `owner` field should be set using the `developer_auth: AuthContext`.
            *   Store the `hashed_end_user_id` within the `graph_metadata` JSONB field (e.g., `graph_metadata = {'ukg_for_user': hashed_end_user_id, ...}`). This links the graph to the end-user without changing the primary owner.
    3.  **Internal UKG Functions:** Create or modify *internal helper functions* (likely within `GraphService` or a new `MemoryService`) specifically for handling UKGs:
        *   `async def _get_ukg(developer_auth: AuthContext, end_user_id: str) -> Optional[Graph]:` - Constructs the UKG name using hashed IDs, calls `db.get_graph` using the developer's auth context and the specific UKG name.
        *   `async def _store_or_update_ukg(developer_auth: AuthContext, end_user_id: str, graph: Graph) -> bool:` - Constructs the name, sets owner/metadata correctly, and calls `db.store_graph` or `db.update_graph` using the developer's auth context.
    4.  **Authorization:** Reconfirm that `db.get_graph`, `db.store_graph`, `db.update_graph` rely solely on the `developer_auth` context for permission checks. The `end_user_id` is used for *naming* and *metadata*, not direct DB access control for the graph itself.
*   **Checkpoint / Verification:**
    *   **Unit Tests:** Write unit tests for the new internal helper functions (`_get_ukg`, `_store_or_update_ukg`).
        *   Mock `db.get_graph`, `db.store_graph`, `db.update_graph`.
        *   Test `_get_ukg`: Verify it constructs the correct hashed name and calls `db.get_graph` with the right developer auth and graph name.
        *   Test `_store_or_update_ukg`:
            *   Verify it sets the `owner` field correctly from `developer_auth`.
            *   Verify it sets the `ukg_for_user` field in `graph_metadata` with the `hashed_end_user_id`.
            *   Verify it calls `db.store_graph` for a new graph and `db.update_graph` for an existing one (based on mock return from `_get_ukg`).
    *   **Integration Sanity Check:**
        *   Use a test script or shell to call the internal functions directly (if possible, or wrap them in a temporary test endpoint).
        *   Create UKG for (DevA, UserX). Check DB: Graph name is `ukg_hash(A)_hash(X)`, owner is DevA, metadata contains `ukg_for_user=hash(X)`.
        *   Create UKG for (DevA, UserY). Check DB: Graph name `ukg_hash(A)_hash(Y)`, owner DevA, metadata `ukg_for_user=hash(Y)`.
        *   Verify `_get_ukg(DevA, UserX)` retrieves the correct graph.

**Task 5: Implement End-User Aware Asynchronous Memory Ingestion**

*   **Goal:** Create the background task logic that updates the specific end-user's UKG based on remembered conversation segments.
*   **Steps:**
    1.  Define the async function signature: `async def process_memory_update(developer_auth: AuthContext, end_user_id: str, conversation_segment: List[Dict], graph_service: GraphService, db: BaseDatabase, **kwargs)`. Note: Pass dependencies like `graph_service` and `db` explicitly if needed by the background task runner. `ChatMessage` might need to be dicts for serialization.
    2.  **Inside the function:**
        *   Call `graph_service._get_ukg(developer_auth, end_user_id)` to get the existing UKG or `None`.
        *   Concatenate text from `conversation_segment`.
        *   Call `graph_service.extract_entities_from_text` on the text. Pass any `prompt_overrides` if they were included in the original request and passed to the background task.
        *   **If UKG exists:** Perform the merge logic (detailed in previous Task 5 steps), adding new entities/relationships and updating sources of existing ones. Use placeholder doc_ids like `conv_{conversation_id}` or similar.
        *   **If UKG is `None`:** Create a *new* `Graph` object. Populate its `entities` and `relationships` from the extraction.
        *   Call `graph_service._store_or_update_ukg(developer_auth, end_user_id, the_graph_object)`.
*   **Checkpoint / Verification:**
    *   **Unit Test:** Write unit test for `process_memory_update`.
        *   Mock `_get_ukg`, `_store_or_update_ukg`, `extract_entities_from_text`.
        *   Test Case 1 (New UKG): Verify `_get_ukg` returns `None`, then `_store_or_update_ukg` is called with a *new* Graph object containing the extracted info, correct owner, and correct `ukg_for_user` metadata.
        *   Test Case 2 (Update UKG): Mock `_get_ukg` to return an existing graph. Verify `_store_or_update_ukg` is called with the *merged* graph data. Check that existing entity sources are updated, and new entities/relationships are added.

**Task 6: Trigger End-User Aware Memory Ingestion from Chat API**

*   **Goal:** Modify the `/chat/completions` endpoint to trigger the background task for the *correct end-user*.
*   **Steps:**
    1.  Ensure `/chat/completions` accepts `ChatCompletionRequest` including `end_user_id`.
    2.  Inject `BackgroundTasks`, `GraphService`, and `BaseDatabase` dependencies into the endpoint.
    3.  Inside the handler, *after* generating the response:
        *   Check `request.remember is True`.
        *   If true:
            *   Get `developer_auth` from `Depends(verify_token)`.
            *   Get `end_user_id` from `request.end_user_id`.
            *   Determine the `conversation_segment` (e.g., last user/assistant pair). Convert `ChatMessage` objects to simple dicts if necessary for the background task.
            *   Add the task: `background_tasks.add_task(process_memory_update, developer_auth=developer_auth, end_user_id=end_user_id, conversation_segment=segment_dicts, graph_service=graph_service, db=db)`.
*   **Checkpoint / Verification:**
    *   Add logging within `process_memory_update` indicating start/finish and printing the `developer_auth.entity_id` and the received `end_user_id`.
    *   Send two requests to `/chat/completions` with `remember=True`:
        *   Req 1: Auth for DevA, `end_user_id="userX"`.
        *   Req 2: Auth for DevA, `end_user_id="userY"`.
    *   **Verify:**
        *   Responses return quickly.
        *   Logs show `process_memory_update` running twice: once for (DevA, userX) and once for (DevA, userY).
        *   Database check: Confirm two distinct UKGs (`ukg_hash(A)_hash(X)` and `ukg_hash(A)_hash(Y)`) exist and were updated.

**Task 7: Integrate End-User Aware UKG Context into Chat Retrieval**

*   **Goal:** Fetch context from the correct end-user's UKG during chat processing.
*   **Steps:**
    1.  In the `/chat/completions` handler's context generation section:
        *   Get `developer_auth` context.
        *   Get `end_user_id` from `request.end_user_id`.
        *   Call `graph_service._get_ukg(developer_auth, end_user_id)` to retrieve the specific end-user's graph.
        *   **If UKG exists:**
            *   Extract entities from the current query (`last_user_message.content`).
            *   Find relevant entities in the *retrieved UKG*.
            *   Traverse the UKG to get related entities.
            *   Extract the `document_ids` and `chunk_sources` referenced by these relevant UKG entities/relationships.
            *   Use `document_service.batch_retrieve_chunks` (with the `developer_auth` context) to fetch the *original text chunks* pointed to by the UKG references. Store as `ukg_context_chunks`.
        *   **Combine Context:** Merge `ukg_context_chunks` (if any) with the standard `rag_chunks`. Apply deduplication and ranking/prioritization.
        *   Build the LLM prompt using the combined context, clearly indicating the source (e.g., "From your memory:", "From documents:").
*   **Checkpoint / Verification:**
    *   **Setup:**
        *   Dev A ingests Doc A (about Topic X) and Doc B (about Topic Y).
        *   Dev A has Chat 1 with End-User X. Discuss Doc A. Set `remember=True`. Verify `ukg_hash(A)_hash(X)` is updated.
        *   Dev A has Chat 2 with End-User Y. Discuss Doc B. Set `remember=True`. Verify `ukg_hash(A)_hash(Y)` is updated.
    *   **Test 1:** Start a new chat request for End-User X. Ask a question related *only* to Topic A (Doc A).
    *   **Verify 1:** The response should leverage knowledge from Doc A (via UKG). Logs should show `_get_ukg` was called for (DevA, UserX). Check the final prompt to the LLM includes context clearly marked as coming from memory/UKG.
    *   **Test 2:** Start a new chat request for End-User Y. Ask the *same* question related *only* to Topic A (Doc A).
    *   **Verify 2:** The response should *not* use knowledge from Doc A via the UKG (unless standard RAG happens to find it). Logs should show `_get_ukg` was called for (DevA, UserY) and likely returned `None` or an empty graph regarding Topic A.
    *   **Test 3:** Start a new chat request for End-User Y. Ask a question related *only* to Topic B (Doc B).
    *   **Verify 3:** The response should leverage knowledge from Doc B (via UKG). Logs show `_get_ukg` was called for (DevA, UserY).

---

This revised plan correctly scopes the UKG to the end-user while maintaining ownership and access control at the developer level. Remember to emphasize clear logging and the importance of the checkpoints for verification.