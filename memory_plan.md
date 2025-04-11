Okay, let's break down how to integrate chat history and a persistent User Knowledge Graph (UKG) for memory into your Morphik system.

**Core Goals:**

1.  **Support Chat Interactions:** Move beyond single query/response to handle multi-turn conversations.
2.  **Implement Persistent Memory:** Allow specific conversations (or parts) to be "remembered" across different chat sessions.
3.  **Leverage Knowledge Graphs:** Use a dedicated User Knowledge Graph (UKG) per user as the primary mechanism for this persistent memory.
4.  **Hyper-personalization:** Utilize the UKG to tailor responses based on the user's remembered history and context.
5.  **Clean Integration:** Fit this new functionality smoothly into the existing architecture.

**Challenges & Current Limitations:**

*   **Single LLM Call:** The current `/query` likely makes one LLM call with RAG context. Chat requires managing history and context across multiple calls.
*   **State Management:** How to track the state of an ongoing conversation?
*   **Graph Creation/Update Performance:** Creating/updating graphs can be slow. Doing this synchronously within a chat response is problematic.
*   **Context Window Limits:** Combining chat history, RAG context, *and* UKG context into a single LLM prompt needs careful management.
*   **API Design:** Need a new way for the client (SDK/UI) to send chat history and memory flags.
*   **Scalability:** A UKG per user could become resource-intensive.

**Proposed Plan:**

**Phase 1: Foundational Chat API & Logic**

1.  **API Design - New Endpoint:**
    *   Introduce a new endpoint, e.g., `/chat/completions`.
    *   **Input:**
        *   `messages`: A list of dictionaries, following the standard OpenAI format (`[{'role': 'user'/'assistant', 'content': '...'}, ...]`). This replaces the single `query` string.
        *   `conversation_id` (Optional): A unique identifier for the conversation. If provided, the server can potentially retrieve past turns (though initially, we might require the client to send full history).
        *   `remember` (Optional, Boolean): A flag (perhaps on the *last* user message) indicating if this turn (and potentially preceding context) should be added to the user's memory (UKG).
        *   Standard RAG parameters (`filters`, `k`, `min_score`, `use_colpali`, etc.).
        *   Standard completion parameters (`max_tokens`, `temperature`).
        *   *Remove* `graph_name`, `hop_depth`, `include_paths` for this endpoint – the UKG will be implicitly used.
    *   **Output:**
        *   Standard `CompletionResponse` model, containing the assistant's reply, usage, sources, etc.

2.  **Core Logic Modifications (`DocumentService` or a new `ChatService`):**
    *   **Receive Chat History:** Accept the `messages` list.
    *   **Context Generation (Crucial Step):**
        *   **Identify "Current Query":** Determine the actual user query to use for retrieval (likely the content of the last 'user' message).
        *   **Standard RAG Retrieval:** Perform `retrieve_chunks` based on the *current query* and `filters`.
        *   **(Future Phase)** **UKG Retrieval:** Query the user's UKG based on the *current query* and potentially entities extracted from recent chat history.
        *   **Combine Context:** Merge context from standard RAG and UKG retrieval. Prioritization/ranking might be needed here.
    *   **Prompt Construction:**
        *   Build the final prompt *messages* list for the LLM. This needs to include:
            *   System Prompt (defining the AI's role, potentially mentioning its access to memory).
            *   Relevant Chat History (selectively, managing token limits).
            *   Combined RAG + UKG Context.
            *   The latest user message.
        *   This requires logic to truncate/summarize older history or context if the window is exceeded.
    *   **LLM Call:** Call `completion_model.complete` with the constructed `messages` list.
    *   **Memory Handling (Trigger):** If the `remember` flag is true, trigger the memory ingestion process (see Phase 2). This should likely happen *asynchronously* after the response is sent.

**Phase 2: User Knowledge Graph (UKG) Implementation**

1.  **UKG Structure & Scope:**
    *   **Scope:** Define one persistent knowledge graph per `user_id`. Name it predictably, e.g., `ukg_{user_id}`.
    *   **Storage:** Use the existing `GraphService` and `db.store_graph`/`db.update_graph` mechanisms. Ensure user permissions/isolation.
    *   **Content:** Store entities and relationships extracted from the "remembered" conversation turns. Consider adding metadata like timestamps or conversation IDs to nodes/edges.

2.  **Memory Ingestion Process (`remember=True` trigger):**
    *   **Asynchronous Task:** When `remember=True` is received, queue an asynchronous background task. This task avoids blocking the chat response.
    *   **Input to Task:** The relevant conversation segment (e.g., the last N turns, or the turns since the last `remember=True`). Also pass the `user_id`.
    *   **Processing:**
        *   Concatenate the text from the relevant conversation turns.
        *   Use `GraphService.extract_entities_from_text` (or a similar function tailored for conversations) to get entities and relationships.
        *   Retrieve the user's UKG (`ukg_{user_id}`). If it doesn't exist, create it.
        *   *Merge* the newly extracted entities/relationships into the existing UKG (similar logic to `GraphService.update_graph`). This is crucial to avoid overwriting and to build cumulative knowledge. Handle potential conflicts or updates intelligently if possible (e.g., updating entity properties).
        *   Save the updated UKG using `db.update_graph`.

3.  **UKG Retrieval (Integration into Phase 1 Core Logic):**
    *   **Querying the UKG:** When handling a `/chat/completions` request:
        *   Get the user's UKG (`ukg_{user_id}`).
        *   Extract relevant entities from the current user query and potentially recent chat history.
        *   Find matching/similar entities in the UKG.
        *   Use `GraphService._expand_entities` and `GraphService._retrieve_entity_chunks` (or similar logic adapted for UKG) to find relevant information *from the documents originally referenced in the UKG*.
        *   *Crucially:* The UKG itself doesn't store the full chunk text, but pointers (doc_id, chunk_num) to the *original* source documents/chunks. Retrieval fetches these original chunks.
    *   **Context Combination:** Decide how to merge UKG context with standard RAG context (e.g., prepend UKG context, interleave, use separate sections in the prompt).

**Phase 3: Refinement & Optimization**

1.  **Context Management:** Implement robust strategies for managing the context window (summarization, selecting key turns, pruning RAG/UKG results).
2.  **Performance:** Optimize UKG update merging. Consider batching updates. Optimize UKG querying.
3.  **Prompt Engineering:** Refine system prompts and context formatting for the LLM to effectively use both chat history and memory.
4.  **Scalability:** Evaluate database/graph performance as UKGs grow. Consider graph sharding or alternative storage if needed.
5.  **Conflict Resolution:** Develop strategies for handling conflicting information added to the UKG over time.
6.  **Explicit Memory Management:** (Optional) Add API endpoints for users to view/manage/delete their UKG content.

**Recommendations:**

1.  **Start with the Chat API:** Implement Phase 1 first. Get the basic chat flow working, passing the full history from the client initially. This decouples chat mechanics from memory.
2.  **Asynchronous UKG Updates:** Memory ingestion *must* be asynchronous to maintain chat responsiveness. Use a task queue (like Celery, RQ, or FastAPI's BackgroundTasks).
3.  **Implicit UKG Usage:** For the `/chat/completions` endpoint, automatically use the `ukg_{user_id}` if it exists. Keep it simple for the developer; they just flag `remember=True`.
4.  **UKG Stores References, Not Content:** The UKG should primarily store the graph structure (entities, relationships) and *references* (doc_id, chunk_num) back to the original ingested data. Retrieval fetches the original chunk content using these references. This keeps the UKG smaller and avoids data duplication.
5.  **User-Scoped Graphs:** Tie UKGs directly to `user_id` for clear personalization and data isolation.
6.  **Refactor `DocumentService`:** Extract reusable components (like chunk retrieval, context formatting) that can be used by both the existing `/query` and the new `/chat/completions` logic. Consider a dedicated `ChatService`.
7.  **Careful Prompting:** The final prompt sent to the LLM needs clear instructions on how to use the different context types (history, RAG docs, UKG info).
8.  **Monitor Performance:** Keep a close eye on graph update times and query latency, especially UKG retrieval.

This plan provides a structured way to evolve from single-query RAG to a more sophisticated chat system with persistent, graph-based memory, addressing the core requirements while acknowledging the technical challenges.