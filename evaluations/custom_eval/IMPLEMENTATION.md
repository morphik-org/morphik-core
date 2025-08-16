# Kevin's Custom RAG Pipeline: Implementation Plan

This document outlines the plan for building a custom RAG pipeline to achieve a high score in the Morphik evaluation framework.

**Expert Advice Summary:** The key to success is a RAG pipeline that excels at complex financial document reasoning, with a strong emphasis on numerical precision, multi-document synthesis, and concise, accurate response generation. The retrieval quality and generation faithfulness are the primary success indicators.

## Phase 1: Document Parsing and Chunking (Foundation)

*   **Goal:** Accurately parse financial documents and create meaningful chunks that preserve context.
*   **Actions:**
    *   Use `unstructured` or `PyMuPDF` to extract text, tables, and other structural elements from the financial documents (10-Q, investor presentations).
    *   Implement a table-aware and section-aware chunking strategy. Instead of fixed-size chunks, we will aim to create chunks that correspond to logical sections or full tables.
*   **Success Metric:** Chunks should be self-contained and meaningful to a language model.

## Phase 2: Embedding and Retrieval (Precision)

*   **Goal:** Retrieve the most relevant context for a given question, even if it spans multiple documents.
*   **Actions:**
    *   Use a high-quality embedding model (e.g., `text-embedding-3-large` via `litellm`).
    *   Set up `pgvector` to store and query document chunks.
    *   Implement a retrieval strategy that fetches chunks from all documents in the dataset.
    *   Use a re-ranker (`BAAI/bge-reranker-large`) to refine the retrieved results and select the most relevant context.
*   **Success Metric:** High `Context Precision` and `Context Recall` scores in the evaluation.

## Phase 3: Generation and Reasoning (Accuracy & Conciseness)

*   **Goal:** Generate concise, factually correct, and numerically accurate answers.
*   **Actions:**
    *   Construct a well-engineered prompt that instructs the LLM to be concise and to perform calculations if necessary.
    *   Use a powerful language model (e.g., `o4-mini`) for generation.
    *   Implement a multi-step reasoning process. For complex questions, we can break down the problem into smaller steps and make multiple calls to the LLM.
    *   Add a post-processing step to validate numerical answers and ensure they meet the required precision (0.1 decimal tolerance).
*   **Success Metric:** High accuracy in the final evaluation, with a low number of `INCORRECT` judgments.

## Phase 4: Iteration and Evaluation

*   **Goal:** Continuously improve the pipeline based on evaluation results.
*   **Actions:**
    *   Run the full evaluation using `evaluate.py`.
    *   Analyze the `*_judged.csv` file to identify patterns in failures.
    *   Iterate on the chunking, retrieval, and generation strategies to address the identified weaknesses.
