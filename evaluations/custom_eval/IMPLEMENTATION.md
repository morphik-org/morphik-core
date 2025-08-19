## Technical Overview

*   **Multi-Modal Embeddings**: The `vidore/colpali-v1.3` model is used to generate patch-level vector embeddings from document page images
*   **Hybrid Data Storage**: A PostgreSQL database with the `pgvector` extension, serves as the primary storage. It stores both the visual patch embeddings and the extracted textual content (from PyMuPDF).
*   **Maximum Similarity query mechanism**: For initial retrieval, a "MaxSim" (Maximum Similarity) query mechanism is employed. This method efficiently identifies the most relevant document pages by finding the maximum similarity between query embeddings and the stored document patch embeddings.
*   **Reranking**: The initial retrieval results are further refined using a powerful reranking model. I used the`BAAI/bge-reranker-large` model instead of the default from the `core/reranker/flag_reranker.py` as the latter required more memory than my local machine had.

---

## Core Pipeline Stages

### 1. Document Ingestion

When new PDF documents are provided, the system systematically processes them page by page. For each page, it performs two primary tasks:

*   **Visual Embedding**: The page is first converted into an image. The ColPali model then analyzes this image to generate a set of "patch" embeddings. These are fine-grained vector representations of different visual regions within that page, and they are efficiently cached to avoid redundant processing.
*   **Text Extraction**: Concurrently, the system extracts all available text, identifies and structures tabular data, and organizes text into logical blocks from the PDF page.

### 2. Querying and Answer Generation

*   **Query Encoding**: The user's question is first broken down into a set of multiple query embeddings using the ColPali model. 

*   **Initial Retrieval (MaxSim Scoring)**: Then, for each of these individual query embeddings, the system performs a similarity search within the PostgreSQL database to identify the most relevant image patches across all stored document pages. **This initial search is performed solely on the visual (image patch) embeddings.** It then aggregates these patch-level similarities to score entire document pages. A page's overall score is determined by summing the "maximum similarity" scores—meaning, for each query embedding, only the score of the single best-matching patch on that page contributes to the total. The top-scoring pages are then selected as initial candidates for further processing. **The extracted text is not used in this initial retrieval step, but is leveraged in subsequent reranking and context building phases.**

*   **Reranking**: These initial candidate pages are then passed to the reranking model (FlagReranker). This model performs a deeper, more contextual analysis by comparing the original question directly against the full content of each candidate page.

*   **Context Building**: The most relevant content (including extracted text, structured tables, and text blocks) from the highest-ranked pages is compiled into a comprehensive context string.

*   **Answer Generation**: Finally, this constructed context, along with the original question, is fed into o4-mini (same as `morphik_eval.py`). 