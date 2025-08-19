# How to Run Kevin's ColPali Evaluation

This guide provides concise instructions to set up and run the `kevin_eval.py` script.

### Technical Overview

This RAG evaluator uses a multi-vector approach for advanced document analysis. The key techniques are:

1.  **Multi-Modal Embeddings**: The `vidore/colpali-v1.3` model generates patch-level vector embeddings from document page images, capturing rich visual and textual details.
2.  **Hybrid Data Storage**: A PostgreSQL database with the `pgvector` extension stores the patch embeddings alongside text extracted via PyMuPDF.
3.  **ColBERT-style Retrieval**: A "MaxSim" query mechanism retrieves the most relevant document pages by finding the maximum similarity between query embeddings and document patch embeddings.
4.  **Reranking**: Results are refined using `BAAI/bge-reranker-large` to improve the final context provided to the LLM.

---

## Setup Instructions

### 1. Install & Configure PostgreSQL

Use [Homebrew](https://brew.sh/) to install PostgreSQL and `pgvector`, then create a database and enable the extension.

```bash
# Install and start services
brew install postgresql pgvector
brew services start postgresql

# Create a database and enable the vector extension
createdb colpali_eval
psql colpali_eval -c "CREATE EXTENSION vector;"
```

### 2. Install Python Dependencies

The project uses `uv` for package management. From the project root, create a virtual environment and sync the dependencies.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies using uv
uv sync
```

### 3. Configure Environment Variables
I already included the URI I used to setup my postgres database in .env.example. Feel free to use the same one for faster setup. 

Next, add your OpenAI API key in the same location
# Your OpenAI API Key
OPENAI_API_KEY="sk-..."
```

---

## Running the Script

Navigate to the script's directory (`evaluations/custom_eval`) to run the evaluation.

### Basic Execution

This command ingests documents, runs questions, and saves results.

```bash
python kevin_eval.py
```

### Debug Logging

Use the `--debug` flags for detailed logging.

```bash
python kevin_eval.py --debug
```

### Other Arguments
-   `--skip-ingestion`: Skip document processing if data is already in the database.
-   `--docs-dir`: Specify a different directory for PDF documents.
-   `--questions`: Specify a different path for the questions CSV file.
