# How to Run Kevin's ColPali Evaluation

This guide provides concise instructions to set up and run the `kevin_eval.py` script.

## Setup Instructions

### 1. Install & Configure PostgreSQL

Use [Homebrew](https://brew.sh/) to install PostgreSQL and `pgvector`. Then, use the `psql` shell to create the specific user and database required for the script.

```bash
# Install and start services
brew install postgresql pgvector
brew services start postgresql

# Connect to the default postgres database to create our user and DB
psql postgres
```

Now, run the following SQL commands inside the `psql` shell:

```sql
-- Create a new superuser with a password
CREATE USER kevin_rag WITH SUPERUSER PASSWORD 'secure_password123';

-- Create the database and set the owner to our new user
CREATE DATABASE kevin_rag_db WITH OWNER kevin_rag;

-- Connect to the new database to enable the vector extension
\c kevin_rag_db

-- Enable the vector extension
CREATE EXTENSION vector;

-- Exit the psql shell
\q
```

### 2. Install Python Dependencies

The project uses `uv` for package management. From the project root, create a virtual environment and sync the dependencies.

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies using uv
uv sync
```

### 3. Configure Environment Variables

The `.env.example` file already contains the correct `POSTGRES_URI` for the database you just created. Remember to add the `OPENAI_API_KEY` before running the program.
---

## Running the Script

Navigate to the script's directory (`evaluations/custom_eval`) to run the evaluation.

### Basic Execution

This command ingests documents, runs questions, and saves results.

```bash
python kevin_eval.py
```

### Caching embeddings

The repo should contain embedding caches in the `.embedding_cache`, which is automatically stored each time an embedding is generated.
If you want to test the embedding part feel free to remove the caches from this directory.

### Other Arguments
-   `--skip-ingestion`: Skip document processing if data is already in the database.
-   `--docs-dir`: Specify a different directory for PDF documents.
-   `--questions`: Specify a different path for the questions CSV file.
-   `--debug`: Enable detailed debug logging.