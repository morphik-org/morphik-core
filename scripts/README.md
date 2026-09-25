# Development Scripts

This directory contains various utility scripts for development.

## Reliance Appeal ingestion

Use `ingest_reliance_appeal.py` to inventory and ingest supported files from a
Reliance Appeal folder. It recursively includes PDF, DOCX, TIFF, PNG, JPG, and
JPEG files, while excluding ZIP archives and generated HTML/CSS/images/XML
export internals. The generated JSON manifest is local runtime output and
contains each relative source path, extracted content when available, detected
document dates, hashes, Morphik IDs, and processing status.

```bash
python scripts/ingest_reliance_appeal.py "C:\path\to\Reliance Appeal" `
  --manifest "C:\path\to\reliance-appeal-manifest.json" `
  --folder "/Reliance Appeal"
```

Run `--dry-run` first to create an inventory without calling Morphik. After
ingestion, retrieve chunks with the stored `external_id` or search the indexed
metadata:

```python
from morphik import Morphik

db = Morphik()
chunks = db.retrieve_chunks(
    query="What evidence supports the appeal?",
    filters={"document_date": {"$gte": "2025-01-01"}},
    k=10,
)
```

## Code Formatting

The project uses the following tools for code formatting:

1. `isort` - Sorts imports alphabetically and automatically separates them into sections
2. `black` - Code formatter that enforces a consistent style
3. `ruff` - Fast Python linter with auto-fixes

### Automatic Formatting with Git Pre-commit Hook

The project has a Git pre-commit hook that automatically formats staged Python files in the following order:

1. `isort`
2. `black` (with line length set to 120)
3. `ruff check --fix`

The pre-commit hook is already installed and should run automatically when you commit changes.

### Manual Formatting

To manually format all Python files in the project, run:

```bash
./scripts/format.sh
```

This script runs the same tools in the same order as the pre-commit hook, but on all Python files in the project.

### Configuration

The tools are configured in the `pyproject.toml` file at the root of the project. This ensures consistent formatting regardless of how the tools are invoked.
