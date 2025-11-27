# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morphik is an AI-native toolset for visually rich documents and multimodal data. It provides end-to-end RAG (Retrieval-Augmented Generation) capabilities for processing, storing, and querying unstructured documents with advanced multimodal search capabilities including ColPali.

## Architecture

### Backend (Python)
- **Core API**: Located in `/core` - FastAPI-based REST API
- **Database**: PostgreSQL with pgvector for vector similarity search
- **Models**: SQLAlchemy models in `core/models/`
- **Services**: Business logic in `core/services/`
- **Routes**: API endpoints in `core/routes/`
- **Vector Store**: Multiple providers (pgvector, TurboPuffer) in `core/vector_store/`
- **Embedding**: Support for multiple providers (OpenAI, Ollama, Azure) in `core/embedding/`
- **Parser**: Document processing and chunking in `core/parser/`

### Frontend (TypeScript/Next.js)
- **Location**: `ee/ui-component/` - Enterprise Edition UI component
- **Tech Stack**: Next.js, TypeScript, ShadCN UI components, Tailwind CSS
- **Purpose**: Web interface for document upload, chat, and search

### Configuration
- **Main Config**: `morphik.toml` - Central configuration file
- **Models**: Registered AI models with provider-specific settings
- **Components**: Database, embedding, completion, parser, storage configurations

## Development Commands

### Python Backend
```bash
# Install dependencies
uv sync

# Run tests
pytest                    # All tests
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest core/tests/       # Core tests
pytest -v -s            # Verbose with output

# Code quality
ruff check               # Linting
ruff check --fix         # Auto-fix issues
black .                  # Code formatting
isort .                  # Import sorting

# Start development server
python start_server.py   # Direct startup
./start-dev.sh          # Docker development environment
```

### Frontend
```bash
cd ee/ui-component

# Install dependencies and run
npm install
npm run dev              # Development server

# Build and quality
npm run build            # Production build
npm run build:package    # Package build
npm run lint             # ESLint
npm run format           # Prettier formatting
npm run format:check     # Check formatting
```

### Docker Development
```bash
# Development environment with hot reload
./start-dev.sh

# Standard Docker setup
docker compose up --build
docker compose down
docker compose down -v   # Reset all data
```

## Key Components and Patterns

### Configuration System
- Central configuration in `morphik.toml`
- Model registration system supports multiple AI providers
- Environment-specific settings via `.env` files
- Docker vs local development configurations

### Database Architecture
- PostgreSQL with pgvector extension
- Multi-tenant ACL system for document permissions
- Async database operations using asyncpg/SQLAlchemy
- Connection pooling and retry mechanisms

### AI Model Integration
- Abstracted model interface supporting OpenAI, Anthropic, Google, Ollama, Azure
- Vision-capable models for multimodal processing
- Embedding models for vector similarity search
- Completion models for chat and generation

### Document Processing Pipeline
- Unstructured document parsing with vision capabilities
- Chunking strategies with contextual awareness
- Metadata extraction and rules-based processing
- Storage abstraction (local, S3, etc.)

### API Design
- RESTful endpoints in `core/routes/`
- FastAPI with automatic OpenAPI documentation
- Async/await patterns throughout
- Comprehensive error handling and validation

## Testing Strategy

- **Unit Tests**: Component-level testing in `core/tests/`
- **Integration Tests**: Full API testing with database
- **SDK Tests**: Python SDK testing in `sdks/python/morphik/tests/`
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
- **Async Support**: pytest-asyncio for testing async code

## Code Style Guidelines

- **Python**: Google Python Style Guide, Black formatting (120 char lines)
- **TypeScript**: Prettier + ESLint, ShadCN UI patterns
- **Imports**: isort with Black profile
- **Linting**: Ruff for Python, ESLint for TypeScript

## Important Files and Directories

- `core/api.py` - Main FastAPI application and route registration
- `core/config.py` - Configuration loading and validation
- `core/services_init.py` - Service initialization and dependency injection
- `morphik.toml` - Central configuration file
- `start_server.py` - Application entry point
- `pyproject.toml` - Python dependencies and tool configuration
- `ee/ui-component/package.json` - Frontend dependencies and scripts

## Development Notes

- The system supports both cloud and self-hosted deployments
- ColPali integration for advanced multimodal search
- Modular architecture allows swapping providers (embedding, storage, etc.)
- Enterprise Edition (ee/) features are available alongside open-source core
- Authentication supports both JWT and development mode
- Comprehensive telemetry and monitoring capabilities
