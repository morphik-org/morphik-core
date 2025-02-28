# DataBridge Core

![DataBridge Demo](db_atf_demo_hq.gif)

DataBridge is a powerful document processing and retrieval system designed for building intelligent document-based applications. It provides a robust foundation for semantic search, document processing, and AI-powered document interactions.

## Documentation

For detailed information about installation, usage, and development:

- [Installation Guide](https://databridge.gitbook.io/databridge-docs/getting-started/installation)
- [Quick Start Guide](https://databridge.gitbook.io/databridge-docs/getting-started/quickstart)
- [API Reference](https://databridge.gitbook.io/databridge-docs/api-reference/overview)

## Core Features

- 🔍 **Semantic Search & Retrieval**
  - Intelligent chunk-based document splitting
  - Two-stage ranking with vector similarity and neural reranking
  - Advanced filtering and metadata support
  - Configurable similarity thresholds and result limits

- 📄 **Document Processing**
  - Support for PDFs, Word documents, text files, and more
  - Intelligent text extraction with structure preservation
  - Video content parsing with transcription and metadata extraction
  - Automatic chunk generation and embedding
  - Metadata and access control management

- 🔌 **Extensible Architecture**
  - Modular design with swappable components
  - Support for custom parsers and embedding models
  - Flexible storage backends (S3, local, etc.)
  - Vector store integrations (PostgreSQL with pgvector)

- 🔐 **Security & Access Control**
  - Fine-grained document access control
  - Reader/Writer/Admin permission levels
  - JWT-based authentication
  - API key management

- 💻 **Deployment Options**
  - Full local deployment support with Ollama for embeddings
  - Cloud deployment with managed services
  - Hybrid deployment options
  - Docker container support

## Key Endpoints

- **Document Operations**
  - `POST /ingest/text`: Ingest text content
  - `POST /ingest/file`: Ingest file (PDF, DOCX, video, etc.)
  - `GET /documents`: List all documents
  - `GET /documents/{doc_id}`: Get document details
  - `DELETE /documents/{doc_id}`: Delete a document

- **Search & Retrieval**
  - `POST /retrieve/chunks`: Search document chunks
  - `POST /retrieve/docs`: Search complete documents
  - `POST /query`: Generate completions using context
  - `GET /documents/{doc_id}/chunks`: Get document chunks

- **System Operations**
  - `GET /health`: System health check
  - `GET /usage/stats`: Get usage statistics
  - `GET /usage/recent`: Get recent operations
  - `POST /api-keys`: Generate API keys

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please open an issue or submit a pull request.

---

Built with ❤️ by DataBridge
