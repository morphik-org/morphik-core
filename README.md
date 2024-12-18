# DataBridge

DataBridge is an extensible, open-source document processing and retrieval system designed for building document-based applications. It provides a modular architecture for integrating document parsing, embedding generation, and vector search capabilities.

## Table of Contents
- [Features](#features)
- [Starting the Server](#starting-the-server)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
  - [Current Integrations](#current-integrations)
  - [Adding New Components](#adding-new-components)
- [API Documentation](#api-documentation)
  - [Key Endpoints](#key-endpoints)
- [License](#license)
- [Contributing](#contributing)

## Features

- 🔌 **Extensible Architecture**: Modular design for easy component extension or replacement
- 🔍 **Vector Search**: Semantic search capabilities
- 🔐 **Authentication**: JWT-based auth with developer and end-user access modes
- 📊 **Components**: Document Parsing (Unstructured API), Vector Store (MongoDB Atlas), Embedding Model (OpenAI), Storage (AWS S3)
- 🚀 **Python SDK**: Simple client SDK for quick integration

## Starting the Server

1. Clone the repository:
```bash
git clone https://github.com/databridge-org/databridge-core.git
```

2. Setup your python environment (Python 3.12 supported, but other versions may work):
```bash
cd databridge-core
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables, using the `.env.example` file as a reference, and creating a `.env` file in the project directory:

```bash
cp .env.example .env
```

<!-- TODO: Add instructions for setting up the environment variables, like setting up monogo account, openai account, etc. -->

5. Generate a local URI:
```bash
python generate_local_uri.py
```
Copy the output and save it for use with the client SDK.

6. Start the server:
```bash
python start_server.py
```
*Tip*: Visit `http://localhost:8000/docs` for the complete OpenAPI documentation.

## Quick Start

Ensure the server is running, then use the SDK to ingest and query documents.

1. Install the SDK:
```bash
pip install databridge-client
```
2. Use the SDK:
```python
import asyncio
from databridge import DataBridge

async def main():
    # Initialize client
    db = DataBridge("your_databridge_uri_here", is_local=True)
    files = ["annual_report_2022.pdf", "marketing_strategy.docx" ,"product_launch_presentation.pptx", "company_logo.png"]
    
    for file in files:
      await db.ingest_file(
          file=file,
          file_name=file,
          metadata={"category": "Company Related"} # Optionally add any metadata
      )
    
    # Query documents
    results = await db.query(
        query="What did our target market say about our product?",
        return_type="chunks",
        filters={"category": "Company Related"}
    )

    print(results)

asyncio.run(main())
```

For other examples <!-- -like how to make xyz in 10 lines of code- --> checkout our [documentation](https://databridge.gitbook.io/databridge-docs)!

## Architecture

DataBridge uses a modular architecture with the following base components that can be extended or replaced:

### Current Integrations

- **Document Parser**: Unstructured API integration for intelligent document processing
  - Extend `BaseParser` to add new parsing capabilities
- **Vector Store**: MongoDB Atlas Vector Search integration
  - Extend `BaseVectorStore` to add new vector stores
- **Embedding Model**: OpenAI embeddings integration
  - Extend `BaseEmbeddingModel` to add new embedding models
- **Storage**: AWS S3 integration
  - Storage utilities can be modified in `utils/`

### Adding New Components

1. Implement the relevant base class from `core/`
2. Register your implementation in the service configuration
3. Update environment variables if needed

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the complete OpenAPI documentation.

### Key Endpoints

- `POST /ingest`: Ingest new documents
- `POST /query`: Query documents using semantic search
- `GET /documents`: List all documents
- `GET /document/{doc_id}`: Get specific document details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please open an issue or submit a pull request.

---

Built with ❤️ by DataBridge.
