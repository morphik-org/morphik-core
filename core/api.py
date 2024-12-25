import json
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Union
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Depends,
    Header,
    UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
from core.models.request import IngestTextRequest, SearchRequest, CompletionQueryRequest
from core.models.documents import (
    Document,
    DocumentResult,
    ChunkResult
)
from core.models.auth import AuthContext, EntityType
from core.completion.base_completion import CompletionResponse
from core.services.document_service import DocumentService
from core.config import get_settings
from core.database.mongo_database import MongoDatabase
from core.vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from core.storage.s3_storage import S3Storage
from core.parser.unstructured_parser import UnstructuredAPIParser
from core.embedding_model.openai_embedding_model import OpenAIEmbeddingModel
from core.completion.openai_completion import OpenAICompletionModel

# Initialize FastAPI app
app = FastAPI(title="DataBridge API")
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize service
settings = get_settings()

# Initialize components
database = MongoDatabase(
    uri=settings.MONGODB_URI,
    db_name=settings.DATABRIDGE_DB,
    collection_name=settings.DOCUMENTS_COLLECTION
)

vector_store = MongoDBAtlasVectorStore(
    uri=settings.MONGODB_URI,
    database_name=settings.DATABRIDGE_DB,
    collection_name=settings.CHUNKS_COLLECTION,
    index_name=settings.VECTOR_INDEX_NAME
)

storage = S3Storage(
    aws_access_key=settings.AWS_ACCESS_KEY,
    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
    default_bucket=settings.S3_BUCKET
)

parser = UnstructuredAPIParser(
    api_key=settings.UNSTRUCTURED_API_KEY,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

embedding_model = OpenAIEmbeddingModel(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.EMBEDDING_MODEL
)

completion_model = OpenAICompletionModel(
    model_name=settings.COMPLETION_MODEL
)

# Initialize document service
document_service = DocumentService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model,
    completion_model=completion_model
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header"
            )

        token = authorization[7:]  # Remove "Bearer "
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Token expired")

        return AuthContext(
            entity_type=EntityType(payload["type"]),
            entity_id=payload["entity_id"],
            app_id=payload.get("app_id"),
            permissions=set(payload.get("permissions", ["read"]))
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/ingest/text", response_model=Document)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a text document."""
    try:
        return await document_service.ingest_text(request, auth)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/file", response_model=Document)
async def ingest_file(
    file: UploadFile,
    metadata: str = Form("{}"),  # JSON string of metadata
    auth: AuthContext = Depends(verify_token)
) -> Document:
    """Ingest a file document."""
    try:
        metadata_dict = json.loads(metadata)
        doc = await document_service.ingest_file(file, metadata_dict, auth)
        return doc  # TODO: Might be lighter on network to just send the document ID.
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid metadata JSON")


@app.post("/search/chunks", response_model=List[ChunkResult])
async def search_chunks(
    request: SearchRequest,
    auth: AuthContext = Depends(verify_token)
):
    """Search for relevant chunks."""
    return await document_service.search_chunks(
        request.query,
        auth,
        request.filters,
        request.k,
        request.min_score
    )


@app.post("/search/docs", response_model=List[DocumentResult])
async def search_documents(
    request: SearchRequest,
    auth: AuthContext = Depends(verify_token)
):
    """Search for relevant documents."""
    return await document_service.search_docs(
        request.query,
        auth,
        request.filters,
        request.k,
        request.min_score
    )


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest,
    auth: AuthContext = Depends(verify_token)
):
    """Generate completion using relevant chunks as context."""
    return await document_service.query(
        request.query,
        auth,
        request.filters,
        request.k,
        request.min_score,
        request.max_tokens,
        request.temperature
    )


@app.get("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None
):
    """List accessible documents."""
    return await document_service.db.get_documents(auth, skip, limit, filters)


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    auth: AuthContext = Depends(verify_token)
):
    """Get document by ID."""
    try:
        doc = await document_service.db.get_document(document_id, auth)
        logger.info(f"Found document: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document: {e}")
        raise e
