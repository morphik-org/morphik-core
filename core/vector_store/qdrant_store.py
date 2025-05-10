import json
import logging
from typing import List, Literal, Optional, Tuple, cast
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import models

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)
QDRANT_COLLECTION_NAME = "vector_embeddings"


def _to_point_id(doc_id: str, chunk_number: int):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk_number}.{doc_id}.internal"))


def _get_qdrant_distance(metric: Literal["cosine", "dotProduct"]) -> models.Distance:
    match metric:
        case "cosine":
            return models.Distance.COSINE
        case "dotProduct":
            return models.Distance.DOT


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, host: str, port: int, https: bool) -> None:
        from core.config import get_settings

        settings = get_settings()

        self.dimensions = settings.VECTOR_DIMENSIONS
        self.collection_name = QDRANT_COLLECTION_NAME
        self.distance = _get_qdrant_distance(settings.EMBEDDING_SIMILARITY_METRIC)
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            https=https,
        )

    async def _create_collection(self):
        return await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimensions,
                distance=self.distance,
                on_disk=True,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )

    async def _check_collection_vector_size(self):
        collection = await self.client.get_collection(self.collection_name)
        params = collection.config.params
        assert params.vectors is not None
        vectors = cast(models.VectorParams, params.vectors)
        if vectors.size != self.dimensions:
            msg = f"Vector collection changed from {vectors.size} to {self.dimensions}. This requires recreating tables and will delete all existing vector data."
            logger.error(msg)
            raise ValueError(msg)
        return True

    async def initialize(self):
        logger.info("Initialize qdrant vector collection")
        try:
            if not await self.client.collection_exists(self.collection_name):
                logger.info("Detected no collection exists. Creating qdrant collection")
                await self._create_collection()
            else:
                await self._check_collection_vector_size()

            await self.client.create_payload_index(
                self.collection_name,
                "document_id",
                models.PayloadSchemaType.UUID,
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing Qdrant store: {str(e)}")
            return False

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        try:
            batch = [
                models.PointStruct(
                    id=_to_point_id(chunk.document_id, chunk.chunk_number),
                    vector=cast(List[float], chunk.embedding),
                    payload={
                        "document_id": chunk.document_id,
                        "chunk_number": chunk.chunk_number,
                        "content": chunk.content,
                        "metadata": json.dumps(chunk.metadata) if chunk.metadata is not None else "{}",
                    },
                )
                for chunk in chunks
            ]
            await self.client.upsert(collection_name=self.collection_name, points=batch)
            return True, [cast(str, p.id) for p in batch]
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False, []

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        try:
            query = None
            if doc_ids is not None:
                query = models.Filter(
                    must=models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=doc_ids),
                    ),
                )

            resp = await self.client.query_points(
                self.collection_name,
                query=query_embedding,
                limit=k,
                query_filter=query,
                with_payload=True,
            )
            return [
                DocumentChunk(
                    document_id=p.payload["document_id"],
                    chunk_number=p.payload["chunk_number"],
                    content=p.payload["content"],
                    embedding=[],
                    metadata=json.loads(p.payload["metadata"]),
                    score=p.score,
                )
                for p in resp.points
                if p.payload is not None
            ]
        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        try:
            if not chunk_identifiers:
                return []

            ids = [_to_point_id(doc_id, chunk_number) for (doc_id, chunk_number) in chunk_identifiers]
            resp = await self.client.retrieve(
                self.collection_name,
                ids=ids,
            )
            return [
                DocumentChunk(
                    document_id=p.payload["document_id"],
                    chunk_number=p.payload["chunk_number"],
                    content=p.payload["content"],
                    embedding=[],
                    metadata=json.loads(p.payload["metadata"]),
                    score=0,
                )
                for p in resp
                if p.payload is not None
            ]
        except Exception as e:
            logger.error(f"Error retrieving chunks by ID: {str(e)}")
            return []

    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        try:
            await self.client.delete(
                self.collection_name,
                points_selector=models.Filter(
                    must=models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
            return False
