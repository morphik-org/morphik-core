import json
from types import SimpleNamespace

import pytest

from core.models.graph import Entity, Graph, Relationship
from core.tools.analysis_tools import ToolError as AnalysisToolError
from core.tools.analysis_tools import document_analyzer, execute_code

# Import tool functions and exceptions
from core.tools.document_tools import ToolError as DocToolError
from core.tools.document_tools import get_timestamp, list_documents, retrieve_chunks, retrieve_document, save_to_memory
from core.tools.graph_tools import ToolError as GraphToolError
from core.tools.graph_tools import knowledge_graph_query, list_graphs


# Helper fake classes to simulate service behavior
class FakeChunk:
    def __init__(self, content, metadata=None, score=0.0, filename=None):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
        self.filename = filename


class FakeDoc:
    def __init__(self, external_id, system_metadata, metadata=None, filename=None):
        self.external_id = external_id
        self.system_metadata = system_metadata
        self.metadata = metadata or {}
        self.filename = filename


class DummyDocumentService:
    """Dummy document service with configurable chunks, docs, and ingest response"""

    def __init__(self, chunks=None, docs=None, ingest_id="mem123"):  # noqa: D107
        self._chunks = chunks or []
        self._docs = docs or []
        self._ingest_id = ingest_id
        # db.get_documents for list_documents
        self.db = SimpleNamespace(get_documents=self._get_documents)
        # graph_service will be set externally when needed
        self.graph_service = None

    async def retrieve_chunks(self, **kwargs):  # pragma: no cover
        return self._chunks

    async def batch_retrieve_documents(self, document_ids, auth=None, end_user_id=None):  # pragma: no cover
        # Return docs matching requested IDs
        return [doc for doc in self._docs if doc.external_id in document_ids]

    async def ingest_text(self, content, filename, metadata, auth=None, end_user_id=None):  # pragma: no cover
        # Return an object with external_id attribute
        return SimpleNamespace(external_id=self._ingest_id)

    async def _get_documents(self, auth=None, skip=0, limit=100, filters=None, system_filters=None):  # pragma: no cover
        return self._docs


class DummyGraphDB:
    def __init__(self, graphs):  # noqa: D107
        self._graphs = graphs

    async def list_graphs(self, auth=None, system_filters=None):  # pragma: no cover
        return self._graphs

    async def get_graph(self, name, auth=None, system_filters=None):  # pragma: no cover
        for g in self._graphs:
            if g.name == name:
                return g
        return None


class DummyGraphService:
    def __init__(self, graphs):  # noqa: D107
        self.db = DummyGraphDB(graphs)
        # simple stub for embedding model
        self.embedding_model = SimpleNamespace(embed_for_query=lambda text: [1.0, 0.0])

    async def _batch_get_embeddings(self, texts):  # pragma: no cover
        # Return dummy embeddings for each text
        return [[1.0, 0.0] for _ in texts]

    def _calculate_cosine_similarity(self, a, b):  # pragma: no cover
        # Perfect similarity for identical dummy embeddings
        return 1.0


@pytest.fixture
def dummy_doc_service():
    """Fixture for DummyDocumentService"""
    return DummyDocumentService()


@pytest.fixture
def dummy_chunks():
    """Sample fake chunks for testing retrieve_chunks"""
    return [
        FakeChunk(content="text1", metadata={}, score=0.5),
        FakeChunk(content="data:image/png;base64,AAA", metadata={"is_image": True}, score=0.9),
        FakeChunk(content="AAA", metadata={"is_image": True}, score=0.8),
    ]


@pytest.fixture
def sample_docs():
    """Sample fake docs for testing retrieve_document and list_documents"""
    return [
        FakeDoc(
            external_id="doc1",
            system_metadata={"content": "hello world", "other": 123},
            metadata={"m": "v"},
            filename="file1.txt",
        )
    ]


@pytest.fixture
def dummy_graph_docs():
    """Fixture for a single Graph object"""
    # Create entities and relationships
    e1 = Entity(label="Entity1", type="TYPE1", properties={}, document_ids=["docA"], id="ent1")
    r1 = Relationship(source_id="ent1", target_id="ent1", type="self", id="rel1")
    g = Graph(name="g1", entities=[e1], relationships=[r1], document_ids=["docA"])
    return [g]


@pytest.mark.asyncio
async def test_retrieve_chunks_errors():
    # Should raise if document_service not provided
    with pytest.raises(DocToolError):
        await retrieve_chunks(query="q")


@pytest.mark.asyncio
async def test_retrieve_chunks_basic(dummy_doc_service, dummy_chunks):
    svc = DummyDocumentService(chunks=dummy_chunks)
    result = await retrieve_chunks(query="q", document_service=svc)
    # First element is header text
    assert result[0]["type"] == "text"
    assert "Found 3 relevant chunks" in result[0]["text"]
    # Check text chunk formatting
    assert any(item for item in result if item["type"] == "text" and "text1" in item["text"])
    # Check image_url for data URL example
    img_items = [item for item in result if item["type"] == "image_url"]
    assert any(img["image_url"]["url"].startswith("data:image/png;base64,AAA") for img in img_items)
    # Base64 without prefix should be prefixed
    assert any(img for img in img_items if img["image_url"]["url"] == "data:image/png;base64,AAA")


@pytest.mark.asyncio
async def test_retrieve_document_and_list_documents(dummy_doc_service, sample_docs):
    svc = DummyDocumentService(docs=sample_docs)
    # retrieve_document errors
    with pytest.raises(DocToolError):
        await retrieve_document(document_id="doc1", document_service=None)
    # not found case
    empty_svc = DummyDocumentService(docs=[])
    missing = await retrieve_document(document_id="missing", document_service=empty_svc)
    assert "not found or not accessible" in missing
    # text format
    text = await retrieve_document(document_id="doc1", document_service=svc)
    assert text == "hello world"
    # metadata format
    meta = await retrieve_document(document_id="doc1", format="metadata", document_service=svc)
    data = json.loads(meta)
    assert data["metadata"]["m"] == "v"
    assert "content" not in data["system_metadata"]
    # list_documents errors
    with pytest.raises(DocToolError):
        await list_documents(document_service=None)
    # list_documents basic
    ld = await list_documents(document_service=svc)
    ld_data = json.loads(ld)
    assert ld_data["count"] == 1
    assert ld_data["documents"][0]["id"] == "doc1"


@pytest.mark.asyncio
async def test_save_to_memory_and_timestamp(dummy_doc_service):
    # save_to_memory errors
    with pytest.raises(DocToolError):
        await save_to_memory(content="c", memory_type="session", document_service=None)
    # successful save
    res = await save_to_memory(content="c", memory_type="long_term", tags=["t1"], document_service=dummy_doc_service)
    parsed = json.loads(res)
    assert parsed["success"] is True
    assert parsed["memory_type"] == "long_term"
    assert parsed["memory_id"] == "mem123"
    # timestamp format
    ts = await get_timestamp()
    assert "T" in ts and ":" not in ts


@pytest.mark.asyncio
async def test_execute_code_safe_and_errors():
    # invalid code
    with pytest.raises(AnalysisToolError):
        await execute_code("", timeout=10)
    with pytest.raises(AnalysisToolError):
        await execute_code("import os", timeout=10)
    with pytest.raises(AnalysisToolError):
        await execute_code("print('hi')", timeout=-1)
    # safe code
    result = await execute_code("print('hello')", timeout=5)
    assert result["success"] is True
    assert "hello" in result["output"]
    assert any(c for c in result["content"] if c.get("text") and "hello" in c["text"])


@pytest.mark.asyncio
async def test_document_analyzer_errors():
    # missing service
    with pytest.raises(AnalysisToolError):
        await document_analyzer(document_id="x", document_service=None)


@pytest.mark.asyncio
async def test_graph_tools_list_and_entity(dummy_graph_docs):
    # list_graphs errors
    with pytest.raises(GraphToolError):
        await list_graphs(document_service=None)
    # no graphs
    svc = DummyDocumentService()
    svc.graph_service = DummyGraphService(graphs=[])
    out = await list_graphs(document_service=svc)
    data = json.loads(out)
    assert data["message"] == "No graphs found"
    # with graphs
    svc.graph_service = DummyGraphService(graphs=dummy_graph_docs)
    out2 = await list_graphs(document_service=svc)
    data2 = json.loads(out2)
    assert "Found 1 graph(s)" in data2["message"]
    assert data2["graphs"][0]["name"] == "g1"


@pytest.mark.asyncio
async def test_knowledge_graph_query_errors_and_entity(dummy_graph_docs):
    # missing service
    with pytest.raises(GraphToolError):
        await knowledge_graph_query(query_type="entity", start_nodes=["e"], document_service=None)
    # missing start_nodes for entity
    svc = DummyDocumentService()
    svc.graph_service = DummyGraphService(graphs=dummy_graph_docs)
    with pytest.raises(GraphToolError):
        await knowledge_graph_query(query_type="entity", start_nodes=[], document_service=svc)
    # valid entity query
    out = await knowledge_graph_query(query_type="entity", start_nodes=["Entity1"], document_service=svc)
    ent = json.loads(out)
    assert ent["label"] == "Entity1"
    assert ent["id"] == "ent1"
