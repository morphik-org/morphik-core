import asyncio
import pytest

from core.services.graph_service import GraphService, ExtractionResult, EntityExtraction, RelationshipExtraction

# Dummy settings for testing
class DummySettings:
    GRAPH_MODEL = "dummy"
    REGISTERED_MODELS = {"dummy": {"model_name": "test-model"}}

# Dummy instructor client to capture messages and return a simple ExtractionResult
class DummyClient:
    def __init__(self):
        self.captured_messages = None
        self.chat = self
        self.completions = self

    async def create(self, model, messages, response_model, **kwargs):
        self.captured_messages = messages
        return ExtractionResult(
            entities=[EntityExtraction(label="TestEntity", type="CONCEPT")],
            relationships=[RelationshipExtraction(source="TestEntity", target="TestEntity", relationship="related_to")]
        )

@pytest.fixture(autouse=True)
def patch_settings_and_instructor(monkeypatch):
    # Patch get_settings to return DummySettings
    import core.services.graph_service as gs_mod
    monkeypatch.setattr(gs_mod, "get_settings", lambda: DummySettings())
    # Prepare dummy instructor and litellm modules for dynamic import
    import sys, types
    dummy_client = DummyClient()
    dummy_instructor = types.SimpleNamespace(
        from_litellm=lambda ac, mode: dummy_client,
        Mode=types.SimpleNamespace(JSON=None)
    )
    dummy_litellm = types.SimpleNamespace(acompletion='dummy')
    # Insert into sys.modules so that import instructor/litellm picks up our dummy
    monkeypatch.setitem(sys.modules, 'instructor', dummy_instructor)
    monkeypatch.setitem(sys.modules, 'litellm', dummy_litellm)
    return dummy_client

@pytest.mark.parametrize("content,expected_system_prefix", [
    ('data:image/png;base64,AAA', 'You are an entity extraction and relationship extraction assistant for images.'),
    ('Plain text content.', 'You are an entity extraction and relationship extraction assistant.'),
])
def test_system_prompt_for_image_vs_text(patch_settings_and_instructor, content, expected_system_prefix):
    service = GraphService(db=None, embedding_model=None, completion_model=None)
    entities, relationships = asyncio.run(
        service.extract_entities_from_text(content, doc_id="doc1", chunk_number=0)
    )
    dummy = patch_settings_and_instructor
    assert dummy.captured_messages is not None
    system_msg, _ = dummy.captured_messages
    assert system_msg['content'].startswith(expected_system_prefix)
    assert entities and entities[0].label == "TestEntity"
    assert relationships and relationships[0].type == "related_to"

@pytest.mark.parametrize("content,expected_user_prefix", [
    ('data:image/png;base64,BBB', 'Extract named entities and their relationships from the following base64-encoded image.'),
    ('Hello world', 'Extract named entities and their relationships from the following text.'),
])
def test_user_prompt_for_image_vs_text(patch_settings_and_instructor, content, expected_user_prefix):
    service = GraphService(db=None, embedding_model=None, completion_model=None)
    entities, relationships = asyncio.run(
        service.extract_entities_from_text(content, doc_id="doc2", chunk_number=1)
    )
    dummy = patch_settings_and_instructor
    _, user_msg = dummy.captured_messages
    assert user_msg['content'].startswith(expected_user_prefix)
    # Validate stub relationship
    assert relationships and relationships[0].type == "related_to" 