"""
Tests for OpenAI SDK compatibility functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from core.api import app
from core.models.openai_compat import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
    OpenAIModelList,
)


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_auth_context():
    """Mock authentication context."""
    return MagicMock(
        entity_type="user",
        entity_id="test_user",
        app_id="test_app",
        permissions=["read", "write"]
    )


@pytest.fixture
def mock_document_service():
    """Mock document service."""
    service = MagicMock()
    service.retrieve_chunks = AsyncMock(return_value=MagicMock(chunks=[]))
    return service


@pytest.fixture
def mock_completion_response():
    """Mock completion response."""
    return MagicMock(
        completion="This is a test response",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="stop"
    )


class TestOpenAICompatibility:
    """Test suite for OpenAI SDK compatibility."""
    
    @patch("core.routes.openai_compat.verify_token")
    @patch("core.routes.openai_compat.get_settings")
    def test_list_models(self, mock_get_settings, mock_verify_token, client, mock_auth_context):
        """Test listing models in OpenAI format."""
        mock_verify_token.return_value = mock_auth_context
        
        # Mock settings with registered models
        mock_settings = MagicMock()
        mock_settings.REGISTERED_MODELS = {
            "gpt-4": {"model_name": "gpt-4", "api_base": "https://api.openai.com"},
            "claude-3": {"model_name": "claude-3", "api_base": "https://api.anthropic.com"}
        }
        mock_get_settings.return_value = mock_settings
        
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert any(model["id"] == "gpt-4" for model in data["data"])
        assert any(model["id"] == "claude-3" for model in data["data"])
    
    @patch("core.routes.openai_compat.verify_token")
    @patch("core.routes.openai_compat.get_settings")
    @patch("core.routes.openai_compat.get_document_service")
    @patch("core.routes.openai_compat.LiteLLMCompletionModel")
    def test_chat_completion_basic(
        self, 
        mock_completion_model_class,
        mock_get_document_service,
        mock_get_settings,
        mock_verify_token,
        client,
        mock_auth_context,
        mock_document_service,
        mock_completion_response
    ):
        """Test basic chat completion."""
        mock_verify_token.return_value = mock_auth_context
        mock_get_document_service.return_value = mock_document_service
        
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.REGISTERED_MODELS = {
            "gpt-4": {"model_name": "gpt-4", "api_base": "https://api.openai.com"}
        }
        mock_get_settings.return_value = mock_settings
        
        # Mock completion model
        mock_completion_model = MagicMock()
        mock_completion_model.complete = AsyncMock(return_value=mock_completion_response)
        mock_completion_model_class.return_value = mock_completion_model
        
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "This is a test response"
        assert data["usage"]["total_tokens"] == 15
    
    @patch("core.routes.openai_compat.verify_token")
    @patch("core.routes.openai_compat.get_settings")
    def test_chat_completion_invalid_model(
        self,
        mock_get_settings,
        mock_verify_token,
        client,
        mock_auth_context
    ):
        """Test chat completion with invalid model."""
        mock_verify_token.return_value = mock_auth_context
        
        # Mock settings with no registered models
        mock_settings = MagicMock()
        mock_settings.REGISTERED_MODELS = {}
        mock_get_settings.return_value = mock_settings
        
        request_data = {
            "model": "invalid-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"]["message"].lower()
    
    @patch("core.routes.openai_compat.verify_token")
    @patch("core.routes.openai_compat.get_settings")
    @patch("core.routes.openai_compat.get_document_service")
    @patch("core.routes.openai_compat.LiteLLMCompletionModel")
    def test_chat_completion_with_rag(
        self,
        mock_completion_model_class,
        mock_get_document_service,
        mock_get_settings,
        mock_verify_token,
        client,
        mock_auth_context,
        mock_completion_response
    ):
        """Test chat completion with RAG enabled."""
        mock_verify_token.return_value = mock_auth_context
        
        # Mock document service with context chunks
        mock_document_service = MagicMock()
        mock_chunks = [
            MagicMock(content="Context chunk 1"),
            MagicMock(content="Context chunk 2")
        ]
        mock_document_service.retrieve_chunks = AsyncMock(
            return_value=MagicMock(chunks=mock_chunks)
        )
        mock_get_document_service.return_value = mock_document_service
        
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.REGISTERED_MODELS = {
            "gpt-4": {"model_name": "gpt-4", "api_base": "https://api.openai.com"}
        }
        mock_get_settings.return_value = mock_settings
        
        # Mock completion model
        mock_completion_model = MagicMock()
        mock_completion_model.complete = AsyncMock(return_value=mock_completion_response)
        mock_completion_model_class.return_value = mock_completion_model
        
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "What's in my documents?"}
            ],
            "use_rag": True,
            "top_k": 5
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        
        # Verify that retrieve_chunks was called
        mock_document_service.retrieve_chunks.assert_called_once()
        
        # Verify that the completion model was called with context chunks
        mock_completion_model.complete.assert_called_once()
        call_args = mock_completion_model.complete.call_args[0][0]
        assert len(call_args.context_chunks) == 2
        assert "Context chunk 1" in call_args.context_chunks
        assert "Context chunk 2" in call_args.context_chunks
    
    def test_openai_message_validation(self):
        """Test OpenAI message model validation."""
        # Valid message
        message = OpenAIMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        
        # Message with multimodal content
        multimodal_content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
        message = OpenAIMessage(role="user", content=multimodal_content)
        assert message.role == "user"
        assert isinstance(message.content, list)
        assert len(message.content) == 2
    
    def test_openai_chat_completion_request_validation(self):
        """Test OpenAI chat completion request validation."""
        request = OpenAIChatCompletionRequest(
            model="gpt-4",
            messages=[
                OpenAIMessage(role="user", content="Hello")
            ],
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.stream is False
        assert request.use_rag is True  # Default value
        assert request.top_k == 5  # Default value


@pytest.mark.asyncio
class TestOpenAICompatibilityAsync:
    """Async test suite for OpenAI SDK compatibility."""
    
    async def test_stream_chat_completion(self):
        """Test streaming chat completion format."""
        from core.routes.openai_compat import stream_chat_completion
        from core.models.completion import CompletionRequest
        
        # Mock completion model with streaming
        mock_completion_model = MagicMock()
        async def mock_stream():
            yield "Hello"
            yield " world"
            yield "!"
        
        mock_completion_model.complete = AsyncMock(return_value=mock_stream())
        
        # Mock OpenAI request
        mock_openai_request = MagicMock()
        mock_openai_request.model = "gpt-4"
        
        # Mock completion request
        completion_request = CompletionRequest(
            query="Hello",
            context_chunks=[],
            stream_response=True
        )
        
        # Collect streaming response
        chunks = []
        async for chunk in stream_chat_completion(
            completion_request, 
            mock_completion_model, 
            mock_openai_request
        ):
            chunks.append(chunk)
        
        # Verify streaming format
        assert len(chunks) >= 4  # Content chunks + final chunk + DONE
        assert any("Hello" in chunk for chunk in chunks)
        assert any("[DONE]" in chunk for chunk in chunks)
        assert all(chunk.startswith("data: ") for chunk in chunks)