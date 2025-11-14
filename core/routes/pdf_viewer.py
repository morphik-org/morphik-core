import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import arq
import fitz
import litellm
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.config import get_settings
from core.dependencies import get_redis_pool
from core.models.auth import AuthContext
from core.pdf_viewer.tools import PDFViewer, get_pdf_viewer_tools_for_litellm
from core.services_init import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["document"])
litellm.drop_params = True


class DocumentChatRequest(BaseModel):
    """Request model for document chat completion."""

    message: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None


async def get_pdf_viewer(
    document_id: str, auth: AuthContext, api_base_url: str = None, session_id: str = None
) -> PDFViewer:
    document = await document_service.db.get_document(document_id, auth)
    as_bytes = await document_service.storage.download_file(**document.storage_info)
    pdf_document = fitz.open(stream=as_bytes, filetype="pdf")

    if session_id is None:
        import uuid

        session_id = str(uuid.uuid4())

    # Use user ID from auth context
    user_id = auth.user_id if auth and hasattr(auth, "user_id") else "anonymous"

    return PDFViewer(
        pdf_document=pdf_document,
        api_base_url=api_base_url,
        session_id=session_id,
        user_id=user_id,
        document_id=document_id,
        document_service=document_service,
        auth=auth,
    )


@router.get("/chat/{chat_id}")
async def get_document_chat_history(
    chat_id: str,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Retrieve the message history for a document chat conversation."""
    history_key = f"document_chat:{chat_id}"
    stored = await redis.get(history_key)

    if not stored:
        return []

    try:
        data = json.loads(stored)
        return data
    except Exception as e:
        logger.error(f"Error parsing chat history from Redis: {e}")
        return []


async def execute_pdf_tool(pdf_viewer: PDFViewer, tool_call) -> dict:
    """Execute a PDF viewer tool call and return the result with additional metadata."""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    try:
        result = {
            "message": "",
            "tool_name": function_name,
            "args": function_args,
            "current_frame": None,
            "metadata": {},
        }

        if function_name == "get_next_page":
            result["message"] = pdf_viewer.get_next_page()
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {"page": pdf_viewer.current_page + 1, "total_pages": pdf_viewer.total_pages}
        elif function_name == "get_previous_page":
            result["message"] = pdf_viewer.get_previous_page()
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {"page": pdf_viewer.current_page + 1, "total_pages": pdf_viewer.total_pages}
        elif function_name == "go_to_page":
            page_number = function_args.get("page_number", 0)
            result["message"] = pdf_viewer.go_to_page(page_number)
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {"page": pdf_viewer.current_page + 1, "total_pages": pdf_viewer.total_pages}
        elif function_name == "zoom_in":
            box_2d = function_args.get("box_2d", [])
            result["message"] = pdf_viewer.zoom_in(box_2d)
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {"zoom_region": box_2d, "page": pdf_viewer.current_page + 1}
        elif function_name == "zoom_out":
            result["message"] = pdf_viewer.zoom_out()
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {"page": pdf_viewer.current_page + 1}
        elif function_name == "get_page_summary":
            page_number = function_args.get("page_number", 0)
            result["message"] = pdf_viewer.get_page_summary(page_number)
            result["metadata"] = {"page": page_number + 1}
        elif function_name == "get_total_pages":
            total = pdf_viewer.get_total_pages()
            result["message"] = f"Total pages in PDF: {total}"
            result["metadata"] = {"total_pages": total}
        elif function_name == "find_most_relevant_page":
            query = function_args.get("query", "")
            result["message"] = await pdf_viewer.find_most_relevant_page(query)
            result["current_frame"] = pdf_viewer.get_current_frame()
            result["metadata"] = {
                "query": query,
                "page": pdf_viewer.current_page + 1,
                "total_pages": pdf_viewer.total_pages,
            }
        else:
            result["message"] = f"Unknown function: {function_name}"

        return result
    except Exception as e:
        logger.error(f"Error executing PDF tool {function_name}: {e}")
        return {
            "message": f"Error executing {function_name}: {str(e)}",
            "tool_name": function_name,
            "args": function_args,
            "current_frame": None,
            "metadata": {"error": str(e)},
        }


@router.post("/chat/{chat_id}/complete")
async def complete_document_chat(
    chat_id: str,
    request: DocumentChatRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Stream a chat completion response for a document chat conversation."""
    try:
        # Validate request
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Get settings and model configuration
        settings = get_settings()
        model_config = settings.REGISTERED_MODELS.get("gemini_flash", {})

        if not model_config:
            raise HTTPException(status_code=500, detail="Model configuration not found")

        # Get chat history
        history_key = f"document_chat:{chat_id}"
        history: List[Dict[str, Any]] = []

        stored = await redis.get(history_key)
        if stored:
            try:
                history = json.loads(stored)
            except Exception as e:
                logger.error(f"Error parsing chat history: {e}")
                history = []

        # Add user message to history
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        history.append(user_message)

        # Get PDF viewer instance
        # For production, this should be the frontend URL where the PDF viewer is hosted
        # For development, it defaults to localhost:3000
        frontend_api_url = getattr(settings, "PDF_VIEWER_FRONTEND_URL", None)
        try:
            pdf_viewer = await get_pdf_viewer(
                request.document_id, auth, api_base_url=frontend_api_url, session_id=request.session_id
            )
        except Exception as e:
            logger.error(f"Failed to initialize PDF viewer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize PDF viewer: {str(e)}")

        # Get PDF viewer tools
        tools = get_pdf_viewer_tools_for_litellm()

        # Generate streaming response
        async def generate_stream():
            full_response = ""
            conversation_messages = []
            all_messages_for_history = []  # Track all messages in order for history

            try:
                # Prepare initial messages for LiteLLM
                conversation_messages = []

                for msg in history:
                    if msg["role"] == "user":
                        # User messages - just copy role and content, ensure content is string
                        content = safe_content_to_string(msg.get("content"))
                        conversation_messages.append({"role": msg["role"], "content": content})
                    elif msg["role"] == "assistant":
                        # Assistant messages - include tool_calls if present, ensure content is string
                        content = safe_content_to_string(msg.get("content"))
                        assistant_msg = {"role": msg["role"], "content": content}
                        if "tool_calls" in msg and msg["tool_calls"]:
                            assistant_msg["tool_calls"] = msg["tool_calls"]
                        conversation_messages.append(assistant_msg)
                    elif msg["role"] == "tool":
                        # Tool response messages - include all required fields, ensure content is string
                        content = safe_content_to_string(msg.get("content"))
                        conversation_messages.append(
                            {
                                "role": msg["role"],
                                "tool_call_id": msg["tool_call_id"],
                                "name": msg["name"],
                                "content": content,
                            }
                        )
                    elif msg["role"] == "system":
                        # System messages - ensure content is string
                        content = safe_content_to_string(msg.get("content"))
                        conversation_messages.append({"role": msg["role"], "content": content})

                # Add system message if none exists
                if not conversation_messages or conversation_messages[0]["role"] != "system":
                    system_message = {
                        "role": "system",
                        "content": "You are a helpful AI assistant that can navigate and analyze PDF documents. You have access to tools to navigate pages, zoom in/out, search the entire document, and view different parts of the document. When answering user questions:\n\n1. Always explain what you're going to do before using tools\n2. Describe what you see in the current view\n3. Use the 'find_most_relevant_page' tool to search the entire document when users ask about specific topics or content\n4. Use navigation tools when you need to see different parts of the document\n5. When users ask you to show them something specific (uses the words 'show me' for example), zoom into the relevant area using the zoom_in tool, especially if the page has a lot of information - this makes it much easier for users to focus on and understand the specific content you're referencing\n6. Provide detailed explanations of your findings\n7. Always give a comprehensive final answer based on what you've discovered. This final anser must directly answer the users original query or - in exceptional cases - ask for additional information. \n\nBe conversational and explain your reasoning as you work through the document. When users ask about specific topics, use the search functionality to find the most relevant pages automatically.",
                    }
                    conversation_messages.insert(0, system_message)

                # Add current frame image to the conversation
                conversation_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is the current view of the PDF:"},
                            {"type": "image_url", "image_url": {"url": pdf_viewer.current_frame}},
                        ],
                    }
                )

                # Tool calling loop - continue until we get a final response
                max_iterations = 10  # Prevent infinite loops
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1

                    # Prepare LiteLLM parameters
                    model_params = {
                        "model": model_config.get("model_name", "gemini/gemini-2.5-flash-preview-05-20"),
                        "messages": conversation_messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "max_tokens": 10000,
                        "temperature": 0.3,  # Lower temperature for more consistent responses
                        "stream": False,  # Use non-streaming for tool calls
                        "num_retries": 3,
                    }
                    if str(model_params["model"]).startswith("gemini"):
                        model_params["api_key"] = settings.GEMINI_API_KEY

                    # Add any additional model config parameters
                    for key, value in model_config.items():
                        if key != "model_name":
                            model_params[key] = value

                    logger.debug(f"Calling LiteLLM with tools, iteration {iteration}")

                    # Call LiteLLM
                    try:
                        # Log the messages being sent to help debug content issues
                        logger.debug(f"Sending {len(conversation_messages)} messages to LiteLLM")
                        for i, msg in enumerate(conversation_messages):
                            content = msg.get("content")
                            if content is None:
                                logger.warning(f"Message {i} has null content: {msg}")
                                # Fix null content on the spot
                                msg["content"] = ""
                            elif not isinstance(content, (str, list)):
                                logger.warning(
                                    f"Message {i} has non-string/non-list content type {type(content)}: {content}"
                                )
                                # Convert to string
                                msg["content"] = str(content)

                        # Final validation - ensure no message has null content
                        for msg in conversation_messages:
                            if msg.get("content") is None:
                                msg["content"] = ""

                        response = await litellm.acompletion(**model_params)
                        response_message = response.choices[0].message
                    except Exception as litellm_error:
                        logger.error(f"LiteLLM error on iteration {iteration}: {litellm_error}")
                        logger.error(f"Model params: {model_params}")
                        # Log the problematic messages
                        for i, msg in enumerate(conversation_messages):
                            logger.error(
                                f"Message {i}: role={msg.get('role')}, content_type={type(msg.get('content'))}, content={msg.get('content')}"
                            )
                        raise litellm_error

                    logger.debug(f"Model response - Content: {response_message.content}")
                    logger.debug(
                        f"Model response - Tool calls: {len(response_message.tool_calls) if response_message.tool_calls else 0}"
                    )

                    # Check if model wants to call tools
                    if response_message.tool_calls:
                        logger.debug(f"Model requested {len(response_message.tool_calls)} tool calls")

                        # Ensure content is never null - convert to empty string if None
                        response_content = safe_content_to_string(response_message.content)

                        # Create assistant stub with tool calls
                        assistant_stub = {
                            "role": "assistant",
                            "content": response_content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                }
                                for tc in response_message.tool_calls
                            ],
                            "timestamp": datetime.now(UTC).isoformat(),
                        }

                        # Send a single SSE to announce the assistant message with tool calls
                        yield f"data: {json.dumps({'type': 'assistant', 'content': assistant_stub['content'], 'tool_calls': assistant_stub['tool_calls']})}\n\n"

                        # Add assistant message to conversation for LiteLLM
                        conversation_messages.append(
                            {
                                "role": "assistant",
                                "content": response_content,
                                "tool_calls": response_message.tool_calls,
                            }
                        )

                        # Store the assistant message with tool calls in history
                        all_messages_for_history.append(assistant_stub)

                        # Execute each tool call
                        for tool_call in response_message.tool_calls:
                            # Send tool call start event immediately
                            tool_start_event = {
                                "type": "tool_start",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "args": json.loads(tool_call.function.arguments),
                                "status": "executing",
                            }
                            yield f"data: {json.dumps(tool_start_event)}\n\n"

                            # Execute the tool and get result
                            tool_result_data = await execute_pdf_tool(pdf_viewer, tool_call)

                            # Extract the message for LiteLLM and ensure it's never null
                            tool_result_message = safe_content_to_string(tool_result_data.get("message", ""))
                            if not tool_result_message:  # If still empty after conversion
                                tool_result_message = "Tool execution completed successfully"

                            # Send tool completion event with results
                            tool_complete_event = {
                                "type": "tool_complete",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": tool_result_message,
                                "metadata": tool_result_data.get("metadata", {}),
                                "current_frame": tool_result_data.get("current_frame"),
                                "args": tool_result_data.get("args", {}),
                                "status": "completed",
                            }
                            yield f"data: {json.dumps(tool_complete_event)}\n\n"

                            # Add tool response to history in native format with metadata
                            tool_response = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": tool_result_message,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "metadata": tool_result_data.get("metadata", {}),
                                "current_frame": tool_result_data.get("current_frame"),
                                "args": tool_result_data.get("args", {}),
                            }
                            all_messages_for_history.append(tool_response)

                            # Add tool response to conversation (without timestamp) - only message for LiteLLM
                            conversation_messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_call.function.name,
                                    "content": tool_result_message,
                                }
                            )

                        # After tool execution, add updated frame view
                        conversation_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Here is the updated view after the tool execution:"},
                                    {"type": "image_url", "image_url": {"url": pdf_viewer.current_frame}},
                                ],
                            }
                        )

                        # If this was the first iteration and no content was provided,
                        # encourage the model to explain what it's doing
                        if iteration == 1 and not response_message.content:
                            conversation_messages.append(
                                {
                                    "role": "user",
                                    "content": "Please explain what you can see in the current view and what you're looking for.",
                                }
                            )

                        # Continue the loop to get model's response to tool results
                        continue
                    else:
                        # No tool calls, we have a final response
                        if response_message.content:
                            content = safe_content_to_string(response_message.content)
                            full_response += content

                            # Stream in larger chunks (~1KB) for better performance
                            chunk_size = 100
                            for i in range(0, len(content), chunk_size):
                                chunk = content[i : i + chunk_size]
                                yield f"data: {json.dumps({'type': 'assistant', 'content': chunk})}\n\n"
                        break

                # If we've reached max iterations without a final response, provide a fallback
                if iteration >= max_iterations and not full_response:
                    fallback_message = (
                        "I've completed the requested actions on the PDF. Please let me know if you need anything else!"
                    )
                    full_response = fallback_message
                    yield f"data: {json.dumps({'type': 'assistant', 'content': fallback_message})}\n\n"

                # Add the final assistant message if we have a response without tool calls
                # or if this is a continuation after tool calls
                if full_response and not (
                    all_messages_for_history
                    and all_messages_for_history[-1]["role"] == "assistant"
                    and all_messages_for_history[-1].get("content") == full_response
                ):
                    final_assistant_message = {
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    all_messages_for_history.append(final_assistant_message)

                # Add all messages to history in order
                for msg in all_messages_for_history:
                    history.append(msg)

                # Store updated history in Redis
                await redis.set(history_key, json.dumps(history))

                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming completion: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        logger.error(f"Error in document chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def safe_content_to_string(content) -> str:
    """
    Safely convert content to string, handling None/null values.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # For multimodal content (text + images), return as-is
        return content
    return str(content)
