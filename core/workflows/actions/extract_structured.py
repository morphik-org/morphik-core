"""
Extract structured data from documents using configurable schema.

This action uses the ExtractionAgent to navigate documents and extract
data according to a provided JSON schema.
"""

import json
import logging
from typing import Any, Dict

import litellm

from core.models.workflows import ActionDefinition
from core.tools.document_navigation_tools import get_document_navigation_tools
from core.tools.extraction_agent import ExtractionAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action definition for registration
# ---------------------------------------------------------------------------

ACTION_DEFINITION = ActionDefinition(
    id="morphik.actions.extract_structured",
    name="Extract Structured Data",
    description="Extract structured data from documents using AI based on a provided schema",
    parameters_schema={
        "type": "object",
        "properties": {
            "schema": {
                "type": "object",
                "description": "JSON Schema defining the structure of data to extract",
            },
        },
        "required": ["schema"],
    },
    output_schema={
        "type": "object",
        "description": "The extracted data matching the provided schema",
    },
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _wrap_schema(user_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Embed user_schema under the key 'extracted_data' for tool IO."""

    # Defensive: ensure object root
    if user_schema.get("type") != "object":
        user_schema = {
            "type": "object",
            "properties": {"value": user_schema},
            "required": ["value"],
        }

    return {
        "type": "object",
        "properties": {"extracted_data": user_schema},
        "required": ["extracted_data"],
        "additionalProperties": False,
    }


def get_extraction_tools(user_schema: Dict[str, Any]):
    """Return tool definition enforcing the wrapped schema."""

    wrapped = _wrap_schema(user_schema)
    return [
        {
            "type": "function",
            "function": {
                "name": "extract_data",
                "description": ("Call exactly once and *only* when you are ready to provide the final extracted JSON."),
                "parameters": wrapped,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Runtime implementation
# ---------------------------------------------------------------------------


async def run(document_service, document_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute extraction using the extraction agent.

    Parameters
    ----------
    document_service : DocumentService
        Service to fetch document & chunks.
    document_id : str
        Target document.
    params : dict
        Action parameters (validated by WorkflowService).

    Returns JSON serialisable dict (must match output_schema).
    """

    schema: Dict[str, Any] = params["schema"]

    # Fetch document
    auth_ctx = params.get("auth")  # Provided by WorkflowService during run
    doc = await document_service.db.get_document(document_id, auth_ctx)
    if not doc:
        raise ValueError(f"Document {document_id} not found or access denied")

    # Get model configuration from settings
    from core.config import get_settings

    settings = get_settings()

    # Get workflow model configuration
    workflow_config = getattr(settings, "WORKFLOWS", {})
    workflow_model_key = workflow_config.get("model")

    if workflow_model_key and hasattr(settings, "REGISTERED_MODELS"):
        # Get the model configuration from registered models
        model_config = settings.REGISTERED_MODELS.get(workflow_model_key, {})
        model_name = model_config.get("model_name", "gpt-4o-mini")
    else:
        # Fallback to a good default model for extraction
        model_name = "gpt-4o-mini"

    # Create extraction agent
    agent = ExtractionAgent(document_service, document_id, auth_ctx)
    await agent.initialize()

    total_pages = agent.get_total_pages()
    tools = get_extraction_tools(schema) + get_document_navigation_tools()

    # Create system message
    system_message = {
        "role": "system",
        "content": (
            "You are an expert information extraction agent. Your task is to extract information EXACTLY as it appears in the document.\n"
            "CRITICAL RULES:\n"
            "• You MUST explore the document thoroughly using the navigation tools before extraction\n"
            "• Extract data EXACTLY as written - do NOT make up, infer, or modify any values\n"
            "• If a field cannot be found in the document, use null or an empty string\n"
            "• For the 'title' field, look for the document's actual title, NOT a random word\n"
            "• Navigate through multiple pages if needed to find all required information\n"
            "• Only call extract_data after you have thoroughly explored the document\n"
            "• The extracted data MUST match the provided schema exactly\n"
            "• Use the document navigation tools to view and analyze pages\n"
            "• The document may contain images, tables, or formatted content"
        ),
    }

    # Create user message with schema and document info
    user_message = {
        "role": "user",
        "content": (
            f"I need you to extract structured data from a document with {total_pages} pages.\n\n"
            f"EXTRACTION SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
            f"IMPORTANT: Start by viewing the first page to understand the document structure. "
            f"For a 'title' field, look for the main document title (usually at the top of the first page). "
            f"Use the navigation tools systematically:\n"
            f"1. First check the total pages with get_total_pages()\n"
            f"2. View each page to analyze its content (images, tables, text)\n"
            f"3. Use get_current_page_content() if you need the extracted text\n"
            f"4. Navigate to specific pages as needed with navigation tools\n"
            f"5. Use find_most_relevant_page() to search for specific information\n"
            f"6. Only extract data that you can actually see in the document\n"
            f"7. Do NOT guess or make up values - leave fields empty if not found"
        ),
    }

    messages = [system_message, user_message]

    # Show the first page to start
    first_page_image = agent.get_current_page_image()
    if first_page_image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the first page of the document:"},
                    {"type": "image_url", "image_url": {"url": first_page_image}},
                ],
            }
        )

    # Prepare model parameters
    model_params = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    # Extract data with retry logic
    extracted_data = None
    max_iterations = 10

    for iteration in range(1, max_iterations + 1):
        logger.debug(f"Extraction iteration {iteration}/{max_iterations}")

        # Call the model
        response = await litellm.acompletion(**model_params)
        response_message = response.choices[0].message

        # Add assistant message to conversation
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": response_message.tool_calls,
            }
        )

        # Check if model wants to call tools
        if response_message.tool_calls:
            logger.debug(f"Model requested {len(response_message.tool_calls)} tool calls")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.debug(f"Executing tool: {function_name} with args: {function_args}")

                # Execute the tool
                if function_name == "extract_data":
                    # Validate against schema
                    from jsonschema import ValidationError, validate

                    try:
                        validate(function_args, _wrap_schema(schema))
                        extracted_data = function_args["extracted_data"]
                        logger.info("extract_data tool produced valid schema output")
                        tool_result = "Data extracted successfully"
                    except ValidationError as ve:
                        logger.warning("extract_data validation failed: %s", ve)
                        tool_result = (
                            "Validation error – please call extract_data again with JSON that matches the schema"
                        )
                elif function_name in [
                    "get_next_page",
                    "get_previous_page",
                    "go_to_page",
                    "get_total_pages",
                    "find_most_relevant_page",
                    "get_current_page_content",
                ]:
                    # Execute document navigation tools
                    tool_result = await _execute_agent_tool(agent, function_name, function_args)

                    # After navigation, add the current page image to the conversation
                    if function_name in ["get_next_page", "get_previous_page", "go_to_page"]:
                        page_image = agent.get_current_page_image()
                        if page_image:
                            # Add a message showing the current page
                            messages.append(
                                {
                                    "role": "tool",
                                    "name": function_name,
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result,
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Here is the current page:"},
                                        {"type": "image_url", "image_url": {"url": page_image}},
                                    ],
                                }
                            )
                            continue  # Skip the normal tool result addition
                else:
                    tool_result = f"Unknown tool: {function_name}"

                # Add tool result to conversation
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result) if tool_result is not None else "",
                        "tool_call_id": tool_call.id,
                    }
                )

            # Update model params with new messages
            model_params["messages"] = messages

            # If we got extracted data, we can break
            if extracted_data is not None:
                logger.info(f"Successfully extracted data after {iteration} iterations")
                break

        else:
            # No tool calls, but maybe the model provided a direct response
            logger.debug("No tool calls in response")
            if response_message.content:
                # Try to parse JSON from the content
                try:
                    extracted_data = json.loads(response_message.content)
                    logger.info(f"Extracted data from direct response after {iteration} iterations")
                    break
                except json.JSONDecodeError:
                    # Not JSON, continue
                    pass
            break

    # Return the extracted data
    if extracted_data is not None:
        logger.info("Returning extracted data: %s", extracted_data)
        return extracted_data

    # If we reach here extraction failed
    raise RuntimeError("Structured extraction failed: model did not return extract_data tool output")


async def _execute_agent_tool(agent: ExtractionAgent, function_name: str, function_args: Dict[str, Any]) -> str:
    """Execute an extraction agent tool."""
    try:
        if function_name == "get_next_page":
            return agent.get_next_page()
        elif function_name == "get_previous_page":
            return agent.get_previous_page()
        elif function_name == "go_to_page":
            return agent.go_to_page(function_args["page_number"])
        elif function_name == "get_total_pages":
            return str(agent.get_total_pages())
        elif function_name == "find_most_relevant_page":
            return await agent.find_most_relevant_page(function_args["query"])
        elif function_name == "get_current_page_content":
            content = agent.get_current_page_content()
            if not content:
                return "This page appears to be empty or contains only images."
            return content
        else:
            return f"Unknown tool: {function_name}"
    except Exception as e:
        logger.error(f"Error executing tool {function_name}: {e}")
        return f"Error executing {function_name}: {str(e)}"


# Export for registry
definition = ACTION_DEFINITION
