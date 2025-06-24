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


def _add_default_items(schema: Dict[str, Any]):
    """Recursively add default `items` to array-typed properties without one.

    Gemini / Vertex requires `items` for every array in JSON-schema. If the
    user omitted it, we default to a simple string array so the schema remains
    valid.
    """
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {"type": "string"}

    # Recurse into nested object/array definitions
    if schema.get("type") == "object":
        for prop in schema.get("properties", {}).values():
            _add_default_items(prop)
    elif schema.get("type") == "array":
        _add_default_items(schema.get("items", {}))


def _ensure_object_schema(user_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the schema has object type at root and fix common omissions."""

    # Defensive: ensure object root
    if user_schema.get("type") != "object":
        user_schema = {
            "type": "object",
            "properties": {"value": user_schema},
            "required": ["value"],
        }

    # Ensure all arrays have an `items` definition to satisfy Gemini/OAI schema requirements
    _add_default_items(user_schema)

    return user_schema


def get_extraction_tools(user_schema: Dict[str, Any]):
    """Return tool definition enforcing the schema."""

    schema = _ensure_object_schema(user_schema)
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_extraction",
                "description": ("Call exactly once and *only* when you are ready to provide the final extracted JSON."),
                "parameters": schema,
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

    # Get workflow model from settings
    model_name = "gpt-4o-mini"  # Default fallback

    # Use workflow model from settings if available
    if hasattr(settings, "WORKFLOW_MODEL") and settings.WORKFLOW_MODEL:
        workflow_model_key = settings.WORKFLOW_MODEL
        if hasattr(settings, "REGISTERED_MODELS") and workflow_model_key in settings.REGISTERED_MODELS:
            model_config = settings.REGISTERED_MODELS[workflow_model_key]
            model_name = model_config.get("model_name", model_config.get("model", "gpt-4o-mini"))
            logger.info(f"Using workflow model from settings: {workflow_model_key} -> {model_name}")
        else:
            logger.warning(f"Workflow model key '{workflow_model_key}' not found in registered models")
    else:
        logger.warning("No workflow model specified in settings, using default")

    # Create extraction agent
    agent = ExtractionAgent(document_service, document_id, auth_ctx)
    await agent.initialize()

    total_pages = agent.get_total_pages()
    # Only include navigation tools that work purely on page images (avoid text-based search)
    allowed_nav_tools = {"get_next_page", "get_previous_page", "go_to_page", "get_total_pages"}
    navigation_tools = [t for t in get_document_navigation_tools() if t["function"]["name"] in allowed_nav_tools]

    tools = get_extraction_tools(schema) + navigation_tools

    # Create system message
    system_message = {
        "role": "system",
        "content": (
            "You are an expert document extraction agent. Your task is to extract ALL fields from the schema.\n\n"
            "CRITICAL WORKFLOW:\n"
            "1. First, understand the document structure:\n"
            "   - Call get_total_pages() to see how many pages exist\n"
            "   - Look at the current page carefully – extract ALL visible schema fields from it\n"
            "   - Note which fields you found and which are still missing\n\n"
            "2. For EACH page you visit:\n"
            "   - Carefully examine the page image (you will ALWAYS receive the page as an image)\n"
            "   - Extract EVERY schema field that appears on that page\n"
            "   - Keep track of what you've found so far\n"
            "   - Many fields may appear on the same page – GET THEM ALL before moving on\n\n"
            "3. Navigation strategy:\n"
            "   - Start at page 1. If fields are still missing after extraction, use get_next_page() to move forward sequentially\n"
            "   - You can also use go_to_page() when you know the exact page number\n"
            "   - Do NOT call any text-search tools; rely solely on visually inspecting each provided image\n"
            "   - When all fields are captured, call submit_extraction exactly once\n\n"
            "4. Extract data EXACTLY as shown:\n"
            "   - No modifications or assumptions\n"
            '   - Empty string "" for missing text fields\n'
            "   - Empty object {} for missing object fields\n"
            "   - Look across forms, tables, headers, footers, and body content on each image\n\n"
            "IMPORTANT: Never navigate away until you've extracted everything visible on the current page!"
        ),
    }

    # Create user message with schema and document info
    user_message = {
        "role": "user",
        "content": (
            f"Extract these fields from the {total_pages}-page document:\n\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"SMART EXTRACTION STRATEGY:\n"
            f"1. First, check get_total_pages() to confirm {total_pages} pages\n"
            f"2. Look at the current page (page 1) – extract ANY schema fields you can see\n"
            f"3. Keep a mental note of what you found vs what's still missing\n"
            f"4. If fields are still missing, sequentially navigate using get_next_page() (or go_to_page)\n"
            f"5. Each time you navigate, extract ALL visible fields from the new page BEFORE moving again\n"
            f"6. Continue until all fields are found or all pages are checked\n"
            f"7. Finally, call submit_extraction exactly once with your findings\n\n"
            f"Remember: Multiple fields often appear on the same page. Don't navigate away\n"
            f"until you've extracted everything visible on the current page!\n\n"
            f"Start by examining the first page carefully."
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
        "tool_choice": "required",  # Force tool use on first call
        "max_tokens": 4096,
        "temperature": 0.1,  # Small temperature for better reasoning
    }

    # Extract data with retry logic
    extracted_data = None
    max_iterations = 20  # Increased to allow thorough document exploration

    for iteration in range(1, max_iterations + 1):
        logger.debug(f"Extraction iteration {iteration}/{max_iterations}")

        # Call the model
        response = await litellm.acompletion(**model_params)
        response_message = response.choices[0].message

        # Attach any tool calls to the assistant message
        assistant_msg = {
            "role": "assistant",
            "content": response_message.content or "",
        }
        if response_message.tool_calls:
            assistant_msg["tool_calls"] = response_message.tool_calls

        # Add assistant message to the conversation history right away
        messages.append(assistant_msg)

        if response_message.tool_calls:
            logger.debug(f"Model requested {len(response_message.tool_calls)} tool calls")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {function_name} with args: {function_args}")

                # Execute the tool
                if function_name == "submit_extraction":
                    # Validate against schema
                    from jsonschema import ValidationError, validate

                    try:
                        validate(function_args, _ensure_object_schema(schema))
                        extracted_data = function_args
                        logger.info("submit_extraction tool produced valid schema output")
                        tool_result = "Data extracted successfully"
                    except ValidationError as ve:
                        logger.warning("submit_extraction validation failed: %s", ve)
                        tool_result = (
                            "Validation error – please call submit_extraction again with JSON that matches the schema"
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

                else:
                    tool_result = f"Unknown tool: {function_name}"

                # Always add tool result to conversation
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result) if tool_result is not None else "",
                        "tool_call_id": tool_call.id,
                    }
                )

            # After processing all tool calls, add page images for navigation tools
            # We need to check which was the last navigation tool called
            last_nav_tool = None
            for tool_call in response_message.tool_calls:
                if tool_call.function.name in allowed_nav_tools:
                    last_nav_tool = tool_call.function.name

            # If there was a navigation tool, show the current page
            if last_nav_tool:
                page_image = agent.get_current_page_image()
                if page_image:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Here is the current page after navigation. "
                                        "IMPORTANT: Extract ALL fields visible on this page before navigating away! "
                                        "Look for names, emails, titles, IDs, and any other schema fields. "
                                        "When ready, either navigate to find missing fields or call submit_extraction if you have everything."
                                    ),
                                },
                                {"type": "image_url", "image_url": {"url": page_image}},
                            ],
                        }
                    )

            # Update model params with new messages
            model_params["messages"] = messages
            # After first iteration, allow auto tool choice
            if iteration == 1:
                model_params["tool_choice"] = "auto"

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
    raise RuntimeError("Structured extraction failed: model did not return submit_extraction tool output")


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
