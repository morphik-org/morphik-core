import json
import logging
import os

from dotenv import load_dotenv
from litellm import acompletion

from core.config import get_settings
from core.models.auth import AuthContext
from core.tools.tools import (
    document_analyzer,
    execute_code,
    knowledge_graph_query,
    list_documents,
    list_graphs,
    retrieve_chunks,
    retrieve_document,
    save_to_memory,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


class MorphikAgent:
    """
    Morphik agent for orchestrating tools via LiteLLM function calling.
    """

    def __init__(
        self,
        document_service,
        model: str = None,
    ):
        self.document_service = document_service
        self.sources = {}
        # Load settings
        self.settings = get_settings()
        self.model = model or self.settings.AGENT_MODEL
        # Load tool definitions (function schemas)
        desc_path = os.path.join(os.path.dirname(__file__), "tools", "descriptions.json")
        with open(desc_path, "r") as f:
            self.tools_json = json.load(f)

        self.tool_definitions = []
        for tool in self.tools_json:
            self.tool_definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )

        # System prompt
        self.system_prompt = """
You are Morphik, an intelligent research assistant. You can use the following tools to help answer user queries:
- retrieve_chunks: retrieve relevant text and image chunks from the knowledge base
- retrieve_document: get full document content or metadata
- document_analyzer: analyze documents for entities, facts, summary, sentiment, or full analysis
- execute_code: run Python code in a safe sandbox
- knowledge_graph_query: query the knowledge graph for entities, paths, subgraphs, or list entities
- list_graphs: list available knowledge graphs
- save_to_memory: save important information to persistent memory
- list_documents: list documents accessible to you

Use function calls to invoke these tools when needed. When you have gathered all necessary information,
instead of providing a direct text response, you must return a structured response with display objects.

Your response should be a JSON array of display objects, each with:
1. "type": either "text" or "image"
2. "content": for text objects, this is markdown content; for image objects, this is a base64-encoded image
3. "source": the source ID of the chunk where you found this information

Example response format:
```json
[
  {
    "type": "text",
    "content": "## Introduction to the Topic\nHere is some detailed information...",
    "source": "doc123-chunk1"
  },
  {
    "type": "text",
    "content": "This analysis shows that...",
    "source": "doc456-chunk2"
  }
]
```

When you use retrieve_chunks, you'll get source IDs for each chunk. Use these IDs in your response.
For example, if you see "Source ID: doc123-chunk4" for important information, attribute it in your response.

Always attribute the information to its specific source. Break your response into multiple display objects
when citing different sources. Use markdown formatting for text content to improve readability.
""".strip()

    async def _execute_tool(self, name: str, args: dict, auth: AuthContext):
        """Dispatch tool calls, injecting document_service and auth."""
        match name:
            case "retrieve_chunks":
                content, sources = await retrieve_chunks(document_service=self.document_service, auth=auth, **args)
                self.sources.update(sources)
                return content
            case "retrieve_document":
                result = await retrieve_document(document_service=self.document_service, auth=auth, **args)
                # Add document as a source if it's a successful retrieval
                if isinstance(result, str) and not result.startswith("Document") and not result.startswith("Error"):
                    doc_id = args.get("document_id", "unknown")
                    source_id = f"doc{doc_id}-full"
                    self.sources[source_id] = {
                        "document_id": doc_id,
                        "document_name": f"Full Document {doc_id}",
                        "chunk_number": "full",
                    }
                return result
            case "document_analyzer":
                result = await document_analyzer(document_service=self.document_service, auth=auth, **args)
                # Track document being analyzed as a source
                if args.get("document_id"):
                    doc_id = args.get("document_id")
                    analysis_type = args.get("analysis_type", "analysis")
                    source_id = f"doc{doc_id}-{analysis_type}"
                    self.sources[source_id] = {
                        "document_id": doc_id,
                        "document_name": f"Document {doc_id} ({analysis_type})",
                        "analysis_type": analysis_type,
                    }
                return result
            case "execute_code":
                res = await execute_code(**args)
                return res["content"]
            case "knowledge_graph_query":
                return await knowledge_graph_query(document_service=self.document_service, auth=auth, **args)
            case "list_graphs":
                return await list_graphs(document_service=self.document_service, auth=auth, **args)
            case "save_to_memory":
                return await save_to_memory(document_service=self.document_service, auth=auth, **args)
            case "list_documents":
                return await list_documents(document_service=self.document_service, auth=auth, **args)
            case _:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self, query: str, auth: AuthContext) -> str:
        """Synchronously run the agent and return the final answer."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        tool_history = []  # Initialize tool history list
        # Get the full model name from the registered models config
        settings = get_settings()
        if self.model not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{self.model}' not found in registered_models configuration")

        model_config = settings.REGISTERED_MODELS[self.model]
        model_name = model_config.get("model_name")

        # Prepare model parameters
        model_params = {
            "model": model_name,
            "messages": messages,
            "tools": self.tool_definitions,
            "tool_choice": "auto",
        }

        # Add any other parameters from model config
        for key, value in model_config.items():
            if key != "model_name":
                model_params[key] = value

        while True:
            logger.info(f"Sending completion request with {len(messages)} messages")
            resp = await acompletion(**model_params)
            logger.info(f"Received response: {resp}")

            msg = resp.choices[0].message
            # If no tool call, return final content
            if not getattr(msg, "tool_calls", None):
                logger.info("No tool calls detected, returning final content")

                # Parse the response as display objects if possible
                display_objects = []
                default_text = ""

                try:
                    # Check if the response is JSON formatted
                    import re

                    # Try to extract JSON content if present using a regex pattern for common JSON formats
                    json_pattern = r'\[\s*{.*}\s*\]|\{\s*".*"\s*:.*\}'
                    json_match = re.search(json_pattern, msg.content, re.DOTALL)

                    if json_match:
                        potential_json = json_match.group(0)
                        parsed_content = json.loads(potential_json)

                        # Handle both array and object formats
                        if isinstance(parsed_content, list):
                            for item in parsed_content:
                                if isinstance(item, dict) and "type" in item and "content" in item:
                                    # Convert to standardized display object format
                                    display_obj = {
                                        "type": item.get("type", "text"),
                                        "content": item.get("content", ""),
                                        "source": item.get("source", "agent-response"),
                                    }
                                    if "caption" in item and item["type"] == "image":
                                        display_obj["caption"] = item["caption"]
                                    if item["type"] == "image":
                                        display_obj["content"] = self.sources[item["source"]]["content"]
                                    display_objects.append(display_obj)
                        elif (
                            isinstance(parsed_content, dict)
                            and "type" in parsed_content
                            and "content" in parsed_content
                        ):
                            # Single display object
                            display_obj = {
                                "type": parsed_content.get("type", "text"),
                                "content": parsed_content.get("content", ""),
                                "source": parsed_content.get("source", "agent-response"),
                            }
                            if "caption" in parsed_content and parsed_content["type"] == "image":
                                display_obj["caption"] = parsed_content["caption"]
                            if item["type"] == "image":
                                display_obj["content"] = self.sources[item["source"]]["content"]
                            display_objects.append(display_obj)

                    # If no display objects were created, treat the entire content as text
                    if not display_objects:
                        default_text = msg.content
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    default_text = msg.content

                # If no structured display objects were found, create a default text object
                if not display_objects and default_text:
                    display_objects.append({"type": "text", "content": default_text, "source": "agent-response"})

                # Create sources from the collected source IDs in display objects
                sources = []
                seen_source_ids = set()

                for obj in display_objects:
                    source_id = obj.get("source")
                    if source_id and source_id != "agent-response" and source_id not in seen_source_ids:
                        seen_source_ids.add(source_id)
                        # Extract document info from source ID if available
                        if "-" in source_id:
                            parts = source_id.split("-", 1)
                            doc_id = parts[0].replace("doc", "")
                            sources.append(
                                {
                                    "sourceId": source_id,
                                    "documentName": f"Document {doc_id}",
                                    "documentId": doc_id,
                                    "content": self.sources.get(source_id, {"content": ""})["content"],
                                }
                            )
                        else:
                            sources.append(
                                {
                                    "sourceId": source_id,
                                    "documentName": "Referenced Source",
                                    "documentId": "unknown",
                                    "content": self.sources.get(source_id, {"content": ""})["content"],
                                }
                            )

                # Add agent response source if not already included
                if "agent-response" not in seen_source_ids:
                    sources.append(
                        {
                            "sourceId": "agent-response",
                            "documentName": "Agent Response",
                            "documentId": "system",
                            "content": msg.content,
                        }
                    )

                # Add sources from document chunks used during the session
                for source_id, source_info in self.sources.items():
                    if source_id not in seen_source_ids:
                        sources.append(
                            {
                                "sourceId": source_id,
                                "documentName": source_info.get("document_name", "Unknown Document"),
                                "documentId": source_info.get("document_id", "unknown"),
                            }
                        )

                # Return final content, tool history, display objects and sources
                return {
                    "response": msg.content,
                    "tool_history": tool_history,
                    "display_objects": display_objects,
                    "sources": sources,
                }

            call = msg.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments)
            logger.info(f"Tool call detected: {name} with args: {args}")

            # Append assistant text and execute tool
            # logger.info(f"Appending assistant text: {msg}")
            # if msg.content:
            #     messages.append({'role': 'assistant', 'content': msg.content})
            messages.append(msg.to_dict(exclude_none=True))
            logger.info(f"Executing tool: {name}")
            result = await self._execute_tool(name, args, auth)
            logger.info(f"Tool execution result: {result}")

            # Add tool call and result to history
            tool_history.append({"tool_name": name, "tool_args": args, "tool_result": result})

            # Append raw tool output (string or structured data)
            content = [{"type": "text", "text": result}] if isinstance(result, str) else result
            messages.append({"role": "tool", "name": name, "content": content, "tool_call_id": call.id})

            logger.info("Added tool result to conversation, continuing...")

    def stream(self, query: str):
        """
        (Streaming stub) In future, this will:
          - yield f"[ToolCall] {tool_name}({args})" when a tool is invoked
          - yield f"[ToolResult] {tool_name} -> {result}" after execution
        For now, streaming is disabled; use run() to get the complete answer.
        """
        raise NotImplementedError("Streaming not supported yet; please use run()")
