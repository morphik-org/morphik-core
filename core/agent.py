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

        # TODO: Evaluate and improve the prompt here please!
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
provide a clear, concise final answer. Include all relevant details and cite your sources.
Always use markdown formatting.
""".strip()

    async def _execute_tool(self, name: str, args: dict, auth: AuthContext):
        """Dispatch tool calls, injecting document_service and auth."""
        match name:
            case "retrieve_chunks":
                return await retrieve_chunks(document_service=self.document_service, auth=auth, **args)
            case "retrieve_document":
                return await retrieve_document(document_service=self.document_service, auth=auth, **args)
            case "document_analyzer":
                return await document_analyzer(document_service=self.document_service, auth=auth, **args)
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
                # Return final content and the history
                return msg.content, tool_history

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
