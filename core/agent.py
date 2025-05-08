import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from litellm import acompletion

from core.config import get_settings
from core.models.auth import AuthContext
from core.services.grounding_service import GroundingService
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
        self.grounding_service = GroundingService()
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
        self.system_prompt_template = """
        You are Morphik, an intelligent research assistant. You can use the following tools to help answer user queries:
{tool_list_string}
Use function calls to invoke these tools when needed. When you have gathered all necessary information,
{final_instruction}
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
            case "publish_response":
                return args
            case _:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self, query: str, auth: AuthContext, rich: bool = False, ground: bool = False) -> Dict[str, Any]:
        """Run the agent and return the final answer, potentially in a rich format."""

        tool_list_string = ""
        for tool_def in self.tool_definitions:
            if tool_def["function"]["name"] != "publish_response" or rich:
                tool_list_string += f"- {tool_def['function']['name']}: {tool_def['function']['description']}\n"

        final_instruction = "provide a clear, concise final answer. Include all relevant details and cite your sources."
        if rich:
            final_instruction = """
            call the `publish_response` function with a `body` (markdown) and a `citations` array.
            Each citation should point to the exact text range or image URL you used. This should be your final action.
            """.strip().replace(
                "\n", " "
            )

        current_system_prompt = self.system_prompt_template.format(
            tool_list_string=tool_list_string, final_instruction=final_instruction
        ).strip()

        messages = [
            {"role": "system", "content": current_system_prompt},
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
            "tools": (
                self.tool_definitions
                if rich
                else [td for td in self.tool_definitions if td.get("function", {}).get("name") != "publish_response"]
            ),
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
                return {"mode": "plain", "body": msg.content}, tool_history

            call = msg.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments)
            logger.info(f"Tool call detected: {name} with args: {args}")

            if name == "publish_response":
                if rich:
                    logger.info(f"Publish response call detected: {args}")
                    if ground and "citations" in args and isinstance(args["citations"], list):
                        logger.info(f"Performing grounding for {len(args['citations'])} citations.")
                        try:
                            grounded_citations = await self.grounding_service.ground_citations(
                                citations=args["citations"],
                                original_user_query=query,
                                agent_answer_body=args.get("body", ""),
                                auth=auth,
                            )
                            args["citations"] = grounded_citations
                            logger.info("Grounding complete.")
                        except Exception as e:
                            logger.error(f"Error during grounding: {e}", exc_info=True)
                    # Decide if we want to return ungrounded or fail.
                    # For now, continue with ungrounded (or partially grounded).
                    return {"mode": "rich", **args}, tool_history
                else:
                    # Model called publish_response in non-rich mode, treat as plain response
                    logger.warning("publish_response called in non-rich mode. Returning plain text.")
                    # We could try to extract body from args if available, or just use msg.content
                    final_content = args.get(
                        "body", msg.content if msg.content else "Agent tried to publish rich response in plain mode."
                    )
                    return {"mode": "plain", "body": final_content}, tool_history

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
