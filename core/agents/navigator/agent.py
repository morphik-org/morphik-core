import os
from logging import getLogger
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.agents.navigator.navigator import DocumentNavigator, image_to_base64
from core.database.base_database import BaseDatabase
from core.models.auth import AuthContext
from core.storage.base_storage import BaseStorage

load_dotenv(override=True)
logger = getLogger(__name__)


class DocumentNavigatorAgent:
    """
    A PDF navigation agent that can navigate through documents and answer deep queries
    using Gemini 2.5 Flash with native PDF vision support and function calling.
    """

    def __init__(self, document_id: str, storage: BaseStorage, db: BaseDatabase, auth: AuthContext):
        self.document_id = document_id
        self.storage = storage
        self.db = db
        self.auth = auth

        # Initialize navigator
        self.navigator = DocumentNavigator(document_id, storage, db, auth)

        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.gemini = genai.Client(api_key=api_key)

        # Conversation state
        self.messages: List[types.Content] = []
        self.is_initialized = False

        # System instruction for the agent
        self.system_instruction = """You are a PDF Document Navigator Agent with advanced document analysis capabilities.

Your role is to help users navigate through PDF documents and answer deep, analytical questions about their content. You have access to navigation tools that allow you to:

1. Navigate between pages (next_page, previous_page, jump_to_page)
2. Zoom into specific regions of pages for detailed analysis
3. Get current page information and navigation state

Key capabilities:
- Analyze text, images, diagrams, charts, and tables within documents
- Extract structured information from documents
- Answer complex questions that require understanding multiple pages
- Provide detailed analysis of document content and structure
- Navigate strategically through documents to find relevant information

When answering questions:
1. First understand what the user is asking for
2. Navigate through the document strategically to gather relevant information
3. Use zoom functionality to examine specific sections in detail when needed
4. Synthesize information from multiple pages if necessary
5. Provide comprehensive, well-structured answers

Always explain your navigation strategy and what you're looking for as you move through the document."""

    async def initialize(self):
        """Initialize the agent and navigator."""
        if not self.is_initialized:
            await self.navigator.initialize()
            self.is_initialized = True
            logger.info(f"DocumentNavigatorAgent initialized for document {self.document_id}")

    def _get_current_page_content(self) -> types.Content:
        """Get the current page as content for Gemini."""
        try:
            current_page = self.navigator.current_page
            page_info = self.navigator.get_current_page_info()

            # Convert image to base64 for Gemini
            image_data = image_to_base64(current_page)

            return types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_data)),
                    types.Part(text=f"Current page: {page_info['current_page']} of {page_info['total_pages']}"),
                ]
            )
        except Exception as e:
            logger.error(f"Error getting current page content: {e}")
            return types.Content(parts=[types.Part(text=f"Error loading current page: {str(e)}")])

    def _create_model_config(self) -> types.GenerateContentConfig:
        """Create the model configuration with tools."""
        return types.GenerateContentConfig(
            tools=self.navigator.get_navigation_tools(),
            system_instruction=self.system_instruction,
            temperature=0.1,  # Lower temperature for more focused analysis
        )

    async def _handle_function_calls(self, response) -> List[types.Content]:
        """Handle function calls from Gemini response."""
        function_responses = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        function_name = function_call.name
                        arguments = dict(function_call.args) if function_call.args else {}

                        try:
                            # Execute the function
                            result = self.navigator.execute_function(function_name, arguments)

                            # Create function response
                            function_response = types.Content(
                                parts=[
                                    types.Part(
                                        function_response=types.FunctionResponse(name=function_name, response=result)
                                    )
                                ]
                            )
                            function_responses.append(function_response)

                            # If the function returned a cropped image, add it to the conversation
                            if "cropped_image" in result and result["cropped_image"]:
                                cropped_image_data = image_to_base64(result["cropped_image"])
                                image_content = types.Content(
                                    parts=[
                                        types.Part(
                                            inline_data=types.Blob(mime_type="image/png", data=cropped_image_data)
                                        ),
                                        types.Part(text="Zoomed region:"),
                                    ]
                                )
                                function_responses.append(image_content)

                            logger.info(f"Executed function {function_name} with result: {result}")

                        except Exception as e:
                            logger.error(f"Error executing function {function_name}: {e}")
                            error_response = types.Content(
                                parts=[
                                    types.Part(
                                        function_response=types.FunctionResponse(
                                            name=function_name, response={"error": str(e)}
                                        )
                                    )
                                ]
                            )
                            function_responses.append(error_response)

        return function_responses

    async def analyze_document(self, query: str, max_iterations: int = 10) -> str:
        """
        Analyze the document and answer a deep query about its content.

        Args:
            query: The user's question about the document
            max_iterations: Maximum number of function call iterations

        Returns:
            The agent's response to the query
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Start conversation with current page and user query
            current_page_content = self._get_current_page_content()
            user_query = types.Content(parts=[types.Part(text=f"User Query: {query}")])

            # Initialize conversation
            self.messages = [current_page_content, user_query]

            config = self._create_model_config()
            iteration = 0

            while iteration < max_iterations:
                # Generate response
                response = self.gemini.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20", contents=self.messages, config=config
                )

                # Add assistant response to conversation
                if hasattr(response, "candidates") and response.candidates:
                    self.messages.append(response.candidates[0].content)

                # Handle function calls
                function_responses = await self._handle_function_calls(response)

                if function_responses:
                    # Add function responses to conversation
                    self.messages.extend(function_responses)

                    # Add current page content after navigation
                    current_page_content = self._get_current_page_content()
                    self.messages.append(current_page_content)

                    iteration += 1
                    continue
                else:
                    # No more function calls, return the response
                    if hasattr(response, "text") and response.text:
                        return response.text
                    elif hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, "content") and candidate.content.parts:
                            text_parts = [
                                part.text for part in candidate.content.parts if hasattr(part, "text") and part.text
                            ]
                            return "\n".join(text_parts) if text_parts else "No text response generated."

                    return "No response generated."

            return f"Analysis completed after {max_iterations} iterations. The agent may need more iterations to fully answer your query."

        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            return f"Error analyzing document: {str(e)}"

    async def navigate(self, query: str) -> str:
        """
        Legacy method for backward compatibility.
        Delegates to analyze_document.
        """
        return await self.analyze_document(query)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the navigator."""
        if not self.is_initialized:
            return {"error": "Navigator not initialized"}

        try:
            page_info = self.navigator.get_current_page_info()
            return {
                "document_id": self.document_id,
                "current_page": page_info["current_page"],
                "total_pages": page_info["total_pages"],
                "has_next": page_info["has_next"],
                "has_previous": page_info["has_previous"],
                "is_initialized": self.is_initialized,
            }
        except Exception as e:
            return {"error": str(e)}

    async def quick_navigate(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a quick navigation action without full analysis.

        Args:
            action: One of 'next', 'previous', 'jump', 'zoom', 'info'
            **kwargs: Additional arguments for the action

        Returns:
            Result of the navigation action
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if action == "next":
                return self.navigator.next_page()
            elif action == "previous":
                return self.navigator.previous_page()
            elif action == "jump":
                page = kwargs.get("page")
                if page is None:
                    return {"error": "Page number required for jump action"}
                return self.navigator.jump_to_page(page)
            elif action == "zoom":
                required_params = ["ymin", "ymax", "xmin", "xmax"]
                if not all(param in kwargs for param in required_params):
                    return {"error": f"Required parameters for zoom: {required_params}"}
                return self.navigator.zoom_in(kwargs["ymin"], kwargs["ymax"], kwargs["xmin"], kwargs["xmax"])
            elif action == "info":
                return self.navigator.get_current_page_info()
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            logger.error(f"Error in quick navigation: {e}")
            return {"error": str(e)}
