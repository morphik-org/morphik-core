"""
Document navigation tools to support headless extraction tasks.

These tools allow structured extraction actions to navigate through
multi-page documents without UI-specific functionality.
"""

# Document navigation tools for extraction
DOCUMENT_NAVIGATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_next_page",
            "description": "Navigate to the next page in the document. Returns success message or error if already at last page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_page",
            "description": "Navigate to the previous page in the document. Returns success message or error if already at first page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_to_page",
            "description": "Navigate to a specific page number in the document. Page numbers are 0-indexed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The page number to navigate to (0-indexed)",
                    }
                },
                "required": ["page_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_summary",
            "description": "Get a summary of a specific page in the document. Useful for understanding page content before navigating to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The page number to get summary for (0-indexed)",
                    }
                },
                "required": ["page_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_pages",
            "description": "Get the total number of pages in the document.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_relevant_page",
            "description": (
                "Search the entire document for content most relevant to a query and automatically navigate to that page. "
                "This tool performs semantic search across all pages and jumps to the most relevant one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content in the document. Can be a question, topic, or keywords.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_page_content",
            "description": "Get the text content of the current page. Returns the extracted text from the page you are currently viewing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def get_document_navigation_tools():
    """Get navigation tools for multi-page document extraction."""
    return DOCUMENT_NAVIGATION_TOOLS
