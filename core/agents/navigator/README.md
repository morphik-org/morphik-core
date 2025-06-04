# PDF Document Navigator Agent

A sophisticated PDF navigation agent powered by Gemini 2.5 Flash that can navigate through PDF documents and answer deep, analytical questions about their content.

## Features

- **Native PDF Vision**: Uses Gemini's native PDF processing capabilities to understand both text and visual content
- **Intelligent Navigation**: Strategic navigation through documents using function calling
- **Deep Analysis**: Answers complex questions requiring synthesis of information from multiple pages
- **Visual Understanding**: Analyzes diagrams, charts, tables, and other visual elements
- **Zoom Functionality**: Can zoom into specific regions for detailed examination
- **Structured Output**: Provides well-organized responses with clear reasoning

## Architecture

The agent consists of two main components:

1. **DocumentNavigator** (`navigator.py`): Handles PDF loading, page navigation, and image processing
2. **DocumentNavigatorAgent** (`agent.py`): Orchestrates the AI analysis using Gemini 2.5 Flash

## Usage

### Basic Setup

```python
from core.agents.navigator.agent import DocumentNavigatorAgent
from core.storage.base_storage import BaseStorage
from core.database.base_database import BaseDatabase
from core.models.auth import AuthContext

# Initialize the agent
agent = DocumentNavigatorAgent(
    document_id="your-document-id",
    storage=storage_instance,
    db=database_instance,
    auth=auth_context
)

# Initialize the agent (loads PDF)
await agent.initialize()
```

### Deep Document Analysis

```python
# Ask complex questions about the document
response = await agent.analyze_document(
    "What are the main findings presented in this research paper? "
    "Please analyze the methodology, results, and conclusions."
)
print(response)
```

### Quick Navigation

```python
# Navigate to specific pages
result = await agent.quick_navigate("jump", page=5)
print(result)

# Move to next page
result = await agent.quick_navigate("next")
print(result)

# Zoom into a specific region (coordinates are 0-1000 scale)
result = await agent.quick_navigate("zoom",
    ymin=100, ymax=400, xmin=200, xmax=800)
print(result)

# Get current navigation state
state = agent.get_current_state()
print(state)
```

## Available Navigation Tools

The agent has access to the following navigation functions:

### 1. `next_page()`
Navigate to the next page in the document.

### 2. `previous_page()`
Navigate to the previous page in the document.

### 3. `jump_to_page(page: int)`
Jump to a specific page number (1-indexed).

### 4. `zoom_in(ymin: int, ymax: int, xmin: int, xmax: int)`
Zoom into a specific rectangular region of the current page.
- Coordinates are normalized to a 1000x1000 scale
- `ymin`, `ymax`: Top and bottom Y coordinates (0-1000)
- `xmin`, `xmax`: Left and right X coordinates (0-1000)

### 5. `get_current_page_info()`
Get information about the current page and navigation state.

## Example Use Cases

### 1. Research Paper Analysis

```python
query = """
Analyze this research paper and provide:
1. The main research question and hypothesis
2. The methodology used
3. Key findings and results
4. Limitations mentioned by the authors
5. Future research directions suggested
"""

response = await agent.analyze_document(query)
```

### 2. Financial Report Analysis

```python
query = """
Extract key financial metrics from this annual report:
1. Revenue and profit trends
2. Major business segments performance
3. Risk factors mentioned
4. Management's outlook for next year
"""

response = await agent.analyze_document(query)
```

### 3. Technical Documentation Review

```python
query = """
Review this technical specification document and identify:
1. System requirements and dependencies
2. API endpoints and their functions
3. Security considerations
4. Implementation guidelines
"""

response = await agent.analyze_document(query)
```

### 4. Chart and Diagram Analysis

```python
query = """
Analyze all charts, graphs, and diagrams in this document.
For each visual element, describe:
1. What type of chart/diagram it is
2. What data it represents
3. Key trends or patterns shown
4. How it supports the document's main points
"""

response = await agent.analyze_document(query)
```

## Configuration

### Environment Variables

Make sure to set your Gemini API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### Model Configuration

The agent uses Gemini 2.5 Flash with the following configuration:
- **Model**: `gemini-2.5-flash-preview-05-20`
- **Temperature**: 0.1 (for focused analysis)
- **Function Calling**: Enabled for navigation tools
- **System Instructions**: Optimized for document analysis

## Advanced Features

### Multi-Page Analysis

The agent can automatically navigate through multiple pages to gather comprehensive information:

```python
query = """
Compare the financial performance across all quarters mentioned in this report.
Navigate through the document to find all relevant financial data and provide
a comprehensive comparison.
"""

response = await agent.analyze_document(query, max_iterations=15)
```

### Structured Output

For applications requiring structured data:

```python
query = """
Extract all company names, dates, and financial figures mentioned in this document.
Format the response as a structured summary with clear categories.
"""

response = await agent.analyze_document(query)
```

### Error Handling

The agent includes comprehensive error handling:

```python
try:
    response = await agent.analyze_document(query)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Analysis error: {e}")
```

## Performance Considerations

- **Document Size**: Supports PDFs up to 1000 pages (Gemini limit)
- **Iteration Limit**: Default max_iterations is 10, can be increased for complex analyses
- **Memory Usage**: PDF images are loaded into memory; consider document size
- **API Costs**: Each function call and image analysis consumes API tokens

## Best Practices

1. **Clear Queries**: Provide specific, well-structured questions for better results
2. **Iteration Limits**: Adjust max_iterations based on document complexity
3. **Error Handling**: Always wrap calls in try-catch blocks
4. **Resource Management**: Initialize once and reuse the agent for multiple queries
5. **Query Optimization**: Break down very complex queries into smaller, focused questions

## Troubleshooting

### Common Issues

1. **"Navigator not initialized"**: Call `await agent.initialize()` before use
2. **"GEMINI_API_KEY not set"**: Set the environment variable
3. **"Document not found"**: Verify document_id and storage configuration
4. **"Invalid PDF"**: Ensure the document is a valid PDF file

### Debugging

Enable logging to see detailed navigation and function call information:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Dependencies

- `google-genai`: Gemini API client
- `pdf2image`: PDF to image conversion
- `Pillow`: Image processing
- `python-dotenv`: Environment variable management

## License

This component is part of the Morphik project and follows the project's licensing terms.
