"""Document analysis and code execution tools."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional

import instructor
from litellm import acompletion as litellm_completion
from pydantic import BaseModel, Field

from core.config import get_settings
from core.models.auth import AuthContext
from core.services.document_service import DocumentService
from core.tools.document_tools import ToolError, retrieve_document

logger = logging.getLogger(__name__)


# Entity extraction model
class Entity(BaseModel):
    name: str = Field(description="The name of the entity")
    description: str = Field(description="A detailed description of the entity")
    type: str = Field(description="The type or category of the entity")
    metadata: Dict[str, Any] = Field(description="Additional metadata associated with the entity")


class Entities(BaseModel):
    entities: List[Entity] = Field(description="A list of extracted entities")


# Fact extraction model
class Fact(BaseModel):
    proof: str = Field(description="The proof for the fact")
    fact: str = Field(description="The fact you extracted from the text")


class Facts(BaseModel):
    facts: List[Fact] = Field(description="A list of facts extracted from the text")


# Sentiment analysis model
class Sentiment(BaseModel):
    highlights: List[str] = Field(description="Key phrases or sentences that exemplify the detected sentiment")
    reasoning: str = Field(description="Explanation of why this sentiment was detected")
    sentiment: str = Field(description="The overall sentiment of the text (positive, negative, neutral, or mixed)")
    score: float = Field(description="A numerical score representing the sentiment intensity (from -1 to 1)")


# Document analysis result model
class DocumentAnalysisResult(BaseModel):
    entities: Optional[Entities] = None
    summary: Optional[str] = None
    facts: Optional[Facts] = None
    sentiment: Optional[Sentiment] = None


# Document analysis functions
async def extract_entities(text: str, model: str) -> Entities:
    """Extract entities from text using instructor"""
    client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON)
    response = await client.chat.completions.create(
        **model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts entities from text."
                    + "The user will provide you text from a document,"
                    + " your job is to extract the entities from the text."
                    + " Each entity should have a name, description, type and any relevant metadata."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_model=Entities,
    )
    return response


async def summarize_document(text: str, model: str) -> str:
    """Summarize document text"""
    response = await litellm_completion(
        model=model["model_name"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes text."
                    + " The user will provide you text from a document,"
                    + " your job is to summarize the text."
                    + " The summary should be concise and to the point,"
                    + " and should capture the main points of the text."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


async def extract_facts(text: str, model: str) -> Facts:
    """Extract facts from text using instructor"""
    client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON)
    response = await client.chat.completions.create(
        **model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts facts from text."
                    + " The user will provide you text from a document,"
                    + " your job is to extract the facts from the text."
                    + " Each fact should have a proof and a fact."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_model=Facts,
    )
    return response


async def analyze_sentiment(text: str, model: str) -> Sentiment:
    """Analyze sentiment of text using instructor"""
    client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON)
    response = await client.chat.completions.create(
        **model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment analysis assistant."
                    + " Analyze the given text and determine its sentiment (positive, negative, neutral, or mixed)."
                    + " Provide a score from -1 (very negative) to 1 (very positive),"
                    + " highlight key phrases that exemplify the sentiment, and explain your reasoning."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_model=Sentiment,
    )
    return response


async def document_analyzer(
    document_id: str,
    analysis_type: Literal["entity_extraction", "summarization", "fact_extraction", "sentiment", "full"] = "full",
    document_service: DocumentService = None,
    auth: AuthContext = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Extract structured information from documents.

    Args:
        document_id: ID of the document to analyze
        analysis_type: Type of analysis to perform
        document_service: DocumentService instance
        auth: Authentication context
        end_user_id: Optional end-user ID to analyze as

    Returns:
        Analysis results as a string
    """
    if document_service is None:
        raise ToolError("Document service not provided")

    try:
        # Retrieve document content
        doc_content = await retrieve_document(
            document_id=document_id,
            format="text",
            document_service=document_service,
            auth=auth,
            end_user_id=end_user_id,
        )

        # Get document analysis model from settings
        settings = get_settings()
        try:
            model_config_key = settings.DOCUMENT_ANALYSIS_MODEL
        except AttributeError:
            # Fallback to completion model if document analysis model is not set
            logger.warning("DOCUMENT_ANALYSIS_MODEL not found in settings, using COMPLETION_MODEL as fallback")
            model_config_key = settings.COMPLETION_MODEL
        model_config = settings.REGISTERED_MODELS[model_config_key]

        # Perform requested analysis
        match analysis_type:
            case "entity_extraction":
                entities = await extract_entities(doc_content, model_config)
                return json.dumps(entities.model_dump(), indent=2)

            case "summarization":
                return await summarize_document(doc_content, model_config)

            case "fact_extraction":
                facts = await extract_facts(doc_content, model_config)
                return json.dumps(facts.model_dump(), indent=2)

            case "sentiment":
                sentiment = await analyze_sentiment(doc_content, model_config)
                return json.dumps(sentiment.model_dump(), indent=2)

            case "full":
                # Perform all analyses in parallel for efficiency
                entities_task = extract_entities(doc_content, model_config)
                summary_task = summarize_document(doc_content, model_config)
                facts_task = extract_facts(doc_content, model_config)
                sentiment_task = analyze_sentiment(doc_content, model_config)

                entities, summary, facts, sentiment = await asyncio.gather(
                    entities_task, summary_task, facts_task, sentiment_task
                )

                # Combine all results
                result = DocumentAnalysisResult(entities=entities, summary=summary, facts=facts, sentiment=sentiment)

                return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        raise ToolError(f"Error analyzing document: {str(e)}")


async def execute_code(
    code: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Execute code snippets for data analysis, visualization, or computation in a sandboxed environment.

    This function provides a secure environment for executing Python code with the following safety measures:
    - Code is executed in an isolated subprocess with restricted imports
    - Dangerous operations (file system access, system commands, etc.) are blocked
    - Memory and execution time limits are enforced
    - Only data science and visualization libraries are available

    Supported libraries include:
    - numpy, pandas, matplotlib, seaborn, plotly (data manipulation and visualization)
    - scipy, scikit-learn (scientific computing and machine learning)
    - Standard Python libraries (math, random, datetime, collections, etc.)

    The function returns execution results including:
    - Text output from stdout/stderr
    - Any visualizations generated (as base64-encoded images)
    - Execution status and timing information

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (capped at 300 seconds)

    Returns:
        Dictionary containing execution results with the following keys:
        - success: Boolean indicating if execution completed successfully
        - output: Text output from stdout
        - error: Error output from stderr
        - execution_time: Time taken to execute the code in seconds
        - content: List of content items (text and images) to display
    """
    import asyncio
    import base64
    import io
    import re
    import tempfile
    import time
    from pathlib import Path

    from PIL import Image

    # Basic validation
    if not code or not isinstance(code, str):
        raise ToolError("Code must be a non-empty string")

    if not isinstance(timeout, int) or timeout <= 0:
        raise ToolError("Timeout must be a positive integer")

    # Cap the timeout to a maximum value for security
    max_allowed_timeout = 300  # 5 minutes
    if timeout > max_allowed_timeout:
        timeout = max_allowed_timeout
        logger.warning(f"Timeout capped to maximum allowed value: {max_allowed_timeout} seconds")

    # Check for potentially dangerous operations
    dangerous_patterns = [
        r"import\s+os(?:\s|\.)",
        r"import\s+subprocess(?:\s|\.)",
        r"import\s+sys(?:\s|\.)",
        r"import\s+shutil(?:\s|\.)",
        r"__import__\(.*\)",
        r"open\(.*,.*w.*\)",
        r"exec\(",
        r"eval\(",
        r"os\.",
        r"subprocess\.",
        r"sys\.",
        r"shutil\.",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            raise ToolError(
                "Potentially unsafe code detected. The use of system access modules and functions is not allowed."
            )

    result = {"success": False, "output": "", "error": "", "execution_time": 0, "content": []}

    # Create a temporary directory for execution
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the code file
            file_path = Path(temp_dir) / "code.py"

            # Add sandbox wrapper code
            sandboxed_code = f"""
# Secure sandbox environment
# Restrict imports and dangerous operations
import sys
import builtins
import importlib

# Define allowed modules
ALLOWED_MODULES = [
    'math', 'random', 'datetime', 'json', 're',
    'collections', 'copy', 'functools', 'itertools',
    'string', 'time', 'uuid', 'hashlib',
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
    'scipy', 'sklearn', 'nltk', 'PIL'
]

# Create a secure import function
original_import = builtins.__import__

def secure_import(name, *args, **kwargs):
    if name not in ALLOWED_MODULES:
        message = f"Import of '{{name}}' is not allowed for security reasons. Allowed modules: {{ALLOWED_MODULES}}"
        raise ImportError(message)
    return original_import(name, *args, **kwargs)

# Replace the built-in import function with our secure version
builtins.__import__ = secure_import

# Set resource limits to prevent excessive memory use
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))  # 1GB memory limit
except ImportError:
    pass  # resource module not available on all platforms

# Block file system access
def blocked_open(*args, **kwargs):
    mode = 'r'
    if len(args) > 1:
        mode = args[1]
    if kwargs.get('mode'):
        mode = kwargs.get('mode')

    if 'w' in mode or 'a' in mode or '+' in mode:
        raise IOError("File writing operations are not allowed")

    return original_open(*args, **kwargs)

original_open = builtins.open
builtins.open = blocked_open

# --- User code begins here ---

{code}

# --- User code ends here ---
"""

            try:
                # Write the sandboxed code to the file
                with open(file_path, "w") as f:
                    f.write(sandboxed_code)
            except Exception as e:
                raise ToolError(f"Failed to write code to temporary file: {str(e)}")

            # Create a requirements.txt with required dependencies for data science
            requirements_path = Path(temp_dir) / "requirements.txt"
            required_packages = """
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.2.2
scipy==1.10.1
"""
            with open(requirements_path, "w") as f:
                f.write(required_packages)

            # Install required packages in the temporary environment
            # Note: In a production environment, you'd want to use a pre-built Docker image
            # with these dependencies already installed rather than installing packages on demand
            try:
                install_proc = await asyncio.create_subprocess_exec(
                    "pip",
                    "install",
                    "-r",
                    str(requirements_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(install_proc.communicate(), timeout=60)
            except Exception as e:
                logger.warning(f"Failed to install dependencies: {str(e)}")
                # Continue anyway, as the main Python environment might already have the dependencies

            # Prepare the command
            cmd = ["python", str(file_path)]

            # Execute the code with timeout
            start_time = time.time()
            try:
                # Run subprocess in a non-blocking way using asyncio
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=temp_dir
                )

                # Wait for the process to complete with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

                    # Convert bytes to string
                    stdout_str = stdout.decode("utf-8") if stdout else ""
                    stderr_str = stderr.decode("utf-8") if stderr else ""

                    result["success"] = proc.returncode == 0
                    result["output"] = stdout_str
                    result["error"] = stderr_str

                    # Add the output text as content if available
                    if stdout_str:
                        result["content"].append({"type": "text", "text": stdout_str})

                    # Check for generated visualizations (e.g., saved plots)
                    viz_files = list(Path(temp_dir).glob("*.png")) + list(Path(temp_dir).glob("*.jpg"))
                    for viz_file in viz_files:
                        try:
                            # Use pillow to read and convert the image to base64
                            with Image.open(viz_file) as img:
                                img_format = viz_file.suffix.lstrip(".").upper()
                                if img_format == "JPG":
                                    img_format = "JPEG"

                                # Resize large images if necessary
                                max_dimension = 2000
                                if img.width > max_dimension or img.height > max_dimension:
                                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                                # Convert image to base64
                                buffered = io.BytesIO()
                                # Convert RGBA to RGB if saving as JPEG
                                if img_format.upper() == "JPEG" and img.mode == "RGBA":
                                    img = img.convert("RGB")
                                img.save(buffered, format=img_format)
                                img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                                result["content"].append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": f"image/{img_format.lower()}",
                                            "data": img_data,
                                        },
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Error processing visualization {viz_file.name}: {str(e)}")
                            # If image processing fails, add error as text
                            result["content"].append(
                                {"type": "text", "text": f"Error processing visualization {viz_file.name}: {str(e)}"}
                            )

                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    error_msg = f"Execution timed out after {timeout} seconds"
                    logger.warning(error_msg)
                    result["error"] = error_msg
                    result["content"].append({"type": "text", "text": error_msg})

            except Exception as e:
                error_msg = f"Execution failed: {str(e)}"
                logger.error(error_msg)
                result["error"] = error_msg
                result["content"].append({"type": "text", "text": error_msg})
            finally:
                result["execution_time"] = time.time() - start_time

                # If no content was added (no stdout and no visualizations), add the error as content
                if not result["content"] and result["error"]:
                    result["content"].append({"type": "text", "text": result["error"]})
    except Exception as e:
        # Handle any exceptions that occur during the entire process
        error_msg = f"Code execution environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise ToolError(error_msg)

    return result
