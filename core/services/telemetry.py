import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ReadableSpan, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

from core.config import get_settings

# Get settings from config
settings = get_settings()

# Telemetry configuration - use settings directly from TOML
TELEMETRY_ENABLED = settings.TELEMETRY_ENABLED
SERVICE_NAME = settings.SERVICE_NAME
EVENT_SCHEMA_VERSION = 1
METADATA_MAX_LENGTH = 256
SENSITIVE_METADATA_KEYS = {"metadata", "request_dump", "request_body"}
REDACTED_METADATA_KEYS = {"query", "folder_name", "folder_path", "full_path"}

# Enable debug logging for OpenTelemetry
os.environ["OTEL_PYTHON_LOGGING_LEVEL"] = "INFO"  # Changed from DEBUG to reduce verbosity


def get_installation_id() -> str:
    """Generate or retrieve a unique anonymous installation ID."""
    id_file = Path.home() / ".databridge" / "installation_id"
    id_file.parent.mkdir(parents=True, exist_ok=True)

    if id_file.exists():
        return id_file.read_text().strip()

    # Generate a new installation ID
    # We hash the machine-id (if available) or a random UUID
    machine_id_file = Path("/etc/machine-id")
    if machine_id_file.exists():
        machine_id = machine_id_file.read_text().strip()
    else:
        machine_id = str(uuid.uuid4())

    # Hash the machine ID to make it anonymous
    installation_id = hashlib.sha256(machine_id.encode()).hexdigest()[:32]

    # Save it for future use
    id_file.write_text(installation_id)
    return installation_id


def _truncate_metadata_value(value: str) -> str:
    if len(value) <= METADATA_MAX_LENGTH:
        return value
    return value[:METADATA_MAX_LENGTH]


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop sensitive metadata keys and normalize risky strings before emitting."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in SENSITIVE_METADATA_KEYS:
            continue
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            # Avoid serializing nested payloads that may contain customer data
            continue
        if key in REDACTED_METADATA_KEYS:
            if isinstance(value, str):
                sanitized[key] = f"redacted:{key}"
            else:
                sanitized[key] = "redacted"
            continue
        if isinstance(value, str):
            sanitized[key] = _truncate_metadata_value(value)
            continue
        sanitized[key] = value
    return sanitized


class JSONLSpanExporter(SpanExporter):
    """Custom span exporter that writes per-operation events to JSONL files."""

    def __init__(self, log_dir: Path, installation_id: str, schema_version: int = EVENT_SCHEMA_VERSION):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.installation_id = installation_id
        self.schema_version = schema_version
        self.worker_pid = os.getpid()
        self.log_file = self.log_dir / f"usage_events_worker_{self.worker_pid}.jsonl"
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        events = []
        for span in spans:
            event = self._span_to_event(span)
            if event:
                events.append(event)

        if not events:
            return SpanExportResult.SUCCESS

        try:
            payload = "".join(json.dumps(event) + "\n" for event in events)
            with self._lock:
                with self.log_file.open("a", encoding="utf-8") as fh:
                    fh.write(payload)
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.warning("Failed to write telemetry events: %s", exc)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Nothing to clean up; required by interface."""

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        """No-op flush hook for compatibility."""
        return True

    def _span_to_event(self, span: ReadableSpan) -> Optional[Dict[str, Any]]:
        """Convert a span into a structured event we can persist."""
        attrs = span.attributes or {}
        operation_type = attrs.get("operation.type")
        if not operation_type:
            return None

        metadata: Dict[str, Any] = {}
        for key, value in attrs.items():
            if key.startswith("metadata."):
                metadata[key.split("metadata.", 1)[1]] = value

        # Guard against None times (span not properly ended)
        if span.start_time is None or span.end_time is None:
            return None

        timestamp = datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=timezone.utc)
        duration_ms = max((span.end_time - span.start_time) / 1_000_000, 0.0)
        status = attrs.get("operation.status") or (
            "error" if span.status.status_code is StatusCode.ERROR else "success"
        )
        error_message = attrs.get("error.message") or span.status.description

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return {
            "schema_version": self.schema_version,
            "timestamp": timestamp.isoformat(),
            "installation_id": self.installation_id,
            "operation_type": operation_type,
            "status": status,
            "duration_ms": duration_ms,
            "user_id": attrs.get("user.id"),
            "app_id": attrs.get("app.id"),
            "tokens_used": _as_int(attrs.get("operation.tokens", 0)),
            "metadata": metadata or None,
            "error": error_message,
            "trace_id": f"{span.context.trace_id:032x}",
            "span_id": f"{span.context.span_id:016x}",
            "worker_pid": self.worker_pid,
        }


# Type variable for function return type
T = TypeVar("T")


class MetadataField:
    """Defines a metadata field to extract and how to extract it."""

    def __init__(
        self,
        key: str,
        source: str,
        attr_name: Optional[str] = None,
        default: Any = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize a metadata field definition.

        Args:
            key: The key to use in the metadata dictionary
            source: The source of the data ('request', 'kwargs', etc.)
            attr_name: The attribute name to extract (if None, uses key)
            default: Default value if not found
            transform: Optional function to transform the extracted value
        """
        self.key = key
        self.source = source
        self.attr_name = attr_name or key
        self.default = default
        self.transform = transform

    def extract(self, args: tuple, kwargs: dict) -> Any:
        """Extract the field value from args/kwargs based on configuration."""
        value = self.default

        if self.source == "kwargs":
            value = kwargs.get(self.attr_name, self.default)
        elif self.source == "request":
            request = kwargs.get("request")
            if request:
                if hasattr(request, "get") and callable(request.get):
                    value = request.get(self.attr_name, self.default)
                else:
                    value = getattr(request, self.attr_name, self.default)

        if self.transform and value is not None:
            value = self.transform(value)

        return value


class MetadataExtractor:
    """Base class for metadata extractors with common functionality."""

    def __init__(self, fields: Optional[List[MetadataField]] = None):
        """Initialize with a list of field definitions."""
        self.fields = fields or []

    def extract(self, args: tuple, kwargs: dict) -> dict:
        """Extract metadata using the field definitions."""
        metadata = {}

        for field in self.fields:
            value = field.extract(args, kwargs)
            if value is not None:  # Only include non-None values
                metadata[field.key] = value

        return metadata

    def __call__(self, *args, **kwargs) -> dict:
        """Make the extractor callable as an instance method."""
        # If called as an instance method, the first arg will be the instance
        # which we don't need for extraction, so we slice it off if there are any args
        actual_args = args[1:] if len(args) > 0 else ()
        return self.extract(actual_args, kwargs)


# Common transforms and utilities for metadata extraction
def parse_json(value, default=None):
    """Parse a JSON string safely, returning default on error."""
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def get_json_type(value):
    """Determine if a JSON value is a list or single object."""
    return "list" if isinstance(value, list) else "single"


def get_list_len(value, default=0):
    """Get the length of a list safely."""
    if value and isinstance(value, list):
        return len(value)
    return default


def is_not_none(value):
    """Check if a value is not None."""
    return value is not None


class TelemetryService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Initialize metadata extractors
        self._setup_metadata_extractors()

        self._log_dir = Path("logs/telemetry")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Always provide a tracer handle so decorators function even if telemetry is disabled.
        self.tracer = trace.get_tracer(__name__)
        self._installation_id: Optional[str] = None

        if not TELEMETRY_ENABLED:
            return

        self._installation_id = get_installation_id()

        # Initialize OpenTelemetry with more detailed resource attributes
        resource = Resource.create(
            {
                "service.name": SERVICE_NAME,
                "service.version": settings.VERSION,
                "installation.id": self._installation_id,
                "environment": settings.ENVIRONMENT,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": "1.0.0",
            }
        )

        # Initialize tracing with the JSONL exporter
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(JSONLSpanExporter(self._log_dir, self._installation_id)))

        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def _setup_metadata_extractors(self):
        """Set up all the metadata extractors with their field definitions."""
        # Common fields that appear in many requests
        common_request_fields = [
            MetadataField("use_colpali", "request"),
            MetadataField("folder_name", "request"),
            MetadataField("end_user_id", "request"),
        ]

        # For retrieval & query we want the same field names as the incoming request
        retrieval_fields = common_request_fields + [
            MetadataField("query", "request", "query"),
            MetadataField("k", "request"),
            MetadataField("min_score", "request"),
            MetadataField("use_reranking", "request"),
        ]

        # Folder operation metadata extractors
        self.create_folder_metadata = MetadataExtractor(
            [
                MetadataField("name", "request"),
                MetadataField("description", "request"),
                MetadataField("owner_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.list_folders_metadata = MetadataExtractor(
            [
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.get_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs", attr_name="folder_id_or_name"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.add_document_to_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs", attr_name="folder_id_or_name"),
                MetadataField("document_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.remove_document_from_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs", attr_name="folder_id_or_name"),
                MetadataField("document_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.delete_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs", attr_name="folder_id_or_name"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        # Set up all the metadata extractors
        self.ingest_text_metadata = MetadataExtractor(
            common_request_fields
            + [
                MetadataField("metadata", "request", default={}),
            ]
        )

        self.ingest_file_metadata = MetadataExtractor(
            [
                MetadataField("filename", "kwargs", transform=lambda file: file.filename if file else None),
                MetadataField(
                    "content_type",
                    "kwargs",
                    transform=lambda file: file.content_type if file else None,
                ),
                MetadataField("metadata", "kwargs", transform=lambda v: parse_json(v, {})),
                MetadataField(
                    "metadata",
                    "kwargs",
                    "ingestion_options",
                    transform=lambda v: parse_json(v, {}).get("metadata", {}),
                ),
                MetadataField("use_colpali", "kwargs"),
                MetadataField(
                    "use_colpali",
                    "kwargs",
                    "ingestion_options",
                    transform=lambda v: parse_json(v, {}).get("use_colpali"),
                ),
                MetadataField("folder_name", "kwargs"),
                MetadataField(
                    "folder_name",
                    "kwargs",
                    "ingestion_options",
                    transform=lambda v: parse_json(v, {}).get("folder_name"),
                ),
                MetadataField("end_user_id", "kwargs"),
                MetadataField(
                    "end_user_id",
                    "kwargs",
                    "ingestion_options",
                    transform=lambda v: parse_json(v, {}).get("end_user_id"),
                ),
            ]
        )

        self.batch_ingest_metadata = MetadataExtractor(
            [
                MetadataField("file_count", "kwargs", "files", transform=get_list_len),
                MetadataField(
                    "metadata_type",
                    "kwargs",
                    "metadata",
                    transform=lambda v: get_json_type(parse_json(v, {})),
                ),
                MetadataField("folder_name", "kwargs"),
                MetadataField("end_user_id", "kwargs"),
            ]
        )

        self.retrieve_chunks_metadata = MetadataExtractor(retrieval_fields)
        self.retrieve_docs_metadata = MetadataExtractor(retrieval_fields)
        self.search_documents_metadata = MetadataExtractor(
            [
                MetadataField("query", "request"),
                MetadataField("limit", "request"),
                MetadataField("filters", "request"),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
            ]
        )

        self.batch_documents_metadata = MetadataExtractor(
            [
                MetadataField(
                    "document_count",
                    "request",
                    transform=lambda req: len(req.get("document_ids", [])) if req else 0,
                ),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
            ]
        )

        self.batch_chunks_metadata = MetadataExtractor(
            [
                MetadataField(
                    "chunk_count",
                    "request",
                    transform=lambda req: len(req.get("sources", [])) if req else 0,
                ),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
                MetadataField("use_colpali", "request"),
            ]
        )

        # Completion / query operation – capture full parameter set
        self.query_metadata = MetadataExtractor(
            retrieval_fields
            + [
                MetadataField("max_tokens", "request"),
                MetadataField("temperature", "request"),
                MetadataField("graph_name", "request"),
                MetadataField("hop_depth", "request"),
                MetadataField("include_paths", "request"),
                MetadataField("schema", "request"),
                MetadataField("chat_id", "request"),
                MetadataField("use_colpali", "request"),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
                MetadataField("padding", "request"),
                MetadataField("inline_citations", "request"),
                # Capture which filter keys were supplied (no values)
                MetadataField(
                    "filter_keys",
                    "request",
                    "filters",
                    transform=lambda v: sorted(v.keys()) if isinstance(v, dict) else None,
                ),
                # Capture llm_config keys in hashed form to avoid leaking names/PII
                MetadataField(
                    "llm_config_hashed_keys",
                    "request",
                    "llm_config",
                    transform=lambda cfg: (
                        {hashlib.sha256(k.encode()).hexdigest()[:8]: type(v).__name__ for k, v in cfg.items()}
                        if isinstance(cfg, dict)
                        else None
                    ),
                ),
                MetadataField(
                    "has_prompt_overrides",
                    "request",
                    "prompt_overrides",
                    transform=lambda v: v is not None,
                ),
                MetadataField(
                    "filters_present",
                    "request",
                    "filters",
                    transform=lambda v: v is not None,
                ),
            ]
        )

        self.document_delete_metadata = MetadataExtractor(
            [
                MetadataField("document_id", "kwargs"),
            ]
        )

        self.document_update_text_metadata = MetadataExtractor(
            [
                MetadataField("document_id", "kwargs"),
                MetadataField("update_strategy", "kwargs", default="add"),
                MetadataField("use_colpali", "request"),
                MetadataField("has_filename", "request", "filename", transform=is_not_none),
            ]
        )

        self.document_update_file_metadata = MetadataExtractor(
            [
                MetadataField("document_id", "kwargs"),
                MetadataField("update_strategy", "kwargs", default="add"),
                MetadataField("use_colpali", "kwargs"),
                MetadataField("filename", "kwargs", transform=lambda file: file.filename if file else None),
                MetadataField(
                    "content_type",
                    "kwargs",
                    transform=lambda file: file.content_type if file else None,
                ),
            ]
        )

        self.document_update_metadata_resolver = MetadataExtractor(
            [
                MetadataField("document_id", "kwargs"),
            ]
        )

        self.create_graph_metadata = MetadataExtractor(
            [
                MetadataField("name", "request"),
                MetadataField("has_filters", "request", "filters", transform=is_not_none),
                MetadataField(
                    "document_count",
                    "request",
                    "documents",
                    transform=lambda docs: len(docs) if docs else 0,
                ),
                MetadataField("has_prompt_overrides", "request", "prompt_overrides", transform=is_not_none),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
            ]
        )

        self.get_graph_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
                MetadataField("folder_name", "kwargs"),
                MetadataField("end_user_id", "kwargs"),
            ]
        )

        self.list_graphs_metadata = MetadataExtractor(
            [
                MetadataField("folder_name", "kwargs"),
                MetadataField("end_user_id", "kwargs"),
            ]
        )

        self.update_graph_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
                MetadataField("has_additional_filters", "request", "additional_filters", transform=is_not_none),
                MetadataField(
                    "additional_document_count",
                    "request",
                    "additional_documents",
                    transform=lambda docs: len(docs) if docs else 0,
                ),
                MetadataField("has_prompt_overrides", "request", "prompt_overrides", transform=is_not_none),
                MetadataField("folder_name", "request"),
                MetadataField("end_user_id", "request"),
            ]
        )

        self.workflow_status_metadata = MetadataExtractor(
            [
                MetadataField("workflow_id", "kwargs"),
                MetadataField("run_id", "kwargs"),
            ]
        )

        # Utility: dump full request payload (excluding giant fields)
        def _safe_dump(req):
            try:
                d = req.dict() if hasattr(req, "dict") else req.model_dump()  # type: ignore[attr-defined]
            except Exception:
                d = None
            # Remove potentially large fields
            if d and "content" in d and isinstance(d["content"], str):
                d["content_len"] = len(d["content"])
                del d["content"]
            return d

        self.query_metadata.fields.append(MetadataField("request_dump", "request", transform=_safe_dump))

    def track(self, operation_type: Optional[str] = None, metadata_resolver: Optional[Callable] = None):
        """
        Decorator for tracking API operations with telemetry.

        Args:
            operation_type: Type of operation or function name if None
            metadata_resolver: Function that extracts metadata from the request/args/kwargs
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract auth from kwargs
                auth = kwargs.get("auth")
                if not auth:
                    # Try to find auth in positional arguments (unlikely, but possible)
                    for arg in args:
                        if hasattr(arg, "entity_id") and hasattr(arg, "permissions"):
                            auth = arg
                            break

                # If we don't have auth, we can't track the operation
                if not auth:
                    return await func(*args, **kwargs)

                # Use function name if operation_type not provided
                op_type = operation_type or func.__name__

                # Generate metadata using resolver or create empty dict
                meta = {}
                if metadata_resolver:
                    meta = metadata_resolver(*args, **kwargs)

                # Approximate token count for common request payloads
                tokens = 0
                request = kwargs.get("request")
                if request:
                    if hasattr(request, "content") and isinstance(request.content, str):
                        tokens = len(request.content.split())
                    elif hasattr(request, "query") and isinstance(request.query, str):
                        tokens = len(request.query.split())

                # Run the function within the telemetry context
                async with self.track_operation(
                    operation_type=op_type,
                    user_id=auth.entity_id,
                    app_id=getattr(auth, "app_id", None),
                    tokens_used=tokens,
                    metadata=meta,
                ):
                    # Call the original function
                    result = await func(*args, **kwargs)
                    return result

            return wrapper

        return decorator

    @asynccontextmanager
    async def track_operation(
        self,
        operation_type: str,
        user_id: str,
        app_id: Optional[str] = None,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking operations via OpenTelemetry spans."""
        if not TELEMETRY_ENABLED:
            yield None
            return

        metadata = sanitize_metadata(metadata or {})
        start_time = time.time()
        error_msg: Optional[str] = None
        status = "success"

        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            span_context = nullcontext(current_span)
        else:
            span_context = self.tracer.start_as_current_span(operation_type)

        span_ref = None

        try:
            with span_context as span:
                span_ref = span
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("user.id", user_id)
                span.set_attribute("operation.tokens", tokens_used)
                if self._installation_id:
                    span.set_attribute("installation.id", self._installation_id)
                if app_id:
                    span.set_attribute("app.id", app_id)

                metadata_copy = metadata.copy()
                for key, value in metadata_copy.items():
                    span.set_attribute(f"metadata.{key}", str(value))

                yield span

        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            if span_ref:
                span_ref.set_status(Status(StatusCode.ERROR))
                span_ref.record_exception(exc)
                span_ref.set_attribute("error.message", error_msg)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            if span_ref:
                span_ref.set_attribute("operation.status", status)
                span_ref.set_attribute("operation.duration_ms", duration_ms)
                if error_msg:
                    span_ref.set_attribute("error.message", error_msg)
