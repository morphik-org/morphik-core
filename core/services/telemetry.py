import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import requests
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from core.config import get_settings

# Get settings from config
settings = get_settings()

# Telemetry configuration - use settings directly from TOML
TELEMETRY_ENABLED = settings.TELEMETRY_ENABLED
HONEYCOMB_ENABLED = settings.HONEYCOMB_ENABLED

# Honeycomb configuration - using proxy to avoid exposing API key in code
# Default to localhost:8080 for the proxy, but allow override from settings
HONEYCOMB_PROXY_ENDPOINT = getattr(settings, "HONEYCOMB_PROXY_ENDPOINT", "https://otel-proxy.onrender.com")
HONEYCOMB_PROXY_ENDPOINT = (
    HONEYCOMB_PROXY_ENDPOINT
    if isinstance(HONEYCOMB_PROXY_ENDPOINT, str) and len(HONEYCOMB_PROXY_ENDPOINT) > 0
    else "https://otel-proxy.onrender.com"
)
SERVICE_NAME = settings.SERVICE_NAME

# Headers for OTLP - no API key needed as the proxy will add it
OTLP_HEADERS = {"Content-Type": "application/x-protobuf"}

# Configure timeouts and retries directly from TOML config
OTLP_TIMEOUT = settings.OTLP_TIMEOUT
OTLP_MAX_RETRIES = settings.OTLP_MAX_RETRIES
OTLP_RETRY_DELAY = settings.OTLP_RETRY_DELAY
OTLP_MAX_EXPORT_BATCH_SIZE = settings.OTLP_MAX_EXPORT_BATCH_SIZE
OTLP_SCHEDULE_DELAY_MILLIS = settings.OTLP_SCHEDULE_DELAY_MILLIS
OTLP_MAX_QUEUE_SIZE = settings.OTLP_MAX_QUEUE_SIZE

# OTLP endpoints - using our proxy instead of direct Honeycomb connection
OTLP_TRACES_ENDPOINT = f"{HONEYCOMB_PROXY_ENDPOINT}/v1/traces"
OTLP_METRICS_ENDPOINT = f"{HONEYCOMB_PROXY_ENDPOINT}/v1/metrics"

# Enable debug logging for OpenTelemetry
os.environ["OTEL_PYTHON_LOGGING_LEVEL"] = "INFO"  # Changed from DEBUG to reduce verbosity
# Add export protocol setting if not already set
if not os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL"):
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"


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


class FileSpanExporter:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.log_dir / "traces.log"

    def export(self, spans):
        with open(self.trace_file, "a") as f:
            for span in spans:
                f.write(json.dumps(self._format_span(span)) + "\n")
        return True

    def shutdown(self):
        pass

    def _format_span(self, span):
        return {
            "name": span.name,
            "trace_id": format(span.context.trace_id, "x"),
            "span_id": format(span.context.span_id, "x"),
            "parent_id": format(span.parent.span_id, "x") if span.parent else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "attributes": dict(span.attributes),
            "status": span.status.status_code.name,
        }


class FileMetricExporter(MetricExporter):
    """File metric exporter for OpenTelemetry."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.log"
        super().__init__()

    def export(self, metrics_data: MetricsData, **kwargs) -> bool:
        """Export metrics data to a file.

        Args:
            metrics_data: The metrics data to export.

        Returns:
            True if the export was successful, False otherwise.
        """
        try:
            with open(self.metrics_file, "a") as f:
                for resource_metrics in metrics_data.resource_metrics:
                    for scope_metrics in resource_metrics.scope_metrics:
                        for metric in scope_metrics.metrics:
                            f.write(json.dumps(self._format_metric(metric)) + "\n")
            return True
        except Exception:
            return False

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> bool:
        """Shuts down the exporter.

        Args:
            timeout_millis: Time to wait for the export to complete in milliseconds.

        Returns:
            True if the shutdown succeeded, False otherwise.
        """
        return True

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Force flush the exporter.

        Args:
            timeout_millis: Time to wait for the flush to complete in milliseconds.

        Returns:
            True if the flush succeeded, False otherwise.
        """
        return True

    def _preferred_temporality(self) -> Dict:
        """Returns the preferred temporality for each instrument kind."""
        return {
            "counter": AggregationTemporality.CUMULATIVE,
            "up_down_counter": AggregationTemporality.CUMULATIVE,
            "observable_counter": AggregationTemporality.CUMULATIVE,
            "observable_up_down_counter": AggregationTemporality.CUMULATIVE,
            "histogram": AggregationTemporality.CUMULATIVE,
            "observable_gauge": AggregationTemporality.CUMULATIVE,
        }

    def _format_metric(self, metric):
        return {
            "name": metric.name,
            "description": metric.description,
            "unit": metric.unit,
            "data": self._format_data(metric.data),
        }

    def _format_data(self, data):
        if hasattr(data, "data_points"):
            return {
                "data_points": [
                    {
                        "attributes": dict(point.attributes),
                        "value": point.value if hasattr(point, "value") else None,
                        "count": point.count if hasattr(point, "count") else None,
                        "sum": point.sum if hasattr(point, "sum") else None,
                        "timestamp": point.time_unix_nano,
                    }
                    for point in data.data_points
                ]
            }
        return {}


class RetryingOTLPMetricExporter(MetricExporter):
    """A wrapper around OTLPMetricExporter that adds better retry logic."""

    def __init__(self, endpoint, headers=None, timeout=10):
        self.exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers, timeout=timeout)
        self.max_retries = OTLP_MAX_RETRIES
        self.retry_delay = OTLP_RETRY_DELAY
        self.logger = logging.getLogger(__name__)
        super().__init__()

    def export(self, metrics_data, **kwargs):
        """Export metrics with retry logic for handling connection issues."""
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.exporter.export(metrics_data, **kwargs)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ProtocolError,
                ReadTimeoutError,
            ):
                retries += 1

                if retries <= self.max_retries:
                    # Use exponential backoff
                    delay = self.retry_delay * (2 ** (retries - 1))
                    # self.logger.warning(
                    #     f"Honeycomb export attempt {retries} failed: {str(e)}. "
                    #     f"Retrying in {delay}s..."
                    # )
                    time.sleep(delay)
                # else:
                # self.logger.error(
                #     f"Failed to export to Honeycomb after {retries} attempts: {str(e)}"
                # )
            except Exception:
                # For non-connection errors, don't retry
                # self.logger.error(f"Unexpected error exporting to Honeycomb: {str(e)}")
                return False

        # If we get here, all retries failed
        return False

    def shutdown(self, timeout_millis=30000, **kwargs):
        """Shutdown the exporter."""
        return self.exporter.shutdown(timeout_millis, **kwargs)

    def force_flush(self, timeout_millis=10000):
        """Force flush the exporter."""
        return self.exporter.force_flush(timeout_millis)

    def _preferred_temporality(self):
        """Returns the preferred temporality."""
        return self.exporter._preferred_temporality()


class RetryingOTLPSpanExporter:
    """A wrapper around OTLPSpanExporter that adds better retry logic."""

    def __init__(self, endpoint, headers=None, timeout=10):
        self.exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=timeout)
        self.max_retries = OTLP_MAX_RETRIES
        self.retry_delay = OTLP_RETRY_DELAY
        self.logger = logging.getLogger(__name__)

    def export(self, spans):
        """Export spans with retry logic for handling connection issues."""
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.exporter.export(spans)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ProtocolError,
                ReadTimeoutError,
            ) as e:
                retries += 1

                if retries <= self.max_retries:
                    # Use exponential backoff
                    delay = self.retry_delay * (2 ** (retries - 1))
                    self.logger.warning(
                        f"Honeycomb trace export attempt {retries} failed: {str(e)}. " f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to export traces to Honeycomb after {retries} attempts: {str(e)}")
            except Exception as e:
                # For non-connection errors, don't retry
                self.logger.error(f"Unexpected error exporting traces to Honeycomb: {str(e)}")
                return False

        # If we get here, all retries failed
        return False

    def shutdown(self):
        """Shutdown the exporter."""
        return self.exporter.shutdown()

    def force_flush(self):
        """Force flush the exporter."""
        try:
            return self.exporter.force_flush()
        except Exception as e:
            self.logger.error(f"Error during trace force_flush: {str(e)}")
            return False


@dataclass
class UsageRecord:
    timestamp: datetime
    operation_type: str
    tokens_used: int
    user_id: str
    app_id: str | None
    duration_ms: float
    status: str
    error: str | None = None
    metadata: Optional[Dict] = None


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

    def __init__(self, fields: List[MetadataField] = None):
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

        if not TELEMETRY_ENABLED:
            return

        self._usage_records: List[UsageRecord] = []
        self._user_totals = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
        self._installation_id = get_installation_id()

        # Initialize OpenTelemetry with more detailed resource attributes
        resource = Resource.create(
            {
                "service.name": SERVICE_NAME,
                "service.version": os.getenv("DATABRIDGE_VERSION", "unknown"),
                "installation.id": self._installation_id,
                "environment": os.getenv("ENVIRONMENT", "production"),
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": "1.0.0",
            }
        )

        # Initialize tracing with both file and OTLP exporters
        tracer_provider = TracerProvider(resource=resource)

        # Always use both exporters
        log_dir = Path("logs/telemetry")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file exporter for local logging
        file_span_processor = BatchSpanProcessor(FileSpanExporter(str(log_dir)))
        tracer_provider.add_span_processor(file_span_processor)

        # Add Honeycomb OTLP exporter with retry logic
        if HONEYCOMB_ENABLED:
            # Create BatchSpanProcessor with improved configuration
            otlp_span_processor = BatchSpanProcessor(
                RetryingOTLPSpanExporter(
                    endpoint=OTLP_TRACES_ENDPOINT,
                    headers=OTLP_HEADERS,
                    timeout=OTLP_TIMEOUT,
                ),
                # Configure batch processing settings
                max_queue_size=OTLP_MAX_QUEUE_SIZE,
                max_export_batch_size=OTLP_MAX_EXPORT_BATCH_SIZE,
                schedule_delay_millis=OTLP_SCHEDULE_DELAY_MILLIS,
            )
            tracer_provider.add_span_processor(otlp_span_processor)

        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        # Initialize metrics with both exporters
        metric_readers = [
            # Local file metrics reader
            PeriodicExportingMetricReader(
                FileMetricExporter(str(log_dir)),
                export_interval_millis=60000,  # Export every minute
            ),
        ]

        # Add Honeycomb metrics reader if API key is available
        if HONEYCOMB_ENABLED:
            try:
                # Configure the OTLP metric exporter with improved error handling
                otlp_metric_exporter = RetryingOTLPMetricExporter(
                    endpoint=OTLP_METRICS_ENDPOINT,
                    headers=OTLP_HEADERS,
                    timeout=OTLP_TIMEOUT,
                )

                # Configure the metrics reader with improved settings
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        otlp_metric_exporter,
                        export_interval_millis=OTLP_SCHEDULE_DELAY_MILLIS,
                        export_timeout_millis=OTLP_TIMEOUT * 1000,
                    )
                )
            except Exception as e:
                print(f"Failed to configure Honeycomb metrics exporter: {str(e)}")

        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)

        # Create metrics
        self.operation_counter = self.meter.create_counter(
            "databridge.operations",
            description="Number of operations performed",
        )
        self.token_counter = self.meter.create_counter(
            "databridge.tokens",
            description="Number of tokens processed",
        )
        self.operation_duration = self.meter.create_histogram(
            "databridge.operation.duration",
            description="Duration of operations",
            unit="ms",
        )

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
                MetadataField("folder_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.add_document_to_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs"),
                MetadataField("document_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.remove_document_from_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs"),
                MetadataField("document_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        self.delete_folder_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs"),
                MetadataField("user_id", "kwargs", "auth", transform=lambda auth: getattr(auth, "entity_id", None)),
            ]
        )
        # Set up all the metadata extractors
        self.ingest_text_metadata = MetadataExtractor(
            common_request_fields
            + [
                MetadataField("metadata", "request", default={}),
                MetadataField("rules", "request", default=[]),
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
                MetadataField("rules", "kwargs", transform=lambda v: parse_json(v, [])),
                MetadataField("use_colpali", "kwargs"),
                MetadataField("folder_name", "kwargs"),
                MetadataField("end_user_id", "kwargs"),
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
                MetadataField(
                    "rules_type",
                    "kwargs",
                    "rules",
                    transform=lambda v: (
                        "per_file"
                        if isinstance(parse_json(v, []), list)
                        and parse_json(v, [])
                        and isinstance(parse_json(v, [])[0], list)
                        else "shared"
                    ),
                ),
                MetadataField("folder_name", "kwargs"),
                MetadataField("end_user_id", "kwargs"),
            ]
        )

        self.retrieve_chunks_metadata = MetadataExtractor(retrieval_fields)
        self.retrieve_docs_metadata = MetadataExtractor(retrieval_fields)

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

        self.usage_stats_metadata = MetadataExtractor([])

        self.recent_usage_metadata = MetadataExtractor(
            [
                MetadataField("operation_type", "kwargs"),
                MetadataField("since", "kwargs", transform=lambda dt: dt.isoformat() if dt else None),
                MetadataField("status", "kwargs"),
            ]
        )

        self.cache_create_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
                MetadataField("model", "kwargs"),
                MetadataField("gguf_file", "kwargs"),
                MetadataField("filters", "kwargs"),
                MetadataField("docs", "kwargs"),
            ]
        )

        self.cache_get_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
            ]
        )

        self.cache_update_metadata = self.cache_get_metadata

        self.cache_add_docs_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
                MetadataField("docs", "kwargs"),
            ]
        )

        self.cache_query_metadata = MetadataExtractor(
            [
                MetadataField("name", "kwargs"),
                MetadataField("query", "kwargs"),
                MetadataField("max_tokens", "kwargs"),
                MetadataField("temperature", "kwargs"),
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

        self.set_folder_rule_metadata = MetadataExtractor(
            [
                MetadataField("folder_id", "kwargs"),
                MetadataField("apply_to_existing", "kwargs", default=True),
                MetadataField(
                    "rule_count",
                    "request",
                    "rules",
                    transform=lambda rules: len(rules) if hasattr(rules, "__len__") else 0,
                ),
                MetadataField(
                    "rule_types",
                    "request",
                    "rules",
                    transform=lambda rules: ([rule.type for rule in rules] if hasattr(rules, "__iter__") else []),
                ),
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
        """
        Context manager for tracking operations with both usage metrics and OpenTelemetry.
        The user_id is hashed to ensure anonymity.
        """
        if not TELEMETRY_ENABLED:
            yield None
            return

        start_time = time.time()
        error_msg: str | None = None
        status = "success"
        current_span = trace.get_current_span()

        try:
            # Add operation attributes to the current span
            current_span.set_attribute("operation.type", operation_type)
            current_span.set_attribute("user.id", user_id)
            if metadata:
                # Create a copy of metadata to avoid modifying the original
                metadata_copy = metadata.copy()

                # Remove the nested 'metadata' field completely if it exists
                if "metadata" in metadata_copy:
                    del metadata_copy["metadata"]

                # Set attributes for all remaining metadata fields
                for key, value in metadata_copy.items():
                    current_span.set_attribute(f"metadata.{key}", str(value))

            yield current_span

        except Exception as e:
            status = "error"
            error_msg = str(e)
            current_span.set_status(Status(StatusCode.ERROR))
            current_span.record_exception(e)
            raise
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Record metrics
            attributes = {
                "operation": operation_type,
                "status": status,
                "installation_id": self._installation_id,
            }
            if app_id:
                attributes["app_id"] = app_id
            self.operation_counter.add(1, attributes)
            if tokens_used > 0:
                self.token_counter.add(tokens_used, attributes)
            self.operation_duration.record(duration, attributes)

            # Record usage
            # Create a sanitized copy of metadata for the usage record
            sanitized_metadata = None
            if metadata:
                sanitized_metadata = metadata.copy()
                # Remove the nested 'metadata' field completely if it exists
                if "metadata" in sanitized_metadata:
                    del sanitized_metadata["metadata"]

            record = UsageRecord(
                timestamp=datetime.now(),
                operation_type=operation_type,
                tokens_used=tokens_used,
                user_id=user_id,
                app_id=app_id,
                duration_ms=duration,
                status=status,
                error=error_msg,
                metadata=sanitized_metadata,
            )

            # In-memory for backward compatibility
            with self._lock:
                self._usage_records.append(record)
                self._user_totals[user_id][operation_type] += tokens_used

            # Async insert into Postgres (fire and forget)
            try:
                import asyncio

                from core.database.logs_db import LogsDB

                async def _persist():
                    db = await LogsDB.get()
                    # Only persist key operations for now
                    if record.operation_type in {"query", "ingest_worker"}:
                        await db.insert(
                            {
                                "timestamp": record.timestamp,
                                "user_id": record.user_id,
                                "app_id": record.app_id or "unknown",
                                "operation_type": record.operation_type,
                                "status": record.status,
                                "duration_ms": record.duration_ms,
                                "tokens_used": record.tokens_used,
                                "metadata": record.metadata or {},
                                "error": record.error,
                            }
                        )

                asyncio.create_task(_persist())
            except Exception:  # pragma: no cover – never fail caller on logging error
                pass

    def get_user_usage(self, user_id: str) -> Dict[str, int]:
        """Get usage statistics for a user."""
        if not TELEMETRY_ENABLED:
            return {}

        with self._lock:
            return dict(self._user_totals[user_id])

    def get_recent_usage(
        self,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Get recent usage records with optional filtering."""
        if not TELEMETRY_ENABLED:
            return []

        with self._lock:
            records = self._usage_records.copy()

        # Apply filters
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if app_id:
            records = [r for r in records if r.app_id == app_id]
        if operation_type:
            records = [r for r in records if r.operation_type == operation_type]
        if since:
            records = [r for r in records if r.timestamp >= since]
        if status:
            records = [r for r in records if r.status == status]

        return records
