from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from collections import defaultdict
import time
from contextlib import asynccontextmanager
import os
import json
from pathlib import Path

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    MetricExporter,
    AggregationTemporality,
    MetricsData,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter


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


@dataclass
class UsageRecord:
    timestamp: datetime
    operation_type: str
    tokens_used: int
    user_id: str
    duration_ms: float
    status: str
    metadata: Optional[Dict] = None


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
        self._usage_records: List[UsageRecord] = []
        self._user_totals = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

        # Initialize OpenTelemetry
        resource = Resource.create({"service.name": "databridge-core"})

        # Create logs directory
        log_dir = Path("logs/telemetry")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracing
        tracer_provider = TracerProvider(resource=resource)

        # Use file exporter for local development
        if os.getenv("ENVIRONMENT", "development") == "development":
            span_processor = BatchSpanProcessor(FileSpanExporter(str(log_dir)))
        else:
            span_processor = BatchSpanProcessor(OTLPSpanExporter())

        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        # Initialize metrics
        if os.getenv("ENVIRONMENT", "development") == "development":
            metric_reader = PeriodicExportingMetricReader(
                FileMetricExporter(str(log_dir)),
                export_interval_millis=60000,  # Export every minute
            )
        else:
            metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())

        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
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

    @asynccontextmanager
    async def track_operation(
        self,
        operation_type: str,
        user_id: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking operations with both usage metrics and OpenTelemetry
        """
        start_time = time.time()
        status = "success"
        current_span = trace.get_current_span()

        try:
            # Add operation attributes to the current span
            current_span.set_attribute("operation.type", operation_type)
            current_span.set_attribute("user.id", user_id)
            if metadata:
                for key, value in metadata.items():
                    current_span.set_attribute(f"metadata.{key}", str(value))

            yield current_span

        except Exception as e:
            status = "error"
            current_span.set_status(Status(StatusCode.ERROR))
            current_span.record_exception(e)
            raise
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Record metrics
            self.operation_counter.add(1, {"operation": operation_type, "status": status})
            if tokens_used > 0:
                self.token_counter.add(tokens_used, {"operation": operation_type})
            self.operation_duration.record(duration, {"operation": operation_type})

            # Record usage
            record = UsageRecord(
                timestamp=datetime.now(),
                operation_type=operation_type,
                tokens_used=tokens_used,
                user_id=user_id,
                duration_ms=duration,
                status=status,
                metadata=metadata,
            )

            with self._lock:
                self._usage_records.append(record)
                self._user_totals[user_id][operation_type] += tokens_used

    def get_user_usage(self, user_id: str) -> Dict[str, int]:
        """Get usage statistics for a user."""
        with self._lock:
            return dict(self._user_totals[user_id])

    def get_recent_usage(
        self,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Get recent usage records with optional filtering."""
        with self._lock:
            records = self._usage_records.copy()

        # Apply filters
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if operation_type:
            records = [r for r in records if r.operation_type == operation_type]
        if since:
            records = [r for r in records if r.timestamp >= since]
        if status:
            records = [r for r in records if r.status == status]

        return records
