import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_MAX_BYTES = 100 * 1024 * 1024
LOG_BACKUP_COUNT = 10


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Convert string to logging level
    level = getattr(logging, log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Check if handlers already exist - if so, skip setup
    if root_logger.handlers:
        # Update existing handlers' levels if needed
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # File handler with rotation to keep disk usage bounded
    file_handler = RotatingFileHandler(
        log_dir / "morphik.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(console_formatter)
    file_handler.setLevel(level)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set levels for specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    # Set debug level for core code to match root logger level
    logging.getLogger("core").setLevel(level)

    # Silence LiteLLM logs to prevent noisy output
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Silence telemetry logs to prevent noisy output
    logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.exporter.otlp.proto.http.metric_exporter").setLevel(logging.CRITICAL)
