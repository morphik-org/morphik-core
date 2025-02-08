import logging
import sys
from pathlib import Path


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

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_dir / "databridge.log")
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
