"""Morphik tests package."""

import logging


# Configure logging for tests
def setup_test_logging():
    """Set up logging configuration for tests to reduce noise.

    This function configures logging specifically for test runs to:
    - Suppress verbose logs from LiteLLM and other external libraries
    - Show only warnings and errors from these libraries
    - Keep INFO level for Morphik core components
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s:%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Set levels for specific loggers
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

    # Set debug level for core code
    logging.getLogger("core").setLevel(logging.INFO)


# Set up logging for test runs
setup_test_logging()
