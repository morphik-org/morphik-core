import uvicorn
import argparse
from dotenv import load_dotenv
from core.config import get_settings
from core.logging_config import setup_logging


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the DataBridge server")
    parser.add_argument(
        "--log",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set up logging first with specified level
    setup_logging(log_level=args.log.upper())

    # Load environment variables from .env file
    load_dotenv()

    # Load settings (this will validate all required env vars)
    settings = get_settings()

    # Start server
    uvicorn.run(
        "core.api:app",
        host=settings.HOST,
        port=settings.PORT,
        loop="asyncio",
        log_level=args.log,
        # reload=settings.RELOAD
    )


if __name__ == "__main__":
    main()
