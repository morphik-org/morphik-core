import uvicorn
import argparse
import sys
import tomli
import requests
import logging
from dotenv import load_dotenv
from core.config import get_settings
from core.logging_config import setup_logging


def check_ollama_running(base_url):
    """Check if Ollama is running and accessible at the given URL."""
    try:
        api_url = f"{base_url}/api/tags"
        response = requests.get(api_url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_ollama_usage_info():
    """Check if Ollama is required based on the configuration file and get base URLs."""
    try:
        with open("databridge.toml", "rb") as f:
            config = tomli.load(f)

        ollama_configs = []

        # Check embedding provider
        if "embedding" in config and config["embedding"].get("provider") == "ollama":
            base_url = config["embedding"].get("base_url")
            if base_url:
                ollama_configs.append({"component": "embedding", "base_url": base_url})

        # Check completion provider
        if "completion" in config and config["completion"].get("provider") == "ollama":
            base_url = config["completion"].get("base_url")
            if base_url:
                ollama_configs.append({"component": "completion", "base_url": base_url})

        # Check rules provider
        if "rules" in config and config["rules"].get("provider") == "ollama":
            base_url = config["rules"].get("base_url")
            if base_url:
                ollama_configs.append({"component": "rules", "base_url": base_url})

        # Check graph provider
        if "graph" in config and config["graph"].get("provider") == "ollama":
            base_url = config["graph"].get("base_url")
            if base_url:
                ollama_configs.append({"component": "graph", "base_url": base_url})

        # Check parser.vision provider
        if (
            "parser" in config
            and "vision" in config["parser"]
            and config["parser"]["vision"].get("provider") == "ollama"
        ):
            base_url = config["parser"]["vision"].get("base_url")
            if base_url:
                ollama_configs.append({"component": "parser.vision", "base_url": base_url})

        return ollama_configs
    except Exception as e:
        logging.error(f"Error checking Ollama configuration: {e}")
        return []


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the DataBridge server")
    parser.add_argument(
        "--log",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip Ollama availability check",
    )
    args = parser.parse_args()

    # Set up logging first with specified level
    setup_logging(log_level=args.log.upper())

    # Load environment variables from .env file
    load_dotenv()

    # Check if Ollama is required and running
    if not args.skip_ollama_check:
        ollama_configs = get_ollama_usage_info()

        if ollama_configs:
            # Group configs by base_url to avoid redundant checks
            base_urls = {}
            for config in ollama_configs:
                if config["base_url"] not in base_urls:
                    base_urls[config["base_url"]] = []
                base_urls[config["base_url"]].append(config["component"])

            all_running = True
            for base_url, components in base_urls.items():
                if not check_ollama_running(base_url):
                    print(f"ERROR: Ollama is not accessible at {base_url}")
                    print(f"This URL is used by these components: {', '.join(components)}")
                    all_running = False

            if not all_running:
                print(
                    "\nPlease ensure Ollama is running at the configured URLs before starting the server"
                )
                print("Run with --skip-ollama-check to bypass this check")
                sys.exit(1)
            else:
                component_list = [config["component"] for config in ollama_configs]
                print(f"Ollama is running and will be used for: {', '.join(component_list)}")

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
