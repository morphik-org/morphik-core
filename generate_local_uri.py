from datetime import datetime, timedelta, UTC
import jwt
from dotenv import load_dotenv
import os
import argparse
import tomli

load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate a databridge URI")
parser.add_argument("--name", default="admin", help="Entity ID for the token (default: admin)")
parser.add_argument(
    "--expiry", default=30, help="Number of days the token is valid for (default: 30)"
)
args = parser.parse_args()
args.name = args.name.replace(" ", "_").lower()

# Get JWT secret from env
jwt_secret = os.getenv("JWT_SECRET_KEY")
if not jwt_secret:
    raise ValueError("JWT_SECRET_KEY not found in environment variables")

# Create payload
payload = {
    "type": "developer",
    "entity_id": args.name,
    "permissions": ["read", "write", "admin"],
    "exp": datetime.now(UTC) + timedelta(days=int(args.expiry)),
}

# Generate token using secret from .env
token = jwt.encode(payload, jwt_secret, algorithm="HS256")


with open("databridge.toml", "rb") as f:
    config = tomli.load(f)
base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")
entity_id = args.name
uri = f"databridge://{entity_id}:{token}@{base_url}"

# Print all components in color using ANSI escape codes
print(f"\033[94mEntity ID:\033[0m {entity_id}")  # Blue
print(f"\033[93mBase URL:\033[0m {base_url}")  # Yellow
print(f"\033[95mExpiry:\033[0m {payload['exp']}")  # Pink

print(f"\033[92mToken:\033[0m {token}")  # Green

print(f"\033[91mFull URI:\033[0m {uri}")  # Red
