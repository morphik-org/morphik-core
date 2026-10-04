#!/usr/bin/env bash

# Morphik Core one-liner installer + server launcher.
# Works on macOS (Apple Silicon or Intel) and Linux.
# Usage:  bash install_and_start.sh
# Requires Docker when using the default local Redis configuration.

set -euo pipefail

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

printf "\n➡️  Detected OS: %s | Arch: %s\n" "$OS" "$ARCH"

# Ensure uv is installed globally (fallback to pipx if available)
if ! command -v uv >/dev/null 2>&1; then
  printf "\n🔧 Installing uv...\n"
  python3 -m pip install --user --upgrade uv || {
    echo "Failed to install uv – please make sure Python 3 & pip are available."; exit 1; }
fi

# Create virtual-env and sync project deps
printf "\n📦 Installing project dependencies with uv...\n"
uv sync

# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure .env exists for python-dotenv (copy from example if missing)
if [[ ! -f .env && -f .env.example ]]; then
  printf "\n📄 Creating default .env from .env.example...\n"
  cp .env.example .env
fi

# Install ColPali engine (multimodal retrieval)
printf "\n📦 Installing ColPali engine...\n"
uv pip install \
  colpali-engine@git+https://github.com/illuin-tech/colpali@80fb72c9b827ecdb5687a3a8197077d0d01791b3

printf "\n🚀 Starting Morphik server...\n\n"
uv run start_server.py
