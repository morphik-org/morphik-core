#!/bin/bash

# Script to format Python code in the specified order
# 1. isort
# 2. black
# 3. ruff

echo "Running isort..."
isort .

echo "Running black..."
black . --line-length=120

echo "Running ruff..."
ruff check --fix .

echo "Formatting complete!"