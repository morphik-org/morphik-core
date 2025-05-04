# Development Scripts

This directory contains various utility scripts for development.

## Code Formatting

The project uses the following tools for code formatting:

1. `isort` - Sorts imports alphabetically and automatically separates them into sections
2. `black` - Code formatter that enforces a consistent style
3. `ruff` - Fast Python linter with auto-fixes

### Automatic Formatting with Git Pre-commit Hook

The project has a Git pre-commit hook that automatically formats staged Python files in the following order:

1. `isort`
2. `black` (with line length set to 120)
3. `ruff check --fix`

The pre-commit hook is already installed and should run automatically when you commit changes.

### Manual Formatting

To manually format all Python files in the project, run:

```bash
./scripts/format.sh
```

This script runs the same tools in the same order as the pre-commit hook, but on all Python files in the project.

### Configuration

The tools are configured in the `pyproject.toml` file at the root of the project. This ensures consistent formatting regardless of how the tools are invoked.
