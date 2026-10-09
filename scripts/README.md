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

## Payroll PDF report

`payroll_report.py` ingests ADP earnings-statement PDFs, extracts pay-period
dates, payroll totals, selected tax fields, regular hours, and safe document
metadata, then writes `documents.csv`, `payroll_summary.csv`, `report.html`,
`yearly_hours_comparison.csv`, `monthly_hours_comparison.csv`, and `styles.css`.

```bash
python scripts/payroll_report.py --input "C:\path\to\payroll-pdfs" --output reports\payroll
```

The report excludes account, advice, and address details from its generated
content. The parser uses the repository's existing PyMuPDF dependency and
does not require pandas or a charting package.

The yearly comparison reports extracted regular hours only; overtime, holiday,
and leave hours are not included. Years with no matching statements are
included with a `no statements` status.

The monthly comparison groups statements by pay-date month and reports
statement counts, extracted regular-hours totals, averages, minimums, and
maximums.
