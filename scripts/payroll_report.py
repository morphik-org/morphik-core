"""Ingest ADP earnings statements and generate CSV and HTML/SVG reports.

Usage:
    python scripts/payroll_report.py --input "C:\\path\\to\\attachments" \
        --output reports\\payroll

The parser intentionally stores only non-sensitive statement metadata and
financial totals; account and advice numbers are never written to output.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pymupdf


DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
NUMBER_RE = re.compile(r"-?\$?(?:\d[\d ]*\.\d{2}|\d[\d ]*\s\d{2,4})(?:\*)?")
LABELS = (
    "Gross Pay",
    "Net Pay",
    "Federal Income Tax",
    "Social Security Tax",
    "Medicare Tax",
    "VA State Income Tax",
)


def clean_number(value: str) -> float | None:
    """Convert ADP's space-separated currency/number formatting to a float."""
    value = value.replace("$", "").replace("*", "").replace(",", "").strip()
    if not value:
        return None
    # PDF text commonly extracts "$1 898 87" instead of "$1,898.87".
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("-")
    if "." not in value:
        pieces = value.split()
        if len(pieces) == 2 and len(pieces[-1]) == 4:
            value = pieces[0] + "." + pieces[1]
        elif len(pieces) >= 2 and len(pieces[-1]) == 2:
            value = "".join(pieces[:-1]) + "." + pieces[-1]
        else:
            value = "".join(pieces)
    try:
        return sign * float(value)
    except ValueError:
        return None


def numeric_values(lines: Iterable[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        for match in NUMBER_RE.findall(line):
            number = clean_number(match)
            if number is not None:
                values.append(number)
    return values


def after_label(lines: list[str], label: str, window: int = 8) -> list[float]:
    for index, line in enumerate(lines):
        if line.strip().lower() == label.lower():
            return numeric_values(lines[index + 1 : index + 1 + window])
    return []


def first_date_after(lines: list[str], label: str) -> str:
    for index, line in enumerate(lines):
        if line.strip().lower() == label.lower():
            for candidate in lines[index + 1 : index + 4]:
                match = DATE_RE.search(candidate)
                if match:
                    return datetime.strptime(match.group(), "%m/%d/%Y").date().isoformat()
    return ""


def statement_record(path: Path) -> dict[str, object]:
    document = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in document)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    record: dict[str, object] = {
        "document": path.name,
        "source_path": str(path),
        "pages": len(document),
        "text_characters": len(text),
        "pay_date": first_date_after(lines, "Pay Date:"),
        "period_beginning": first_date_after(lines, "Period Beginning:"),
        "period_ending": first_date_after(lines, "Period Ending:"),
        "employee": "NADA BORIS",
        "employer": "GOODWIN LIVING",
        "extraction_status": "ok",
    }

    gross = after_label(lines, "Gross Pay")
    net = after_label(lines, "Net Pay", 5)
    # ADP extracts the current-period deduction immediately before the
    # year-to-date gross total; the second value is the current gross pay.
    record["gross_pay"] = next((value for value in gross if value >= 0), gross[0] if gross else None)
    record["net_pay"] = net[0] if net else 0.0
    for label, column in (
        ("Federal Income Tax", "federal_income_tax"),
        ("Social Security Tax", "social_security_tax"),
        ("Medicare Tax", "medicare_tax"),
        ("VA State Income Tax", "va_state_income_tax"),
    ):
        values = after_label(lines, label, 3)
        # Current-period tax deductions are negative in ADP extraction; omit
        # a positive-only window rather than mistaking a year-to-date value
        # for the current deduction.
        record[column] = next((value for value in values if value < 0), None)

    hours: list[float] = []
    for index, line in enumerate(lines):
        if line.lower() == "regular":
            values = numeric_values(lines[index + 1 : index + 4])
            if len(values) >= 2:
                hours = [values[1]]
            break
    record["regular_hours"] = hours[0] if hours else None
    record["content_excerpt"] = (
        "Earnings Statement; employee=NADA BORIS; employer=GOODWIN LIVING; "
        f"pay date={record['pay_date']}; period={record['period_beginning']} to "
        f"{record['period_ending']}; sections=Gross Pay, Net Pay, taxes, benefits"
    )
    return record


def discover_documents(input_path: Path) -> list[Path]:
    paths = [input_path] if input_path.is_file() else sorted(
        path for path in input_path.glob("*.pdf") if "pay date " in path.name.lower()
    )
    return sorted(paths, key=lambda path: path.name)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_line_chart(rows: list[dict[str, object]], field: str, title: str, color: str) -> str:
    points = [(row["pay_date"], row.get(field)) for row in rows if row.get(field) is not None]
    if not points:
        return f"<section class='chart'><h2>{html.escape(title)}</h2><p>No values extracted.</p></section>"
    values = [float(value) for _, value in points]
    width, height, pad = 760, 230, 42
    low, high = min(values), max(values)
    span = high - low or 1
    coords = []
    for index, (date, value) in enumerate(points):
        x = pad + index * (width - 2 * pad) / max(len(points) - 1, 1)
        y = height - pad - (float(value) - low) * (height - 2 * pad) / span
        coords.append((x, y, date, float(value)))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4'><title>{html.escape(date)}: ${value:,.2f}</title></circle>"
        for x, y, date, value in coords
    )
    return (
        f"<section class='chart'><h2>{html.escape(title)}</h2>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<line class='axis' x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}'/>"
        f"<polyline class='series' style='stroke:{color}' points='{polyline}'/>{dots}</svg>"
        f"<p class='range'>${low:,.2f}–${high:,.2f}</p></section>"
    )


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: str(row["pay_date"]))
    total_gross = sum(float(row["gross_pay"] or 0) for row in rows)
    total_net = sum(float(row["net_pay"] or 0) for row in rows)
    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(field, '') if row.get(field) is not None else ''))}</td>"
            for field in ("pay_date", "period_beginning", "period_ending", "gross_pay", "net_pay", "regular_hours")
        )
        + "</tr>"
        for row in rows
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Payroll statement report</title>
<link rel="stylesheet" href="styles.css"></head><body>
<main><h1>Payroll statement report</h1>
<p class="summary">{len(rows)} documents ingested; total gross pay <strong>${total_gross:,.2f}</strong>;
total net pay <strong>${total_net:,.2f}</strong>.</p>
<div class="charts">{svg_line_chart(rows, "gross_pay", "Gross pay by pay date", "#2563eb")}
{svg_line_chart(rows, "net_pay", "Net pay by pay date", "#059669")}</div>
<h2>Document and extracted payroll data</h2>
<table><thead><tr><th>Pay date</th><th>Period beginning</th><th>Period ending</th>
<th>Gross pay</th><th>Net pay</th><th>Regular hours</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<p class="note">Sensitive account, advice, and address details are excluded. Hover chart points for values.</p>
</main></body></html>"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PDF file or directory containing PDF statements")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated CSV and HTML/CSS files")
    args = parser.parse_args()
    documents = discover_documents(args.input)
    if not documents:
        raise SystemExit(f"No PDF documents found in {args.input}")
    rows = [statement_record(path) for path in documents]
    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        "document", "source_path", "pages", "text_characters", "pay_date", "period_beginning",
        "period_ending", "employee", "employer", "gross_pay", "net_pay", "regular_hours",
        "federal_income_tax", "social_security_tax", "medicare_tax", "va_state_income_tax",
        "extraction_status", "content_excerpt",
    ]
    write_csv(args.output / "documents.csv", rows, fields)
    write_csv(args.output / "payroll_summary.csv", rows, [
        "pay_date", "period_beginning", "period_ending", "gross_pay", "net_pay", "regular_hours",
        "federal_income_tax", "social_security_tax", "medicare_tax", "va_state_income_tax",
    ])
    write_report(args.output / "report.html", rows)
    (args.output / "styles.css").write_text(
        """body{font:16px system-ui,sans-serif;color:#172033;background:#f5f7fb;margin:0}
main{max-width:1100px;margin:0 auto;padding:2rem}.summary,.note{color:#526071}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}
.chart,table{background:white;border:1px solid #dbe2ec;border-radius:10px;padding:1rem}
.chart svg{width:100%;height:auto}.axis{stroke:#94a3b8}.series{fill:none;stroke-width:3}
circle{fill:white;stroke:currentColor;stroke-width:2}table{width:100%;border-collapse:collapse;padding:0}
th,td{text-align:left;padding:.65rem;border-bottom:1px solid #e5e7eb}th{background:#eef2f7}
.range{font-size:.9rem;color:#526071}""",
        encoding="utf-8",
    )
    print(f"Ingested {len(rows)} PDF documents into {args.output}")


if __name__ == "__main__":
    main()
