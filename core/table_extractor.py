"""Basic table extraction for the MVP."""

from __future__ import annotations

import csv
from pathlib import Path


def _save_table_csv(rows: list[list[str]], output_dir: Path, document_id: str, table_index: int) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document_id}_table_{table_index}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return str(path)


def extract_tables_from_text(text: str) -> list[dict]:
    """Find simple tabular fragments in extracted text."""

    tables: list[dict] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if "\t" in line or "|" in line:
            delimiter = "\t" if "\t" in line else "|"
            cells = [cell.strip() for cell in line.split(delimiter) if cell.strip()]
            if len(cells) >= 2:
                current.append(cells)
                continue
        if len(current) >= 2:
            tables.append({"page": None, "rows": current})
        current = []
    if len(current) >= 2:
        tables.append({"page": None, "rows": current})
    return tables


def extract_tables(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    document_id: str = "document",
    text: str = "",
) -> tuple[list[dict], list[str]]:
    """Extract tables with pdfplumber when available, otherwise use a text fallback."""

    warnings: list[str] = []
    tables: list[dict] = []
    path = Path(pdf_path)

    if path.exists():
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    for table in page.extract_tables() or []:
                        rows = [[str(cell or "").strip() for cell in row] for row in table if row]
                        if rows:
                            tables.append({"page": page_number, "rows": rows})
        except Exception as exc:
            warnings.append(f"Table extraction via pdfplumber skipped: {exc}")

    if not tables and text:
        tables = extract_tables_from_text(text)

    if output_dir and tables:
        out_dir = Path(output_dir)
        for index, table in enumerate(tables, start=1):
            table["csv_path"] = _save_table_csv(table["rows"], out_dir, document_id, index)

    return tables, warnings
