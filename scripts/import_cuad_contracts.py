"""Import real contract texts from the downloaded CUAD dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "external" / "cuad" / "extracted" / "CUADv1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample_texts" / "contract"


def _clean_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()[:max_chars].strip()


def import_cuad_contracts(source: Path, output_dir: Path, limit: int, max_chars: int) -> dict:
    """Extract contract contexts from CUAD JSON into classifier training text files."""

    if not source.exists():
        return {"imported": 0, "warning": f"CUAD source file was not found: {source}"}

    data = json.loads(source.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    imported = 0

    for item in data.get("data", []):
        if imported >= limit:
            break
        title = re.sub(r"[^A-Za-z0-9_-]+", "_", str(item.get("title") or f"contract_{imported + 1}")).strip("_")
        paragraphs = item.get("paragraphs") or []
        if not paragraphs:
            continue
        context = _clean_text(str(paragraphs[0].get("context") or ""), max_chars=max_chars)
        if len(context) < 500:
            continue
        imported += 1
        output_path = output_dir / f"cuad_contract_{imported:03d}_{title[:60]}.txt"
        output_path.write_text(context, encoding="utf-8")

    return {
        "imported": imported,
        "source": str(source),
        "output_dir": str(output_dir),
        "label": "contract",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Import CUAD contract texts into data/sample_texts/contract")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=12000)
    return parser


def main() -> int:
    """Run the importer."""

    args = build_parser().parse_args()
    result = import_cuad_contracts(args.source, args.output_dir, args.limit, args.max_chars)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("imported", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

