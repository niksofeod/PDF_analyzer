"""Command-line interface for processing PDF documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from core.document_classifier import train_baseline_model
from core.pipeline import PROJECT_ROOT, process_document


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Process PDF documents with the Document AI MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process one PDF file")
    process_parser.add_argument("pdf_path", type=Path, help="Path to a PDF document")
    process_parser.add_argument("--json", action="store_true", help="Print the full JSON result")

    train_parser = subparsers.add_parser("train-classifier", help="Train the document type classifier")
    train_parser.add_argument(
        "--sample-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_texts",
        help="Directory with class subfolders and .txt samples",
    )
    train_parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "document_classifier.pkl",
        help="Where to save the trained .pkl model",
    )
    return parser


def _print_summary(result: dict) -> None:
    fields = {name: data for name, data in result.get("fields", {}).items() if data.get("value")}
    print(f"document_id: {result['document_id']}")
    print(f"pdf_type: {result['pdf_type']}")
    print(f"document_type: {result['document_type']['label']} ({result['document_type']['confidence']:.2f})")
    print(f"fields_found: {', '.join(fields.keys()) if fields else 'none'}")
    print(f"signatures_found: {len(result.get('visual_marks', {}).get('signatures', []))}")
    print(f"stamps_found: {len(result.get('visual_marks', {}).get('stamps', []))}")
    print(f"json_path: {result['json_path']}")
    if result.get("warnings"):
        print("warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    args = build_parser().parse_args(argv)
    if args.command == "process":
        result = process_document(args.pdf_path)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_summary(result)
        return 0
    if args.command == "train-classifier":
        result = train_baseline_model(args.sample_dir, args.model_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("saved") else 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
