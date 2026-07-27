#!/usr/bin/env python3
"""Reject executable Torch dependencies while preserving pure-Rust format parsing."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORED_DIRS = {
    ".git", ".venv", "build", "dist", "node_modules", "site", "target", "vendor",
}
SCANNED_SUFFIXES = {
    ".c", ".cfg", ".cpp", ".cs", ".h", ".hpp", ".ini", ".json", ".ps1",
    ".py", ".rs", ".sh", ".toml", ".yaml", ".yml",
}

# These modules only decode external checkpoint tags using Tensor Engine code.
FORMAT_PARSER_ALLOWLIST = {
    "src/hf_compat/unpickler.rs",
    "src/io/pytorch_loader.rs",
}

RULES = {
    "python import": re.compile(r"^\s*(?:from|import)\s+torch\b", re.MULTILINE),
    "Torch-style alias": re.compile(r"\btensor_engine\s+as\s+torch\b"),
    "Torch API call": re.compile(r"\btorch\."),
    "Rust tch integration": re.compile(r"\btch(?:-backend|::|\s*=)"),
    "Torch installation": re.compile(
        r"(?:pip|uv|conda|poetry|requirements?).{0,120}\b(?:torch|libtorch|tch)\b",
        re.IGNORECASE,
    ),
}


def iter_source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.resolve() != Path(__file__).resolve():
            yield path


def main() -> int:
    violations: list[tuple[str, int, str, str]] = []
    for path in iter_source_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for rule_name, pattern in RULES.items():
            if relative in FORMAT_PARSER_ALLOWLIST and rule_name == "Torch API call":
                continue
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                snippet = lines[line_number - 1].strip()
                violations.append((relative, line_number, rule_name, snippet))

    if violations:
        print("Zero-Torch policy violations:")
        for relative, line, rule, snippet in violations:
            print(f"  {relative}:{line}: {rule}: {snippet}")
        return 1

    print("Zero-Torch policy passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
