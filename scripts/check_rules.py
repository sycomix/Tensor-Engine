#!/usr/bin/env python3
"""Enforce repository code-quality rules defined in .mwx/rules.

The scanner uses practical static checks for common violations in the rules file,
then compares against a baseline so CI can block newly introduced violations
without requiring a one-shot cleanup of historical debt.

Usage:
  python scripts/check_rules.py
  python scripts/check_rules.py --update-baseline
  python scripts/check_rules.py --no-baseline
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES_FILE = ROOT / ".mwx" / "rules"
BASELINE_FILE = ROOT / ".mwx" / "rules-baseline.txt"

SCANNED_SUFFIXES = {
    ".py",
    ".rs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".ps1",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cs",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".php",
    ".rb",
}

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "vendor",
    "third_party",
    "dist",
    "build",
    "target",
    "OpenBLAS-0.3.30-x64-64",
    "site",
}

IGNORED_FILES = {
    ".mwx/rules",
    ".mwx/rules-baseline.txt",
    "scripts/check_rules.py",
    "examples/llava_project/scripts/check_rules.py",
}

FATAL_PATTERNS = {
    "TODO_MARKERS": re.compile(r"\b(TODO|FIXME|HACK)\b"),
    "ELLIPSIS": re.compile(r"\.\.\."),
    "BARE_PASS": re.compile(r"^\s*pass\s*$"),
    "RAISE_NOT_IMPLEMENTED": re.compile(r"^\s*raise\s+NotImplementedError\s*\("),
    "RETURN_NULL": re.compile(r"\breturn\s+null\b", re.IGNORECASE),
    "DEAD_IF_FALSE": re.compile(r"\bif\s*\(\s*(false|0)\s*\)", re.IGNORECASE),
    "PLACEHOLDER_TEXT": re.compile(r"(?i)\b(lorem ipsum|john doe|foo|bar)\b"),
    "HARD_SECRET": re.compile(
        r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*(=|:|\(|\"|')"
    ),
}

WARN_PATTERNS = {
    "BROAD_EXCEPT": re.compile(r"^\s*except\s*(Exception|:)"),
    "DEBUG_PRINT": re.compile(r"^\s*print\s*\("),
}

WARN_IGNORED_PREFIXES = (
    "examples/",
    "experiments/",
    "tests/",
    "scripts/",
    "docs/",
    "benches/",
)


def should_emit_warning(rel_path: str, warning_name: str, line: str) -> bool:
    """Emit warnings for production-oriented paths while filtering known utility/demo noise."""
    if rel_path.endswith(".md"):
        return False

    if rel_path.startswith(WARN_IGNORED_PREFIXES):
        return False

    if warning_name == "DEBUG_PRINT":
        # Prints in Rust code are often deliberate for benches/examples and are covered above.
        # Keep Python/JS print checks in production paths.
        suffix = Path(rel_path).suffix.lower()
        if suffix in {".rs", ".c", ".cpp", ".h", ".hpp"}:
            return False

    if warning_name == "BROAD_EXCEPT":
        # Keep broad except warnings focused on Python production code.
        if not rel_path.endswith(".py"):
            return False
        if "if __name__ == \"__main__\"" in line:
            return False

    return True


def is_ignored(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in IGNORED_FILES:
        return True
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True
    return False


def _normalize_violation(path: Path, line_no: int, kind: str) -> str:
    return f"{path.as_posix()}:{line_no}:{kind}"


def _load_baseline() -> set[str]:
    if not BASELINE_FILE.exists():
        return set()
    values: set[str] = set()
    for raw in BASELINE_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values.add(line)
    return values


def _write_baseline(fatals: list[tuple[Path, int, str, str]]) -> None:
    entries = sorted(_normalize_violation(p, ln, kind) for p, ln, kind, _ in fatals)
    header = [
        "# Baseline for scripts/check_rules.py",
        "# Format: path:line:kind",
    ]
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text("\n".join(header + entries) + "\n", encoding="utf-8")


def check_file(path: Path) -> tuple[list[tuple[Path, int, str, str]], list[tuple[Path, int, str, str]]]:
    if path.suffix.lower() not in SCANNED_SUFFIXES:
        return ([], [])
    if is_ignored(path):
        return ([], [])

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    fatals: list[tuple[Path, int, str, str]] = []
    warns: list[tuple[Path, int, str, str]] = []

    rel = path.relative_to(ROOT)
    for i, line in enumerate(lines, 1):
        for name, pat in FATAL_PATTERNS.items():
            if name == "ELLIPSIS":
                if path.suffix.lower() == ".md":
                    continue
                if re.search(r"(print\(|logger\.|echo |Write-Host )", line):
                    continue
            if pat.search(line):
                fatals.append((rel, i, f"FATAL:{name}", line.strip()))

        for name, pat in WARN_PATTERNS.items():
            if pat.search(line) and should_emit_warning(rel.as_posix(), name, line):
                warns.append((rel, i, f"WARN:{name}", line.strip()))

    return (fatals, warns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce .mwx/rules with baseline support")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite .mwx/rules-baseline.txt with current fatal findings",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Treat all fatal findings as failures (ignore baseline)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not RULES_FILE.exists():
        logger.error("Rules file missing: %s", RULES_FILE)
        return 2

    all_fatals: list[tuple[Path, int, str, str]] = []
    all_warns: list[tuple[Path, int, str, str]] = []

    for p in ROOT.rglob("*"):
        if p.is_file():
            fatals, warns = check_file(p)
            all_fatals.extend(fatals)
            all_warns.extend(warns)

    if args.update_baseline:
        _write_baseline(all_fatals)
        logger.info("Updated baseline: %s", BASELINE_FILE.relative_to(ROOT))
        logger.info("Baseline entries: %d", len(all_fatals))
        return 0

    baseline = set() if args.no_baseline else _load_baseline()

    new_fatals: list[tuple[Path, int, str, str]] = []
    existing_fatals: list[tuple[Path, int, str, str]] = []
    for f in all_fatals:
        key = _normalize_violation(f[0], f[1], f[2])
        if key in baseline:
            existing_fatals.append(f)
        else:
            new_fatals.append(f)

    if new_fatals:
        logger.error("NEW fatal rule violations found (%d):", len(new_fatals))
        for file, line, kind, snippet in new_fatals:
            logger.error("%s:%d [%s] %s", file, line, kind, snippet)
    else:
        logger.info("No new fatal rule violations.")

    if existing_fatals and not args.no_baseline:
        logger.warning("Existing baseline fatal violations: %d", len(existing_fatals))

    if all_warns:
        logger.warning("Warnings (%d):", len(all_warns))
        for file, line, kind, snippet in all_warns:
            logger.warning("%s:%d [%s] %s", file, line, kind, snippet)

    return 1 if new_fatals else 0


if __name__ == "__main__":
    sys.exit(main())
