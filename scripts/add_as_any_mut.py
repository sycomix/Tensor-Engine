"""Add missing `as_any_mut` methods to `impl Module for ... {}` blocks.

This helper is used to keep runtime downcasting consistent across the module system.
It scans Rust source files and inserts an `as_any_mut` method into `impl Module for`
blocks that don't already contain one.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from pathlib import Path


_IMPL_MODULE_RE = re.compile(r"impl\s+Module\s+for\s+[^\{]+\{")


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    """Return the index just past the matching closing brace for `text[open_brace_index]`."""
    depth = 1
    i = open_brace_index + 1
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return i


def _patch_file(path: Path) -> int:
    """Patch a single Rust file. Returns number of inserted blocks."""
    original = path.read_text(encoding="utf-8")
    out = original
    idx = 0
    inserted = 0

    while True:
        match = _IMPL_MODULE_RE.search(out, idx)
        if match is None:
            break
        start = match.start()
        brace_idx = match.end() - 1  # points at '{'
        end = _find_matching_brace(out, brace_idx)
        block = out[start:end]

        if "as_any_mut(" not in block:
            insert_point = end - 1  # position of '}'
            snippet = "\n    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }\n"
            out = out[:insert_point] + snippet + out[insert_point:]
            inserted += 1
            # Advance beyond inserted snippet to avoid re-processing the same block.
            idx = insert_point + len(snippet) + 1
        else:
            idx = end

    if inserted > 0 and out != original:
        path.write_text(out, encoding="utf-8")
    return inserted


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "src",
        help="Root folder to scan for .rs files (default: repo/src)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        default=True,
        help="Disable scripts/verify_as_any_mut.py after modifications",
    )
    args = parser.parse_args()

    root: Path = args.root
    logger.info("Scanning for impl Module blocks under %s", root)

    total_inserted = 0
    try:
        rust_files = list(root.rglob("*.rs"))
    except OSError as exc:
        logger.error("Failed to enumerate Rust files under %s: %s", root, exc)
        return 2

    for p in rust_files:
        try:
            inserted = _patch_file(p)
        except OSError as exc:
            logger.error("Failed to patch %s: %s", p, exc)
            return 2
        if inserted > 0:
            logger.info("Updated %s (%d blocks)", p, inserted)
            total_inserted += inserted

    logger.info("Inserted as_any_mut into %d impl blocks", total_inserted)

    if args.verify and total_inserted > 0:
        try:
            subprocess.check_call(["python", str(repo_root / "scripts" / "verify_as_any_mut.py")])
            logger.info("Verification passed after updates")
        except subprocess.CalledProcessError as exc:
            logger.error("Verification failed after updates: %s", exc)
            return 1
        except OSError as exc:
            logger.error("Failed to run verification script: %s", exc)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

