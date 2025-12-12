#!/usr/bin/env python3
"""
Run a rules compliance scan similar to rules.md.

This is a best-effort static check and reports findings for manual
review. It checks for: TODO_ markers, ellipsis usage, bare 'pass' in
function bodies, bare 'print(' usage outside tests, obvious hardcoded
secret patterns, and broad excepts like 'except:' or 'except Exception:'.

Usage: python scripts/check_rules.py
Exit 0 if no violations, non-zero otherwise.
"""
from __future__ import annotations
import re
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FATAL_PATTERNS = {
    'TODO_MARKERS': re.compile(r"\b(TODO|FIXME|HACK)\b"),
    'ELLIPSIS': re.compile(r"\.\.\."),
    # bare pass inside a block (def/for/while)
    'BARE_PASS': re.compile(r"(^\s*pass\s*$)|(^\s*raise\s+NotImplementedError\(\))"),
    # obvious hardcoded secret patterns like API_KEY = "..." or SECRET: "..."
    'HARD_SECRET': re.compile(r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*(=|:|\(|\"|')"),
}

WARN_PATTERNS = {
    # print usage may be okay in CLI scripts; warn if present
    'PRINT_USAGE': re.compile(r"^\s*print\s*\(") ,
    'BROAD_EXCEPT': re.compile(r"^\s*except\s*(Exception|:)"),
}

IGNORED_DIRS = {'.git', 'venv', '__pycache__', 'third_party'}
IGNORED_FILES = {'rules.md', 'scripts/check_rules.py', 'check_rules.py'}
FATAL_VIOLATIONS: list[tuple] = []
WARN_VIOLATIONS: list[tuple] = []


def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True
    return False


def check_file(p: Path):
    if p.suffix.lower() not in {'.py', '.md', '.sh', '.ps1'}:
        return
    if is_ignored(p):
        return
    # allow matching by name or by path relative to ROOT to keep ignores flexible
    if p.name in IGNORED_FILES or p.relative_to(ROOT).as_posix() in IGNORED_FILES:
        return
    text = p.read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        # skip detecting ellipsis or TODOs in rules.md itself
        if p.name == 'rules.md':
            continue
        for name, pat in FATAL_PATTERNS.items():
            if pat.search(line):
                FATAL_VIOLATIONS.append((p.relative_to(ROOT), i, 'FATAL:' + name, line.strip()))
        for name, pat in WARN_PATTERNS.items():
            if pat.search(line):
                WARN_VIOLATIONS.append((p.relative_to(ROOT), i, 'WARN:' + name, line.strip()))


def main():
    for p in ROOT.rglob('*'):
        if p.is_file():
            check_file(p)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if not (FATAL_VIOLATIONS or WARN_VIOLATIONS):
        logger.info('No rule violations found.')
        return 0

    if FATAL_VIOLATIONS:
        logger.error('FATAL rule violations found:')
        for file, line, kind, snippet in FATAL_VIOLATIONS:
            logger.error('%s:%d [%s] %s', file, line, kind, snippet)
    if WARN_VIOLATIONS:
        logger.warning('WARNINGS:')
        for file, line, kind, snippet in WARN_VIOLATIONS:
            logger.warning('%s:%d [%s] %s', file, line, kind, snippet)

    # Return non-zero only if there are fatal violations; warnings do not cause CI failure
    return 1 if FATAL_VIOLATIONS else 0


if __name__ == '__main__':
    sys.exit(main())
