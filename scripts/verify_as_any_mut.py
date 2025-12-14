#!/usr/bin/env python3
"""
Verify that:
 - All `impl Module for` blocks contain `fn as_any_mut(`
 - No `impl Operation for` block contains `as_any_mut`

Exits with non-zero code if any check fails.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

rs_files = list(ROOT.rglob('src/**/*.rs'))

impl_module_pattern = re.compile(r'impl\s+Module\s+for\s+([A-Za-z0-9_:\<\>]+)')
impl_trait_pattern = re.compile(r'impl\s+([A-Za-z0-9_:\<\>]+)\s+for\s+([A-Za-z0-9_:\<\>]+)')

errors = []

for path in rs_files:
    s = path.read_text(encoding='utf-8')
    # Find all impl occurrences (start positions)
    for m in re.finditer(r'impl\s+(?:[A-Za-z0-9_:<>]+)\s+for\s+[A-Za-z0-9_:<>]+', s):
        start = m.start()
        # find '{' after start
        brace_pos = s.find('{', start)
        if brace_pos == -1:
            continue
        # find matching closing brace using simple brace matching
        i = brace_pos
        depth = 0
        end = -1
        L = len(s)
        while i < L:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end == -1:
            errors.append(f"Unmatched braces in {path} at position {brace_pos}")
            continue
        impl_block = s[start:end + 1]
        # check trait and type
        trait_match = impl_trait_pattern.match(s[start: start + 256])
        if not trait_match:
            continue
        trait_name = trait_match.group(1)
        type_name = trait_match.group(2)
        if trait_name == 'Module':
            # must contain as_any_mut
            if 'fn as_any_mut' not in impl_block and 'as_any_mut(&mut self)' not in impl_block and 'as_any_mut<' not in impl_block:
                errors.append(f"Missing as_any_mut for Module impl on {type_name} in {path}")
        if trait_name == 'Operation':
            if 'as_any_mut' in impl_block:
                errors.append(f"Operation impl for {type_name} in {path} contains as_any_mut (should not)")

if errors:
    print("ERROR: as_any_mut verification failed:\n")
    for e in errors:
        print(" - ", e)
    sys.exit(2)

print("OK: as_any_mut verification passed. All Module impls contain as_any_mut and Operation impls do not.")
sys.exit(0)
