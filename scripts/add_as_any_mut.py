import re
from pathlib import Path

root = Path(r"d:\Tensor-Engine\src")

# Insert as_any_mut only into impl Module for ... { ... } blocks missing as_any_mut
print('Scanning for impl Module blocks...')
count = 0
for p in root.rglob('*.rs'):
    s = p.read_text(encoding='utf-8')

    # iterate through the file and find 'impl Module for' occurrences
    out = s
    idx = 0
    changed = False
    while True:
        match = re.search(r"impl\s+Module\s+for\s+[^\{]+\{", out[idx:])
        if not match:
            break
        start = idx + match.start()
        brace_idx = start + match.end() - match.start()
        # find matching closing brace by tracking nested braces
        depth = 1
        i = brace_idx
        while i < len(out) and depth > 0:
            if out[i] == '{':
                depth += 1
            elif out[i] == '}':
                depth -= 1
            i += 1
        end = i
        block = out[start:end]

        # If the block doesn't contain 'as_any_mut', insert before the final '}'
        if 'as_any_mut(' not in block:
            # Insert the as_any_mut snippet just before the closing '}' of the impl block
            insert_point = end - 1  # position of '}'
            snippet = " fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }"
            out = out[:insert_point] + snippet + out[insert_point:]
            changed = True
            count += 1
            # advance idx past the modified block
            idx = insert_point + len(snippet) + 1
        else:
            # move idx beyond this block to find other impls
            idx = end

    if changed:
        p.write_text(out, encoding='utf-8')
        print('Updated:', p)

print('Updated blocks:', count)

# Run verification to ensure we didn't accidentally add as_any_mut to the wrong trait impls
import subprocess
try:
    subprocess.check_call(["python", "scripts/verify_as_any_mut.py"])
    print("Verification passed after updates")
except subprocess.CalledProcessError as e:
    print("Verification failed after updates. Please inspect the changes.")
    raise

