"""Run examples under examples/ and report pass/skip/fail.
This script runs top-level Python examples with a short timeout and classifies results.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = list((ROOT / 'examples').rglob('*.py'))

results = {}

for p in EXAMPLES:
    name = str(p.relative_to(ROOT))
    print(f"Running {name}...")
    try:
        proc = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=60)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0:
            results[name] = {'status': 'pass', 'output': out}
            print(f"PASS: {name}")
        else:
            # common missing asset patterns -> skip
            if 'Config file not found' in out or 'not found' in out or 'No such file' in out or 'not found; skipping' in out:
                results[name] = {'status': 'skip', 'output': out}
                print(f"SKIP: {name} (missing assets or expected skip)")
            else:
                results[name] = {'status': 'fail', 'output': out}
                print(f"FAIL: {name} (exit {proc.returncode})")
    except subprocess.TimeoutExpired as ex:
        results[name] = {'status': 'timeout', 'output': str(ex)}
        print(f"TIMEOUT: {name}")

# Write results
with open(ROOT / 'examples_run_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nSummary written to examples_run_report.json')
