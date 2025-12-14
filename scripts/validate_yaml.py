#!/usr/bin/env python3
import sys

import yaml

files = ['.github/workflows/update-bench-baseline.yml', '.github/workflows/bench-schedule.yml']
errors = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            yaml.safe_load(fh)
        print(f + ' parsed OK')
    except Exception as e:
        print(f + ' parse error: ' + str(e))
        errors += 1
sys.exit(errors)
