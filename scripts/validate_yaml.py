#!/usr/bin/env python3
import sys
import logging

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

files = ['.github/workflows/update-bench-baseline.yml', '.github/workflows/bench-schedule.yml']
errors = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            yaml.safe_load(fh)
        logger.info("%s parsed OK", f)
    except Exception as e:
        logger.error("%s parse error: %s", f, e)
        errors += 1
sys.exit(errors)
