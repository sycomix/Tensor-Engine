#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# A small script to compare the mean time of a single Criterion bench (estimates.json) between two runs
# Usage: compare_criterion_reports.py baseline_estimates.json new_estimates.json bench_label threshold_percent

def load_mean_ns(estimates_file: Path):
    with open(estimates_file) as f:
        data = json.load(f)
    # If this file is an estimates.json produced by Criterion, it has a top-level 'mean' with a point_estimate in nanoseconds
    if 'mean' in data and isinstance(data['mean'], dict) and 'point_estimate' in data['mean']:
        return float(data['mean']['point_estimate'])
    # Otherwise, if it's a report summary with 'benchmarks' array, try to find the right named benchmark
    if 'benchmarks' in data and isinstance(data['benchmarks'], list):
        # Choose the first benchmark's mean if we can't match - not ideal but pragmatic for now
        for entry in data['benchmarks']:
            if 'mean' in entry and 'point_estimate' in entry['mean']:
                return float(entry['mean']['point_estimate'])
    raise ValueError(f"Could not parse mean from {estimates_file}")

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: compare_criterion_reports.py baseline.json new.json bench_label threshold_percent mode')
        print('mode: regress (fail if new slower) | improve (fail if new faster than threshold)')
        sys.exit(2)
    baseline = Path(sys.argv[1])
    new = Path(sys.argv[2])
    bench_label = sys.argv[3]
    threshold_percent = float(sys.argv[4])
    mode = sys.argv[5]

    baseline_ns = load_mean_ns(baseline)
    new_ns = load_mean_ns(new)
    baseline_ms = baseline_ns / 1e6
    new_ms = new_ns / 1e6
    print(f"Baseline {bench_label}: {baseline_ms:.6f} ms; New: {new_ms:.6f} ms; Threshold: {threshold_percent}%")
    if baseline_ns <= 0:
        print('Baseline has non-positive value, cannot compare')
        sys.exit(2)
    diff_percent = (new_ns - baseline_ns) / baseline_ns * 100.0
    if mode == 'regress':
        if diff_percent > threshold_percent:
            print(f"Regression detected: {diff_percent:.2f}%")
            sys.exit(1)
        else:
            print('OK')
            sys.exit(0)
    elif mode == 'improve':
        # Improvement means new is less than baseline by threshold percent
        if diff_percent < -threshold_percent:
            print(f"Improvement detected: {diff_percent:.2f}%")
            sys.exit(0)
        else:
            print('No improvement')
            sys.exit(1)
    else:
        print('Unknown mode, expected regress or improve')
        sys.exit(2)
