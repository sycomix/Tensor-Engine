#!/usr/bin/env python3
"""Analyze detailed sampling results for non-ASCII rates and per-step rank/prob stats."""
import json
from pathlib import Path
from statistics import median

DETAILED = Path('scripts/sampling_results_detailed.json')
FALLBACK = Path('scripts/sampling_results.json')
if DETAILED.exists():
    IN_PATH = DETAILED
elif FALLBACK.exists():
    IN_PATH = FALLBACK
else:
    raise SystemExit("No sampling results found; run collect_sampling_runs.py first")

def is_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)

with open(IN_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

profile_rows = []
OUT = Path('scripts/sampling_analysis_detailed.txt')
with open(OUT, 'w', encoding='utf-8') as out:
    out.write('Profile\tSeeds\tAvgNonAsciiTokenFrac\tNonAsciiOutputs\tFracChosenOutsideTopK\tMedianRank\tAvgEntropy\tFracNonAsciiChosenTokens\n')
    for profile, seeds in data.items():
        seed_count = len(seeds)
        # per-seed non-ascii summary (based on final token_strs/decoded)
        non_ascii_fracs = []
        non_ascii_output_count = 0
        # per-step aggregates
        ranks = []
        entropies = []
        chosen_outside = 0
        chosen_non_ascii = 0
        total_steps = 0

        for seed_str, rec in seeds.items():
            token_strs = rec.get('token_strs', [])
            decoded = rec.get('decoded', '')
            non_ascii_token_count = sum(1 for t in token_strs if is_non_ascii(t))
            non_ascii_frac = non_ascii_token_count / max(1, len(token_strs))
            non_ascii_fracs.append(non_ascii_frac)
            if is_non_ascii(decoded):
                non_ascii_output_count += 1

            per_step = rec.get('per_step', [])
            for s in per_step:
                total_steps += 1
                ranks.append(s.get('rank', -1))
                entropies.append(s.get('entropy', 0.0))
                if not s.get('chosen_in_topk', True):
                    chosen_outside += 1
                if is_non_ascii(s.get('token_str', '')):
                    chosen_non_ascii += 1

        avg_non_ascii_frac = sum(non_ascii_fracs) / max(1, seed_count)
        frac_chosen_outside = chosen_outside / max(1, total_steps)
        med_rank = median([r for r in ranks if r > 0]) if any(r > 0 for r in ranks) else -1
        avg_entropy = sum(entropies) / max(1, len(entropies))
        frac_nonascii_chosen = chosen_non_ascii / max(1, total_steps)

        profile_rows.append((profile, seed_count, avg_non_ascii_frac, non_ascii_output_count, frac_chosen_outside, med_rank, avg_entropy, frac_nonascii_chosen))

        out.write(f"{profile}\t{seed_count}\t{avg_non_ascii_frac:.3f}\t{non_ascii_output_count}\t{frac_chosen_outside:.3f}\t{med_rank}\t{avg_entropy:.3f}\t{frac_nonascii_chosen:.3f}\n")

# Print a short summary
print("Profile summary:")
print("profile\tseeds\tavg_non_ascii_token_frac\tnon_ascii_outputs\tfrac_chosen_outside_topk\tmedian_rank\tavg_entropy\tfrac_nonascii_chosen")
for r in profile_rows:
    print(f"{r[0]}\t{r[1]}\t{r[2]:.3f}\t{r[3]}\t{r[4]:.3f}\t{r[5]}\t{r[6]:.3f}\t{r[7]:.3f}")

print("Wrote detailed analysis to:", OUT)
