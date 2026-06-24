import glob

total = 0
for f in sorted(glob.glob("src/**/*.rs", recursive=True)):
    content = open(f, encoding="utf-8", errors="replace").read()
    lines = content.split("\n")
    test_starts = {i for i,l in enumerate(lines) if "#[cfg(test)]" in l}
    prod = [(i+1,l.rstrip()) for i,l in enumerate(lines) if ".unwrap()" in l and not any(i >= ts for ts in test_starts)]
    if prod:
        print(f"{f}: {len(prod)}")
        for n,l in prod[:5]:
            print(f"  L{n}: {l.strip()[:120]}")
        if len(prod) > 5:
            print(f"  ... and {len(prod)-5} more")
        total += len(prod)
print(f"\nTotal production unwraps: {total}")
