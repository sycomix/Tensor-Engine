import glob
total=0
for f in sorted(glob.glob("src/**/*.rs",recursive=True)):
 with open(f,errors="replace") as fh:
  lines=fh.read().split("\n")
 test_starts={i for i,l in enumerate(lines) if "#[cfg(test)]" in l}
 prod=[(i+1,l.rstrip()) for i,l in enumerate(lines) if ".unwrap()" in l and not any(i>=ts for ts in test_starts)]
 if prod:
  print(f"{f}: {len(prod)}")
  for n,l in prod[:5]:
   print(f"  L{n}: {l.strip()[:100]}")
  if len(prod)>5:
   print(f"  ... +{len(prod)-5} more")
  total+=len(prod)
print(f"Total: {total}")
