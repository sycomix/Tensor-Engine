import os  
import sys  
src = 'src'  
results = {}  
for root, dirs, files in os.walk(src):  
    for fname in files:  
        if not fname.endswith('.rs'): continue  
        fpath = os.path.join(root, fname)  
        try:  
            with open(fpath, encoding='utf-8', errors='replace') as f:  
                lines = f.readlines()  
        except:  
            continue  
        test_starts = {i for i, l in enumerate(lines) if '#[cfg(test)]' in l}  
        prod = []  
        for i, l in enumerate(lines):  
            if '.unwrap()' in l and not any(i  for ts in test_starts):  
                prod.append((i+1, l.rstrip()))  
        if prod:  
            results[fpath] = prod  
for fpath, uw in sorted(results.items()):  
    print(f'=== {fpath} ({len(uw)}) ===')  
    for n, line in uw[:10]:  
        print(f'  L{n}: {line.strip()[:120]}')  
    if len(uw)  
        print(f'  ... and {len(uw)-10} more')  
print(f'TOTAL: {sum(len(v) for v in results.values())} production unwraps') 
