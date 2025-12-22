from pathlib import Path
files = [
    r"e:\Tensor-Engine\references\trustformers\trustformers-c\src\codegen\generators\go.rs",
    r"e:\Tensor-Engine\references\trustformers\trustformers-c\src\codegen\generators\php.rs",
    r"e:\Tensor-Engine\references\trustformers\trustformers-c\src\codegen\generators\swift.rs",
    r"e:\Tensor-Engine\references\trustformers\trustformers-serve\src\migration\config_migration.rs",
]
for fp in files:
    p=Path(fp)
    if not p.exists():
        print(f"MISSING: {fp}")
        continue
    s=p.read_text(errors='replace')
    bal=0
    first_neg=None
    pos_first_neg=None
    for i,ch in enumerate(s):
        if ch=='{': bal+=1
        elif ch=='}': bal-=1
        if bal<0 and first_neg is None:
            first_neg=i
            pos_first_neg=i
            break
    print("\nFile:", fp)
    # --- scan string literals for single braces ---
    issues=[]
    idx=0
    while True:
        q=s.find('"', idx)
        if q==-1: break
        # find closing quote naive (skip escaped quotes)
        j=q+1
        lit=''
        escaped=False
        while j<len(s):
            ch=s[j]
            if ch=='"' and s[j-1]!='\\':
                break
            lit+=ch
            j+=1
        if j>=len(s): break
        # check for single { or } not doubled
        for k,c in enumerate(lit):
            if c=='{':
                if not (k+1<len(lit) and lit[k+1]=='{') and not (k>0 and lit[k-1]=='{'):
                    issues.append((q+1+k,'{', lit[max(0,k-30):k+30]))
            if c=='}':
                if not (k+1<len(lit) and lit[k+1]=='}') and not (k>0 and lit[k-1]=='}'):
                    issues.append((q+1+k,'}', lit[max(0,k-30):k+30]))
        idx=j+1

    if issues:
        print('Found', len(issues), 'single brace occurrences in string literals (potential mis-escaped braces):')
        for pos,ch,ctx in issues[:10]:
            print('pos', pos, ch, 'ctx', ctx)

    if first_neg is not None:
        print('FIRST_NEG at char', first_neg)
        start=max(0, first_neg-200)
        end=min(len(s), first_neg+200)
        context=s[start:end]
        print('--- context ---')
        print(context)
        print('--- end ---')
        continue
    if bal!=0:
        print('FINAL_BALANCE', bal)
        # find deepest imbalance position by scanning and tracking max imbalance
        b=0
        max_pos=0
        max_b=0
        for i,ch in enumerate(s):
            if ch=='{': b+=1
            elif ch=='}': b-=1
            if b>max_b:
                max_b=b
                max_pos=i
        start=max(0, max_pos-200)
        end=min(len(s), max_pos+200)
        print('MAX_OPEN at', max_pos, 'max depth', max_b)
        print('--- context around max ---')
        print(s[start:end])
        print('--- end ---')
    else:
        print('No imbalance')
