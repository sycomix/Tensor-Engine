#!/usr/bin/env python3
import re
from pathlib import Path
p = Path('e:/Tensor-Engine/src/nn/transformer.rs')
text = p.read_text(encoding='utf-8')
backup = p.with_suffix('.rs.bak')
backup.write_text(text, encoding='utf-8')

# Replace q.reshape(...).unwrap().permute(...).reshape(...).unwrap() patterns for q,k,v
pattern = re.compile(r"(?P<var>\b[qkv]\w*)\.reshape\(vec!\[b, seq, self\.num_heads, head_dim\]\)\.unwrap\(\)\.permute\(vec!\[0, 2, 1, 3\]\)\.reshape\(vec!\[b \* self\.num_heads, seq, head_dim\]\)\.unwrap\(\)")

def repl_multi(m):
    var = m.group('var')
    return f"match reshape_for_multihead(&{var}, b, seq, self.num_heads, head_dim) {{ Ok(t) => t, Err(e) => {{ log::error!(\"MultiHeadAttention.forward_impl: failed to reshape {var}: {{}}\", e); return x.clone(); }} }}"

new_text = pattern.sub(repl_multi, text)

# Replace out.reshape([...]).unwrap() -> checked
pattern_out = re.compile(r"out\.reshape\(vec!\[b, self\.num_heads, seq, head_dim\]\)\.unwrap\(\)")
new_text = pattern_out.sub("match out.reshape(vec![b, self.num_heads, seq, head_dim]) { Ok(t) => t, Err(e) => { log::error!(\"MultiHeadAttention.forward_impl: failed to reshape attention output to heads: {}\", e); return x.clone(); } }", new_text)

# Replace out3.reshape([...single...])
pattern_out4 = re.compile(r"out3\.reshape\(vec!\[b, seq, self\.d_model\]\)\.unwrap\(\)")
new_text = pattern_out4.sub("match out3.reshape(vec![b, seq, self.d_model]) { Ok(t) => t, Err(e) => { log::error!(\"MultiHeadAttention.forward_impl: failed to reshape attention output final: {}\", e); return x.clone(); } }", new_text)

# Replace rb_arr.view().into_dimensionality::<ndarray::Ix2>().unwrap() occurrences
pattern_rb = re.compile(r"rb_arr\.view\(\)\.into_dimensionality::<ndarray::Ix2>\(\)\.unwrap\(\)")
new_text = pattern_rb.sub("match rb_arr.view().into_dimensionality::<ndarray::Ix2>() { Ok(v) => v, Err(e) => { log::error!(\"MultiHeadAttention.forward_impl: relative bias array dimension error: {}\", e); return x.clone(); } }", new_text)

# Write back only if changes made
if new_text != text:
    p.write_text(new_text, encoding='utf-8')
    print('transformer.rs updated; backup at', backup)
else:
    print('No changes made')
