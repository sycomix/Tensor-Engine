import tensor_engine as te
b=te.TransformerBlock(d_model=2048,d_ff=8192,num_heads=32)
try:
    named=list(b.named_parameters(''))
except Exception as e:
    named=[]
print('Named params count:', len(named))
for name,_ in named[:80]:
    print(name)
