param(
    [string]$Name = 'bert-base-uncased',
    [string]$Out = 'examples/tokenizer'
)

python ./scripts/download_tokenizer.py --name $Name --out $Out
