param(
    [int]$Epochs = 3,
    [int]$Batch = 4,
    [int]$CheckpointInterval = 1,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $ExtraArgs
)

. .\..\venv\Scripts\Activate.ps1
# Build base command
$cmd = @('python', '-m', 'examples.train_llava', '--epochs', $Epochs, '--batch', $Batch, '--checkpoint-interval', $CheckpointInterval)
if ($ExtraArgs) {
    $cmd += $ExtraArgs
}
& $cmd
