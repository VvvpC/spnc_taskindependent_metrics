param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$pathEntries = @(
    (Join-Path $repoRoot "src"),
    (Join-Path $repoRoot "src\\Project"),
    (Join-Path $repoRoot "src\\Morphology_Research"),
    (Join-Path $repoRoot "src\\Optuna_TaskIndependent_Metrics"),
    (Join-Path $repoRoot "src\\ParetoFront_CQandMC"),
    (Join-Path $repoRoot "src\\Plot_Functions"),
    (Join-Path $repoRoot "src\\Test_Temporary")
)

$existingPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = ($pathEntries -join ';')
if ($existingPythonPath) {
    $env:PYTHONPATH = "$env:PYTHONPATH;$existingPythonPath"
}

& python @PythonArgs
