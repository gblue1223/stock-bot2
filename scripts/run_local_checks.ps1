[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$ExtractedDir,
    [string]$OutputDir,
    [string]$Bundle,
    [int]$ValidationEpisodes = 8,
    [int]$Workers = 2,
    [switch]$PlanOnly,
    [switch]$SkipRegression
)

$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython = Join-Path $taskRoot '.venv64\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw "64-bit Python environment not found: $taskPython"
}
if (-not $ExtractedDir) { $ExtractedDir = Join-Path $taskRoot 'data\extracted_episodes_v2' }
$taskArguments = @('-X', 'utf8', '-u', '-m', 'ai_trader.grpo.local_checks',
    '--checkpoint', $Checkpoint, '--extracted-dir', $ExtractedDir,
    '--validation-episodes', [string]$ValidationEpisodes, '--num-workers', [string]$Workers)
if ($OutputDir) { $taskArguments += @('--output-dir', $OutputDir) }
if ($Bundle) { $taskArguments += @('--bundle', $Bundle) }
if ($PlanOnly) { $taskArguments += '--plan-only' }
if ($SkipRegression) { $taskArguments += '--skip-regression' }
Push-Location -LiteralPath $taskRoot
try {
    & $taskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
