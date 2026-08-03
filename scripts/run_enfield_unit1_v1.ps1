param(
    [switch]$DryRun,
    [switch]$Execute,
    [switch]$ContractOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$RepositoryRoot = "D:\Segro_PoC_codex_v2"
$ExpectedBranch = "feature/evidence-first-foundation"
$ConfigPath = "config\enfield_unit1_runner_v1.json"

Set-Location $RepositoryRoot

$Branch = git branch --show-current
if ($Branch -ne $ExpectedBranch) {
    Write-Error "Expected branch $ExpectedBranch but found $Branch"
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Error "Runner configuration not found: $ConfigPath"
}

$SelectedModes = @($DryRun, $Execute, $ContractOnly) | Where-Object { $_ }
if ($SelectedModes.Count -ne 1) {
    Write-Error "Specify exactly one of -DryRun, -Execute or -ContractOnly."
}

$VenvActivate = Join-Path $RepositoryRoot ".venv\Scripts\Activate.ps1"
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if ((Test-Path -LiteralPath $VenvActivate) -and (Test-Path -LiteralPath $VenvPython)) {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $VenvPython -c "import typer, pydantic" *> $null
    $VenvProbeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    $VenvReady = ($VenvProbeExitCode -eq 0)
}
else {
    $VenvReady = $false
}

if ($VenvReady) {
    . $VenvActivate
}

$env:PYTHONPATH = "src"

$Arguments = @("-m", "segro_evidence_extraction", "run-unit", "--config", $ConfigPath)
if ($DryRun) {
    $Arguments += "--dry-run"
}
if ($Execute) {
    $Arguments += "--execute"
}
if ($ContractOnly) {
    $Arguments += "--contract-only"
}
if ($Resume) {
    $Arguments += "--resume"
}

python @Arguments
$ExitCode = $LASTEXITCODE
Write-Host "Unit Runner output root: output\unit_runs\enfield_unit1"
Write-Host "Exit code: $ExitCode"
exit $ExitCode
