# StrataRPCD worker (PC C): runs DeepSeek-V4.1-Flash's layers 20-39 and the head for the coordinator (PC A).
# Run it from the StrataRPCD folder once setup has made the model's pack on this PC (README-STRATARPCD.md):
#   $env:STRATA_RPC_TOKEN = "the same long secret as on PC A"
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1
# Ctrl+C stops it. Every switch is optional; the defaults are split 20 and BF16 rows (-Wire fp32: exact rows).
param(
    [string]$Root = "",                  # the StrataRPCD folder (default: the folder above this script's)
    [string]$Config = "",                # setup's run config (default: the strata-deepseek-v4.1-flash-*.json in $Root)
    [string]$Listen = "0.0.0.0:39501",   # where the coordinator connects (this PC's port)
    [int]$Split = 20,                    # the worker's first layer: the same on both PCs
    [ValidateSet("fp32", "bf16", "fp16")][string]$Wire = "bf16",   # the rows on the link, the same on both PCs: bf16 (half the bytes), fp32 exact
    [int]$Context = 0,                   # 0: the config's --max-context (it must match the coordinator's)
    [int]$PrefillChunk = 0,              # the prompt chunk on this PC (0: the engine's own, 4096 on a 16 GB card): smaller leaves more VRAM for experts
    [switch]$Vision,                     # encode images on A's GPU; image routing on both PCs
    [switch]$Sessions,                   # opt-in manual paired disk sessions, on both PCs
    [string]$SessionDir = "",            # this PC's own session folder
    [switch]$Stats,                      # STRATA_LAYER_STAGE_STATS=1: each prompt's time in the log
    [switch]$DryRun                      # print what it would run and stop (nothing starts)
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = Split-Path -Parent $here }
$exe = Join-Path $here "strata-dsv41.exe"
if (-not (Test-Path $exe)) { throw "strata-dsv41.exe is not beside this script ($here)" }
if (-not $env:STRATA_RPC_TOKEN) { throw 'set $env:STRATA_RPC_TOKEN = "<the same long secret as on PC A>" in this window first' }
if (-not $Config) {
    $found = @(Get-ChildItem -Path $Root -Filter "strata-deepseek-v4.1-flash-*.json" | Where-Object { $_.Name -notlike "*stratarpcd*" })
    if ($found.Count -ne 1) { throw "found $($found.Count) setup configs (strata-deepseek-v4.1-flash-*.json) in ${Root}: pass -Config" }
    $Config = $found[0].FullName
}
$cfg = Get-Content $Config -Raw | ConvertFrom-Json

# setup's engine arguments (the pack, the context, ...), then this PC's role
$a = New-Object System.Collections.Generic.List[string]
$given = @($cfg.args)
$useVision = $Vision -or ($given -contains "--layer-stage-vision")
$useSessions = $Sessions -or ($given -contains "--layer-stage-sessions")
if (-not $SessionDir) {
    $si = [array]::IndexOf($given, "--layer-stage-session-dir")
    if ($si -ge 0 -and $si + 1 -lt $given.Count) { $SessionDir = [string]$given[$si + 1] }
}
for ($i = 0; $i -lt $given.Count; $i++) {
    $k = [string]$given[$i]
    if ($k -in @("--layer-stage-vision", "--layer-stage-sessions")) { continue }
    if ($useVision -and $k -eq "--vision") { $i++; continue }
    if ($k -like "--layer-stage-*" -or ($Context -gt 0 -and $k -eq "--max-context") -or ($PrefillChunk -gt 0 -and $k -eq "--prefill-chunk")) { $i++; continue }
    $a.Add($k)
}
if ($Context -gt 0) { $a.Add("--max-context"); $a.Add([string]$Context) }
if ($PrefillChunk -gt 0) { $a.Add("--prefill-chunk"); $a.Add([string]$PrefillChunk) }
if ($useVision) { $a.Add("--vision"); $a.Add("on"); $a.Add("--layer-stage-vision") }
if ($useSessions) { $a.Add("--layer-stage-sessions") }
if ($useSessions) {
    if (-not $SessionDir) { $SessionDir = Join-Path $Root "sessions-worker" }
    $a.Add("--layer-stage-session-dir"); $a.Add($SessionDir)
}
foreach ($x in @("--layer-stage-server", $Listen, "--layer-stage-split", [string]$Split, "--layer-stage-wire", $Wire)) { $a.Add($x) }

if ($cfg.env) { foreach ($p in $cfg.env.PSObject.Properties) { Set-Item -Path "env:$($p.Name)" -Value ([string]$p.Value) } }
if ($Stats) { $env:STRATA_LAYER_STAGE_STATS = "1" }
$log = Join-Path $Root ("stratarpcd-worker-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
Write-Host "StrataRPCD worker"
Write-Host "  engine: $exe"
Write-Host "  config: $Config"
Write-Host ("  args:   " + ($a -join " "))
Write-Host "  log:    $log"
Write-Host "  Loading takes several minutes; then it prints 'READY layer-stage worker' and waits for PC A."
if ($DryRun) { Write-Host "DRYRUN worker: $exe $($a -join ' ')"; exit 0 }
if ($cfg.cwd) { Set-Location $cfg.cwd }
$ErrorActionPreference = "Continue"
& $exe @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
exit $LASTEXITCODE
