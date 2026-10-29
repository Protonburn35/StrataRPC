# StrataRPCG worker (the 5070 Ti PC): runs GLM-5.3-Flash's layers Split .. 44 (and the output head, unless
# -HeadOnCoordinator) for the coordinator (the 4090 PC).  Run it from the StrataRPCG folder once START-HERE.bat has
# set the model up on this PC (docs\STRATARPCG.md):
#   $env:STRATA_RPC_TOKEN = "the same long secret as on the 4090 PC"
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-worker.ps1
# Ctrl+C stops it.  Every switch is optional, and -Split, -Wire, -HeadOnCoordinator, -Context and -Sessions must be
# the same on both PCs.
param(
    [string]$Root = "",                  # the StrataRPCG folder (default: two folders above this script)
    [string]$Config = "",                # setup's config (default: the one maya-*.json for GLM in $Root)
    [string]$Listen = "0.0.0.0:39501",   # where the coordinator connects (this PC's port: the same as StrataRPCD's)
    [int]$Split = 24,                    # the worker's first layer (1-44): the same on both PCs
    [ValidateSet("fp32", "bf16", "fp16")][string]$Wire = "bf16",   # the rows on the link: bf16 (half the bytes), fp32 exact
    [switch]$HeadOnCoordinator,          # the output head and the sampling on the 4090 instead (both PCs)
    [int]$Context = 0,                   # 0: the config's --max-context (it must match the coordinator's)
    [int]$Gpu = 0,                       # this PC's GPU (nvidia-smi's number)
    [switch]$Sessions,                   # opt-in paired disk sessions (saved chats), on both PCs
    [string]$SessionDir = "",            # this PC's half of each saved chat (default: <Root>\sessions-worker)
    [switch]$Stats,                      # STRATA_LAYER_STAGE_STATS=1: each prompt's and every 256 tokens' times in the log
    [switch]$DryRun                      # print what it would run and stop (nothing starts)
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $here) }
if (-not $env:STRATA_RPC_TOKEN -and -not $DryRun) {
    throw 'set $env:STRATA_RPC_TOKEN = "<the same long secret as on the 4090 PC>" in this window first'
}
if (-not $Config) {
    $found = @(Get-ChildItem -Path $Root -Filter "maya-*.json" | Where-Object {
        $_.Name -notlike "*stratarpcg*" -and $_.Name -notlike "*.shared-settings.json" -and
        (Get-Content $_.FullName -Raw) -match '"--glm-pack"' })
    if ($found.Count -ne 1) { throw "found $($found.Count) GLM configs (maya-*.json with --glm-pack) in ${Root}: pass -Config" }
    $Config = $found[0].FullName
}
$cfg = Get-Content $Config -Raw | ConvertFrom-Json
$exe = [string]$cfg.exe
if (-not (Test-Path $exe)) { throw "the engine $exe is missing: run START-HERE.bat on this PC first" }

# setup's engine arguments (the pack, the context, the expert pool, KV streaming, the expert profile), without what a
# layer stage does not take, then this PC's role
$a = New-Object System.Collections.Generic.List[string]
$given = @($cfg.args)
$drop1 = @("--layer-stage-sessions")                                  # flags without a value
$drop2 = @("--parallel", "--dflash", "--layer-split", "--split-device") # flags with a value
for ($i = 0; $i -lt $given.Count; $i++) {
    $k = [string]$given[$i]
    if ($drop1 -contains $k) { continue }
    if ($drop2 -contains $k -or $k -like "--layer-stage-*" -or ($Context -gt 0 -and $k -eq "--max-context")) { $i++; continue }
    $a.Add($k)
}
if ($Context -gt 0) { $a.Add("--max-context"); $a.Add([string]$Context) }
foreach ($x in @("--layer-stage-server", $Listen, "--layer-stage-split", [string]$Split, "--layer-stage-wire", $Wire,
                 "--layer-stage-gpu", "0")) { $a.Add($x) }
if ($HeadOnCoordinator) { $a.Add("--layer-stage-head"); $a.Add("coordinator") }
if ($Sessions) {
    if (-not $SessionDir) { $SessionDir = Join-Path $Root "sessions-worker" }
    $a.Add("--layer-stage-sessions"); $a.Add("--layer-stage-session-dir"); $a.Add($SessionDir)
}

# the engine's environment as the server gives it (serve/server.py engine_env): this GPU only, the config's settings,
# the CUDA libraries setup found first on PATH
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = [string]$Gpu
if ($cfg.env) { foreach ($p in $cfg.env.PSObject.Properties) { Set-Item -Path "env:$($p.Name)" -Value ([string]$p.Value) } }
if ($cfg.lib_dirs) {
    $dirs = @($cfg.lib_dirs | Where-Object { Test-Path $_ })
    if ($dirs.Count -gt 0) { $env:PATH = (($dirs + @($env:PATH)) -join ";") }
}
if ($Stats) { $env:STRATA_LAYER_STAGE_STATS = "1" }
$log = Join-Path $Root ("stratarpcg-worker-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
Write-Host "StrataRPCG worker"
Write-Host "  engine: $exe"
Write-Host "  config: $Config"
Write-Host ("  args:   " + ($a -join " "))
Write-Host "  log:    $log"
Write-Host "  Loading takes several minutes; then it prints 'READY layer-stage worker' and waits for the 4090 PC."
if ($DryRun) { Write-Host "DRYRUN worker: $exe $($a -join ' ')"; exit 0 }
if ($cfg.cwd) { Set-Location $cfg.cwd }
$ErrorActionPreference = "Continue"
& $exe @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
exit $LASTEXITCODE
