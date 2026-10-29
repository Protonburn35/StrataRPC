# StrataRPCG coordinator (the 4090 PC): the API and the web page, the embedding, pictures and layers 0 .. Split-1;
# layers Split .. 44 run on the worker (the 5070 Ti PC), which must be started first and show READY.  Run it from the
# StrataRPCG folder once START-HERE.bat has set the model up on this PC (docs\STRATARPCG.md):
#   $env:STRATA_RPC_TOKEN = "the same long secret as on the 5070 Ti PC"
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-coordinator.ps1
# It writes <config>-stratarpcg.json (setup's config with the layer stage added) and starts Project Maya with it:
# the page and the API at http://127.0.0.1:8081 by default.  -Split, -Wire, -HeadOnCoordinator, -Context and
# -Sessions must be the same on both PCs.
param(
    [string]$Root = "",                       # the StrataRPCG folder (default: two folders above this script)
    [string]$Config = "",                     # setup's config (default: the one maya-*.json for GLM in $Root)
    [string]$Worker = "192.168.50.143:39501", # the worker's address and port
    [int]$Split = 24,                         # the worker's first layer (1-44): the same on both PCs
    [ValidateSet("fp32", "bf16", "fp16")][string]$Wire = "bf16",   # the rows on the link: bf16 (half the bytes), fp32 exact
    [switch]$HeadOnCoordinator,               # the output head and the sampling on this PC (both PCs)
    [int]$Context = 0,                        # 0: the config's --max-context (it must match the worker's)
    [int]$Gpu = 0,                            # this PC's GPU (nvidia-smi's number)
    [int]$Port = 8081,                        # the API and the page
    [string]$ApiHost = "127.0.0.1",           # where the API listens (127.0.0.1: this PC only)
    [switch]$NoVision,                        # text only (by default pictures are read when setup set them up)
    [switch]$Sessions,                        # opt-in paired disk sessions (saved chats), on both PCs
    [string]$SessionDir = "",                 # this PC's saved chats (default: <Root>\sessions-host)
    [switch]$Stats,                           # STRATA_LAYER_STAGE_STATS=1: each prompt's and every 256 tokens' times in the log
    [switch]$DryRun                           # print what it would run and stop (nothing starts)
)
$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 writes some arrays as {"value": [...], "Count": n} without this (PowerShell 7 has no such data)
Remove-TypeData System.Array -ErrorAction SilentlyContinue
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $here) }
if (-not $env:STRATA_RPC_TOKEN -and -not $DryRun) {
    throw 'set $env:STRATA_RPC_TOKEN = "<the same long secret as on the 5070 Ti PC>" in this window first'
}
if (-not $Config) {
    $found = @(Get-ChildItem -Path $Root -Filter "maya-*.json" | Where-Object {
        $_.Name -notlike "*stratarpcg*" -and $_.Name -notlike "*.shared-settings.json" -and
        (Get-Content $_.FullName -Raw) -match '"--glm-pack"' })
    if ($found.Count -ne 1) { throw "found $($found.Count) GLM configs (maya-*.json with --glm-pack) in ${Root}: pass -Config" }
    $Config = $found[0].FullName
}
$cfg = Get-Content $Config -Raw | ConvertFrom-Json
if (-not (Test-Path ([string]$cfg.exe))) { throw "the engine $($cfg.exe) is missing: run START-HERE.bat on this PC first" }
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py) -and -not $DryRun) { throw "$py is missing: run START-HERE.bat on this PC first" }

# setup's engine arguments without what a layer stage does not take, then this PC's role
$a = New-Object System.Collections.Generic.List[string]
$given = @($cfg.args)
$drop1 = @("--layer-stage-sessions")
$drop2 = @("--parallel", "--dflash", "--layer-split", "--split-device")
for ($i = 0; $i -lt $given.Count; $i++) {
    $k = [string]$given[$i]
    if ($drop1 -contains $k) { continue }
    if ($drop2 -contains $k -or $k -like "--layer-stage-*" -or ($Context -gt 0 -and $k -eq "--max-context")) { $i++; continue }
    $a.Add($k)
}
if ($Context -gt 0) { $a.Add("--max-context"); $a.Add([string]$Context) }
foreach ($x in @("--layer-stage-client", $Worker, "--layer-stage-split", [string]$Split, "--layer-stage-wire", $Wire,
                 "--layer-stage-gpu", "0")) { $a.Add($x) }
if ($HeadOnCoordinator) { $a.Add("--layer-stage-head"); $a.Add("coordinator") }
if ($Sessions) { $a.Add("--layer-stage-sessions") }

# the layer stage's config, beside setup's: the same model, this PC's GPU, the API's address
$stage = $cfg | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$stage.args = @($a)
$stage | Add-Member -NotePropertyName gpu -NotePropertyValue @($Gpu) -Force
$stage | Add-Member -NotePropertyName port -NotePropertyValue $Port -Force
$stage | Add-Member -NotePropertyName host -NotePropertyValue $ApiHost -Force
if ($NoVision -and $stage.PSObject.Properties["vision"]) { $stage.PSObject.Properties.Remove("vision") }
if ($Sessions) {
    if (-not $SessionDir) { $SessionDir = Join-Path $Root "sessions-host" }
    $stage | Add-Member -NotePropertyName slot_save_path -NotePropertyValue $SessionDir -Force
} elseif ($stage.PSObject.Properties["slot_save_path"]) {
    $stage.PSObject.Properties.Remove("slot_save_path")
}
$stageCfg = Join-Path (Split-Path -Parent $Config) ([IO.Path]::GetFileNameWithoutExtension($Config) + "-stratarpcg.json")
$stage.log = [IO.Path]::ChangeExtension($stageCfg, ".log")
if ($Stats) { $env:STRATA_LAYER_STAGE_STATS = "1" }
Write-Host "StrataRPCG coordinator"
Write-Host "  config: $stageCfg (from $Config)"
Write-Host ("  engine: " + $cfg.exe + " " + ($a -join " "))
Write-Host "  worker: $Worker (start it first and wait for READY)"
Write-Host ("  pictures: " + $(if ($stage.PSObject.Properties["vision"]) { "on" } else { "off" }))
if ($Sessions) { Write-Host "  saved chats: $SessionDir (POST /slots/0?action=save|restore)" }
Write-Host "  page and API: http://${ApiHost}:$Port"
if ($DryRun) {
    Write-Host "DRYRUN coordinator: $py maya.py --config $stageCfg --port $Port --host $ApiHost"
    Write-Host ("DRYRUN engine: " + $cfg.exe + " " + ($a -join " "))
    exit 0
}
$stage | ConvertTo-Json -Depth 20 | Set-Content -Path $stageCfg -Encoding UTF8
Set-Location $Root
$ErrorActionPreference = "Continue"
& $py (Join-Path $Root "maya.py") --config $stageCfg --port $Port --host $ApiHost
exit $LASTEXITCODE
