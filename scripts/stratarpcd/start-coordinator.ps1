# StrataRPCD coordinator (PC A): the API server; its engine runs DeepSeek-V4.1-Flash's embedding, Engram and layers 0-19
# here and hands each token's rows to the worker on PC C (start-worker.ps1 there first).
# Run it from the StrataRPCD folder once setup has made the model's pack on this PC (README-STRATARPCD.md):
#   $env:STRATA_RPC_TOKEN = "the same long secret as on PC C"
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-coordinator.ps1 -Worker 192.168.50.143:39501
# The API is then at http://127.0.0.1:<port>/v1 (the port setup chose, or -Port). Ctrl+C stops it.
param(
    [string]$Worker = "",   # PC C's address and port
    [string]$Root = "",                  # the StrataRPCD folder (default: the folder above this script's)
    [string]$Config = "",                # setup's run config (default: the strata-deepseek-v4.1-flash-*.json in $Root)
    [int]$Split = 0,                    # the worker's first layer: the same on both PCs
    [ValidateSet("fp32", "bf16", "fp16")][string]$Wire = "bf16",   # the rows on the link, the same on both PCs: bf16 (half the bytes), fp32 exact
    [int]$Context = 0,                   # 0: the config's --max-context (it must match the worker's)
    [int]$PrefillChunk = 0,              # the prompt chunk on this PC (0: the engine's own, 8192 on a 24 GB card): smaller leaves more VRAM for experts
    [int]$Port = 0,                      # 0: the config's API port
    [int]$MiddleEnd = 0,                # 0: config value or A+C; 24: opt-in A+B+C
    [switch]$Vision,                     # encode images on A's GPU; image routing on both PCs
    [switch]$Sessions,                   # opt-in manual paired disk sessions, on both PCs
    [string]$SessionDir = "",            # this PC's own session folder
    [switch]$Stats,                      # STRATA_LAYER_STAGE_STATS=1: each prompt's and every 256 steps' time in the log
    [switch]$DryRun                      # print what it would run and stop (nothing starts)
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = Split-Path -Parent $here }
$exe = Join-Path $here "strata-dsv41.exe"
if (-not (Test-Path $exe)) { throw "strata-dsv41.exe is not beside this script ($here)" }
if (-not $env:STRATA_RPC_TOKEN) { throw 'set $env:STRATA_RPC_TOKEN = "<the same long secret as on PC C>" in this window first' }
if (-not $Config) {
    $found = @(Get-ChildItem -Path $Root -Filter "strata-deepseek-v4.1-flash-*.json" | Where-Object { $_.Name -notlike "*stratarpcd*" })
    if ($found.Count -ne 1) { throw "found $($found.Count) setup configs (strata-deepseek-v4.1-flash-*.json) in ${Root}: pass -Config" }
    $Config = $found[0].FullName
}
$cfg = Get-Content $Config -Raw | ConvertFrom-Json

# setup's engine arguments, then this PC's role; the config itself is left as setup wrote it
$a = New-Object System.Collections.Generic.List[string]
$given = @($cfg.args)
foreach ($setting in @(@("Split", "--layer-stage-split"), @("MiddleEnd", "--layer-stage-middle-end"), @("Worker", "--layer-stage-client"))) {
    $current = Get-Variable -Name $setting[0] -ValueOnly
    if (-not $current) {
        $index = [array]::IndexOf($given, $setting[1])
        if ($index -ge 0 -and $index + 1 -lt $given.Count) { Set-Variable -Name $setting[0] -Value $given[$index + 1] }
    }
}
if (-not $Split) { $Split = 20 }
if (-not $Worker) { throw "pass -Worker or use a config containing --layer-stage-client" }
if ($MiddleEnd -and $MiddleEnd -notin @(24, 28, 32, 36)) { throw "MiddleEnd must be 24, 28, 32 or 36" }
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
    if (-not $SessionDir) { $SessionDir = if ($cfg.slot_save_path) { $cfg.slot_save_path } else { Join-Path $Root "sessions" } }
    $cfg | Add-Member -NotePropertyName slot_save_path -NotePropertyValue $SessionDir -Force
}
foreach ($x in @("--layer-stage-client", $Worker, "--layer-stage-split", [string]$Split, "--layer-stage-wire", $Wire)) { $a.Add($x) }
if ($MiddleEnd) { $a.Add("--layer-stage-middle-end"); $a.Add([string]$MiddleEnd) }
$cfg.exe = $exe
$cfg.args = $a.ToArray()
if ($Port -gt 0) { $cfg.port = $Port }
$port = [int]$cfg.port
$out = Join-Path $Root $(if ($MiddleEnd) { "strata-stratarpcd-abc-coordinator.runtime.json" } else { "strata-stratarpcd-coordinator.json" })
[System.IO.File]::WriteAllText($out, ($cfg | ConvertTo-Json -Depth 20), (New-Object System.Text.UTF8Encoding $false))

if ($Stats) { $env:STRATA_LAYER_STAGE_STATS = "1" }
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $Root "python\python.exe" }
if (-not (Test-Path $py) -and -not $DryRun) { throw "no Python in $Root (.venv or python\): run START-HERE.bat (setup) on this PC first" }
Write-Host "StrataRPCD coordinator"
Write-Host "  engine: $exe"
Write-Host "  config: $out (from $Config)"
Write-Host ("  args:   " + ($a -join " "))
Write-Host "  worker: $Worker  (it must be running: start-worker.ps1 on PC C)"
Write-Host "  API:    http://127.0.0.1:$port/v1  (the engine's log: $($cfg.log))"
if ($DryRun) { Write-Host "DRYRUN coordinator: $py serve\server.py --engine strata --config $out --port $port"; exit 0 }
Set-Location $Root
& $py (Join-Path $Root "serve\server.py") --engine strata --config $out --port $port
exit $LASTEXITCODE
