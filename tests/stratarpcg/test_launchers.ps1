# StrataRPCG's start scripts (scripts/stratarpcg/*.ps1) on a made-up setup: the worker's command line (dry runs) and
# the coordinator's config, which it writes and hands to a stand-in maya.py.  No GPU, no model.  Windows PowerShell 5.1
# (CI) or PowerShell 7:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\stratarpcg\test_launchers.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$onWindows = [IO.Path]::DirectorySeparatorChar -eq '\'
$ps = if ($onWindows -and $PSVersionTable.PSEdition -ne "Core") { "powershell" } else { (Get-Process -Id $PID).Path }
$t = Join-Path ([IO.Path]::GetTempPath()) ("stratarpcg-launchers-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$failed = 0

function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Output "ok   $what" } else { Write-Output "FAIL $what"; $script:failed++ }
}
function Run([string]$script, [string[]]$more) {
    $ErrorActionPreference = "Continue"   # (5.1 under Stop: the child's stderr through 2>&1 would end this script)
    $f = Join-Path $t "scripts/stratarpcg/$script"
    $out = & $ps -NoProfile -ExecutionPolicy Bypass -File $f @more 2>&1 | Out-String
    return @{ out = $out; code = $LASTEXITCODE }
}
function Args-Of([string]$path) {
    return @((Get-Content $path -Raw | ConvertFrom-Json).args)
}

try {
    New-Item -ItemType Directory -Force (Join-Path $t "scripts/stratarpcg"), (Join-Path $t "engine"), (Join-Path $t "packs/glm") | Out-Null
    Copy-Item (Join-Path $repo "scripts/stratarpcg/start-worker.ps1"), (Join-Path $repo "scripts/stratarpcg/start-coordinator.ps1") (Join-Path $t "scripts/stratarpcg")
    $exe = Join-Path $t "engine/maya.exe"
    Set-Content $exe "not an exe"
    # setup's config as maya.py's write_config writes it (two GPUs and --parallel, as a PC of its own could have)
    $cfg = [ordered]@{
        exe = $exe; cwd = $t; tokenizer = (Join-Path $t "packs/glm/tokenizer"); model_name = "GLM-5.3-Flash"
        args = @("--glm-pack", (Join-Path $t "packs/glm"), "--max-context", "524288", "--expert-cache", "auto", "--prefill", "auto",
                 "--kv-streaming", "on", "--parallel", "2", "--dflash", (Join-Path $t "d.gguf"), "--expert-profile", (Join-Path $t "p.txt"))
        gpu = @(0, 1); sampling = [ordered]@{ temperature = 1.0; top_p = 0.95 }; lib_dirs = @((Join-Path $t "engine")); port = 8080
        env = [ordered]@{ STRATA_TEST_SETTING = "7" }
        vision = [ordered]@{ exe = (Join-Path $t "v.exe"); mmproj = "m.gguf"; model = "v.gguf"; gpu = $true }
        log = (Join-Path $t "maya-gsq-rco-3.0bit.log")
    }
    $cfgPath = Join-Path $t "maya-gsq-rco-3.0bit.json"
    [IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 10))
    # the coordinator starts .venv's Python with maya.py: a stand-in that only says it ran
    $venv = Join-Path $t ".venv/Scripts"
    New-Item -ItemType Directory -Force $venv | Out-Null
    if ($onWindows) {
        & python -m venv (Join-Path $t ".venv")
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    } else {
        $py3 = (Get-Command python3).Source
        New-Item -ItemType SymbolicLink -Path (Join-Path $venv "python.exe") -Target $py3 | Out-Null
    }
    Set-Content (Join-Path $t "maya.py") "import sys`nprint('STAND-IN maya.py ' + ' '.join(sys.argv[1:]))`n"

    $env:STRATA_RPC_TOKEN = $null
    Remove-Item Env:STRATA_RPC_TOKEN -ErrorAction SilentlyContinue

    # the worker: setup's arguments without --parallel / --dflash, then its role
    $r = Run "start-worker.ps1" @("-DryRun")
    $line = ($r.out -split "`n" | Where-Object { $_ -like "DRYRUN worker:*" } | Select-Object -First 1)
    Check ($r.code -eq 0 -and $line) "worker dry run without a token"
    Check ($line -match "--glm-pack \S+ --max-context 524288 --expert-cache auto --prefill auto --kv-streaming on --expert-profile \S+ --layer-stage-server 0\.0\.0\.0:39501 --layer-stage-split 24 --layer-stage-wire bf16 --layer-stage-gpu 0\s*$") "worker default command"
    Check ($line -notmatch "--parallel|--dflash|--layer-stage-head|--layer-stage-sessions") "worker drops --parallel and --dflash, no opt-ins"
    $r = Run "start-worker.ps1" @("-DryRun", "-HeadOnCoordinator", "-Sessions", "-Context", "262144", "-Split", "22", "-Wire", "fp32", "-Listen", "0.0.0.0:39600")
    $line = ($r.out -split "`n" | Where-Object { $_ -like "DRYRUN worker:*" } | Select-Object -First 1)
    Check (([regex]::Matches($line, "--max-context")).Count -eq 1 -and $line -match "--max-context 262144") "worker -Context replaces the context"
    Check ($line -match "--layer-stage-server 0\.0\.0\.0:39600 --layer-stage-split 22 --layer-stage-wire fp32 --layer-stage-gpu 0 --layer-stage-head coordinator --layer-stage-sessions --layer-stage-session-dir \S*sessions-worker") "worker opt-ins"
    $r = Run "start-worker.ps1" @()
    Check ($r.code -ne 0 -and $r.out -match "STRATA_RPC_TOKEN") "worker refuses to start without the token"

    # the coordinator: its config beside setup's, then maya.py with it
    $env:STRATA_RPC_TOKEN = "test-secret"
    $stageCfg = Join-Path $t "maya-gsq-rco-3.0bit-stratarpcg.json"
    $r = Run "start-coordinator.ps1" @()
    Check ($r.code -eq 0 -and $r.out -match "STAND-IN maya\.py --config \S*maya-gsq-rco-3\.0bit-stratarpcg\.json --port 8081 --host 127\.0\.0\.1") "coordinator starts maya.py with its config"
    $c = Get-Content $stageCfg -Raw | ConvertFrom-Json
    $a = @($c.args)
    Check (($a -join " ") -match "--kv-streaming on --expert-profile \S+ --layer-stage-client 192\.168\.50\.143:39501 --layer-stage-split 24 --layer-stage-wire bf16 --layer-stage-gpu 0$") "coordinator default arguments"
    Check (-not ($a -contains "--parallel") -and -not ($a -contains "--dflash")) "coordinator drops --parallel and --dflash"
    Check ((@($c.gpu).Count -eq 1) -and (@($c.gpu)[0] -eq 0) -and $c.port -eq 8081 -and $c.host -eq "127.0.0.1") "coordinator's GPU, port and host"
    Check ($null -ne $c.vision -and -not $c.PSObject.Properties["slot_save_path"]) "pictures on, saved chats off by default"
    Check ($c.env.STRATA_TEST_SETTING -eq "7" -and $c.sampling.top_p -eq 0.95 -and @($c.lib_dirs).Count -eq 1) "setup's settings kept"
    $raw = Get-Content $stageCfg -Raw
    Check ($raw -notmatch '"Count"' -and $raw -notmatch '"value"') "arrays written as arrays (Windows PowerShell 5.1)"
    $r = Run "start-coordinator.ps1" @("-HeadOnCoordinator", "-Sessions", "-NoVision", "-Context", "262144", "-Split", "22", "-Worker", "10.0.0.2:39600", "-Port", "8090", "-Gpu", "1")
    $c = Get-Content $stageCfg -Raw | ConvertFrom-Json
    $a = @($c.args) -join " "
    Check ($r.code -eq 0 -and $a -match "--max-context 262144" -and ([regex]::Matches($a, "--max-context")).Count -eq 1) "coordinator -Context"
    Check ($a -match "--layer-stage-client 10\.0\.0\.2:39600 --layer-stage-split 22 --layer-stage-wire bf16 --layer-stage-gpu 0 --layer-stage-head coordinator --layer-stage-sessions$") "coordinator opt-ins"
    Check ($null -eq $c.vision -and $c.slot_save_path -like "*sessions-host" -and @($c.gpu)[0] -eq 1 -and $c.port -eq 8090) "coordinator -NoVision, -Sessions, -Gpu, -Port"
    $r = Run "start-coordinator.ps1" @()
    $c = Get-Content $stageCfg -Raw | ConvertFrom-Json
    Check ($null -ne $c.vision -and -not $c.PSObject.Properties["slot_save_path"] -and (@($c.args) -join " ") -notmatch "head|sessions") "a start without the opt-ins turns them off again"

    # setup's config is left as it was
    $orig = Get-Content $cfgPath -Raw | ConvertFrom-Json
    Check ((@($orig.args) -join " ") -match "--parallel 2" -and (@($orig.args) -join " ") -notmatch "layer-stage") "setup's own config untouched"

    # two GLM configs: the scripts ask which
    [IO.File]::WriteAllText((Join-Path $t "maya-other.json"), ($cfg | ConvertTo-Json -Depth 10))
    $r = Run "start-worker.ps1" @("-DryRun")
    Check ($r.code -ne 0 -and $r.out -match "pass -Config") "two GLM configs: -Config asked for"
    $r = Run "start-worker.ps1" @("-DryRun", "-Config", $cfgPath)
    Check ($r.code -eq 0) "-Config picks one"
} finally {
    Remove-Item Env:STRATA_RPC_TOKEN -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue
}
if ($failed -gt 0) { Write-Output "$failed check(s) failed"; exit 1 }
Write-Output "all launcher checks passed"
exit 0
