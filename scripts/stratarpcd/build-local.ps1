# Builds strata-dsv41.exe (the StrataRPCD engine) on this PC, as the GitHub workflow does (it runs this script too).  The
# source is the stratarpcd ZIP extracted into a folder of its own (C:\StrataRPCD-project); the result is a folder to copy
# into C:\StrataRPCD\engine-rpcd on each PC: <source>\build-rpcd\engine-rpcd (the exe, the cuBLAS DLLs, the start scripts).
# Once: Visual Studio 2022 Build Tools (C++) and the NVIDIA CUDA Toolkit 13.2 (README-STRATARPCD.md, "Building the engine
# yourself").  Then, after extracting a new source ZIP over the old one:
#   cd C:\StrataRPCD-project
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcd\build-local.ps1 -Python C:\StrataRPCD\.venv\Scripts\python.exe
# Stop the model on this PC while it builds: the compiler needs the RAM.
param(
    [string]$Root = "",              # the source folder (default: the one this script is in, two levels up)
    [string]$Python = "",            # a Python with pip (default: <source>\.venv, else C:\StrataRPCD\.venv, else the one on PATH)
    [string]$Arch = "89-real;120",   # the GPUs: RTX 40 (sm_89) and RTX 50 (sm_120), so one exe runs on both PCs
    [int]$Jobs = 0,                  # compiles at a time (default: the CPU's threads, at most 8: each can take a few GB of RAM)
    [string]$Cuda = "",              # the CUDA Toolkit folder (default: the newest v13.x installed)
    [string]$Install = "",           # also copy the result into this engine folder (C:\StrataRPCD\engine-rpcd); stop the model there first
    [string]$BuildId = "",           # what --version prints (default: local-<date>-<time>)
    [switch]$Clean                   # build from scratch
)
$ErrorActionPreference = "Stop"
$started = Get-Date
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) {
    foreach ($c in @((Split-Path -Parent (Split-Path -Parent $here)), (Split-Path -Parent $here))) {
        if (Test-Path (Join-Path $c "src\dsv41\ds4_stage.cpp")) { $Root = $c; break }
    }
    if (-not $Root) { throw "the StrataRPCD source is not above ${here} (pass -Root C:\StrataRPCD)" }
}
$Root = (Resolve-Path $Root).Path
if (-not (Test-Path (Join-Path $Root "src\dsv41\ds4_stage.cpp"))) { throw "$Root is not the StrataRPCD source folder (no src\dsv41\ds4_stage.cpp)" }
if ($Jobs -lt 1) { $Jobs = [Math]::Max(1, [Math]::Min([Environment]::ProcessorCount, 8)) }

# Python (the source's .venv, the StrataRPCD folder's, else the one on PATH), CMake and Ninja (requirements-build.txt into that
# Python when missing)
$py = $Python
if (-not $py) {
    foreach ($c in @((Join-Path $Root ".venv\Scripts\python.exe"), "C:\StrataRPCD\.venv\Scripts\python.exe")) {
        if (Test-Path $c) { $py = $c; break }
    }
}
if (-not $py) {
    $found = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $found) { throw "no Python: pass -Python C:\StrataRPCD\.venv\Scripts\python.exe (setup made it)" }
    $py = $found.Source
}
if (-not (Test-Path $py)) { throw "no Python at $py" }
$venv = Split-Path -Parent $py
function Find-Tool([string]$name) {
    foreach ($d in @($venv, (Join-Path $venv "Scripts"))) {
        $p = Join-Path $d "$name.exe"
        if (Test-Path $p) { return $p }
    }
    $g = Get-Command "$name.exe" -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
    return $null
}
$cmake = Find-Tool "cmake"
$ninja = Find-Tool "ninja"
if (-not $cmake -or -not $ninja) {
    Write-Host "Installing CMake and Ninja into $py ..."
    & $py -m pip install --disable-pip-version-check --quiet -r (Join-Path $Root "requirements-build.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip could not install CMake and Ninja (requirements-build.txt)" }
    $cmake = Find-Tool "cmake"
    $ninja = Find-Tool "ninja"
    if (-not $cmake -or -not $ninja) { throw "CMake or Ninja is still missing after pip" }
}

# Visual Studio 2022 (CUDA 13.2 takes no newer one) and its newest MSVC toolset: with VS 2019's (14.29) nvcc drops C++20
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio 2022 Build Tools are not installed (README-STRATARPCD.md, Building the engine yourself)" }
$vs = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath
if (-not $vs) { throw "no Visual Studio 2022 with the C++ tools (workload Microsoft.VisualStudio.Workload.VCTools)" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$msvc = Get-ChildItem (Join-Path $vs "VC\Tools\MSVC") -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
    Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
if (-not $msvc -or [version]$msvc.Name -lt [version]"14.30") { throw "Visual Studio 2022's own MSVC toolset (14.3x or 14.4x) is not installed in $vs" }

# the CUDA Toolkit: the compiler, the runtime and cuBLAS
if (-not $Cuda) {
    $base = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    $all = @(Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^v13\.\d+$' } |
        Sort-Object { [version]$_.Name.Substring(1) } -Descending)
    if ($all.Count -gt 0) { $Cuda = $all[0].FullName } elseif ($env:CUDA_PATH) { $Cuda = $env:CUDA_PATH }
}
if (-not $Cuda) { throw "no CUDA Toolkit 13.x found (README-STRATARPCD.md, Building the engine yourself); or pass -Cuda <folder>" }
$Cuda = (Resolve-Path $Cuda).Path
$nvcc = Join-Path $Cuda "bin\nvcc.exe"
$missing = @()
if (-not (Test-Path $nvcc)) { $missing += "nvcc" }
if (-not (Get-ChildItem (Join-Path $Cuda "nvvm") -Recurse -Filter cicc.exe -ErrorAction SilentlyContinue | Select-Object -First 1)) { $missing += "nvvm" }
if (-not (Test-Path (Join-Path $Cuda "include\cuda_runtime.h"))) { $missing += "cudart" }
if (-not (Test-Path (Join-Path $Cuda "include\cublas_v2.h"))) { $missing += "cublas_dev" }
if (-not (Get-ChildItem (Join-Path $Cuda "lib") -Recurse -Filter cublas.lib -ErrorAction SilentlyContinue | Select-Object -First 1)) { $missing += "cublas_dev (cublas.lib)" }
if ($missing.Count -gt 0) { throw "the CUDA Toolkit in $Cuda lacks: $($missing -join ', ') (install those components)" }

# llama.cpp (ggml, for the CPU lane's experts): setup's pinned copy, as a ZIP
$ErrorActionPreference = "Continue"
Push-Location $Root
try { $llama = & $py -c "import setup; print(setup.get_llama_cpp())" | Select-Object -Last 1 } finally { Pop-Location }
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0 -or -not $llama -or -not (Test-Path (Join-Path $llama "ggml\CMakeLists.txt"))) { throw "could not get the llama.cpp source (setup.get_llama_cpp)" }

# A source ZIP extracted over the old one keeps the files' commit times, which can be older than the last build's objects:
# Ninja would not see the change.  A file whose contents differ from the last build's gets the current time.
$bd = Join-Path $Root "build-rpcd"
if ($Clean -and (Test-Path $bd)) { Remove-Item -Recurse -Force $bd }
New-Item -ItemType Directory -Force $bd | Out-Null
$list = Join-Path $bd "source-hashes.txt"
$old = @{}
if (Test-Path $list) {
    foreach ($l in Get-Content $list) {
        $p = $l.Split("`t")
        if ($p.Count -eq 2) { $old[$p[1]] = $p[0] }
    }
}
$files = @(Get-Item (Join-Path $Root "CMakeLists.txt"))
foreach ($d in @("cmake", "src", "include", "third_party\ggml")) {   # (setup.py ENGINE_SOURCES)
    $x = Join-Path $Root $d
    if (Test-Path $x) { $files += @(Get-ChildItem $x -Recurse -File) }
}
$now = Get-Date
$hashes = New-Object System.Collections.Generic.List[string]
$touched = 0
foreach ($f in $files) {
    $h = (Get-FileHash -Algorithm SHA1 -LiteralPath $f.FullName).Hash
    $rel = $f.FullName.Substring($Root.Length + 1)
    if ($old.Count -gt 0 -and $old[$rel] -ne $h) {
        $f.LastWriteTime = $now
        $touched++
    }
    $hashes.Add("$h`t$rel")
}
if ($old.Count -gt 0) { Write-Host "$touched source files changed since the last build" }

# configure and build in a cmd with VS 2022's environment (/d: no AutoRun, so nothing else sets up another toolset first).  The
# build's own cmd.exe children run cmd's AutoRun all the same: `conda init cmd.exe` puts its activation there, and with a
# conda CUDA that activation selects VS 2019's toolset before each compile (nvcc then drops C++20).
foreach ($k in @("HKCU:\Software\Microsoft\Command Processor", "HKLM:\Software\Microsoft\Command Processor")) {
    $ar = (Get-ItemProperty -Path $k -Name AutoRun -ErrorAction SilentlyContinue).AutoRun
    if ($ar) { Write-Warning "cmd's AutoRun ($k) runs before every compile: $ar - if the build fails with C++20 errors, clear it for the build" }
}
$id = if ($BuildId) { $BuildId } else { "local-" + (Get-Date -Format "yyyyMMdd-HHmm") }
$cmdFile = Join-Path $bd "build.cmd"
$cmdLines = @(
    "@echo off",
    "set `"VSCMD_ARG_VCVARS_VER=`"",
    "set `"VCToolsVersion=`"",
    "call `"$vcvars`" -vcvars_ver=$($msvc.Name) >nul",
    "if errorlevel 1 exit /b 1",
    "`"$cmake`" -S `"$Root`" -B `"$bd`" -G Ninja `"-DCMAKE_MAKE_PROGRAM=$ninja`" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DCMAKE_CUDA_FLAGS=-std=c++20 `"-DCMAKE_CUDA_COMPILER=$nvcc`" `"-DCUDAToolkit_ROOT=$Cuda`" `"-DCMAKE_CUDA_ARCHITECTURES=$Arch`" `"-DSTRATA_GGML_DIR=$llama`" -DSTRATA_BUILD_ID=$id",
    "if errorlevel 1 exit /b 1",
    "`"$cmake`" --build `"$bd`" --parallel $Jobs --target strata-dsv41",
    "if errorlevel 1 exit /b 1"
)
[System.IO.File]::WriteAllLines($cmdFile, $cmdLines)
Write-Host "Building strata-dsv41 ($Arch) with MSVC $($msvc.Name) and $Cuda, $Jobs at a time, in $bd ..."
$ErrorActionPreference = "Continue"
& cmd.exe /d /c $cmdFile
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($code -ne 0) { throw "the build failed (the lines above say where); -Clean starts it afresh" }
[System.IO.File]::WriteAllLines($list, $hashes)
$exe = Join-Path $bd "strata-dsv41.exe"
if (-not (Test-Path $exe)) { throw "the build finished without $exe" }

# the folder to copy into C:\StrataRPCD\engine-rpcd: the exe, the cuBLAS DLLs it loads (cudart is linked in), the start scripts
$out = Join-Path $bd "engine-rpcd"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item $exe $out
foreach ($n in @("cublas64_13.dll", "cublasLt64_13.dll")) {
    $src = Get-ChildItem (Join-Path $Cuda "bin") -Recurse -Filter $n -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $src) { throw "$n is not under $Cuda\bin: the exe needs it beside it" }
    Copy-Item $src.FullName $out
}
Copy-Item (Join-Path $here "start-*.ps1") $out
$configs = Join-Path $Root "configs\stratarpcd"
if (Test-Path $configs) {
    Copy-Item (Join-Path $configs "*.json") $out
}
$instructions = Join-Path $Root "docs\STRATARPCD-FIRST-RUN.md"
if (Test-Path $instructions) { Copy-Item $instructions $out }
Set-Content (Join-Path $out "BUILD.txt") "$id, CUDA $(Split-Path -Leaf $Cuda), MSVC $($msvc.Name), $Arch"
Write-Host ""
Write-Host "Built: $out"
Get-ChildItem $out | Format-Table Name, Length | Out-String | Write-Host
if ($Install) {
    New-Item -ItemType Directory -Force $Install | Out-Null
    $dst = Join-Path $Install "strata-dsv41.exe"
    try {
        if (Test-Path $dst) { Copy-Item $dst (Join-Path $Install "strata-dsv41.prev.exe") -Force }
        Copy-Item (Join-Path $out "*") $Install -Force
    } catch { throw "could not copy into $Install (is the model still running on this PC?): $_" }
    $ErrorActionPreference = "Continue"
    & $dst --version
    $ErrorActionPreference = "Stop"
    Write-Host "Installed into $Install (the previous exe is strata-dsv41.prev.exe)"
} else {
    Write-Host "Copy everything in it into C:\StrataRPCD\engine-rpcd on each PC (with the model stopped there)."
}
Write-Host ("Done in {0:N1} minutes." -f ((Get-Date) - $started).TotalMinutes)
exit 0
