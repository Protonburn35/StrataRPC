# Which flags make this runner's nvcc compile C++20 (the dialect the engine's sources are written in), for the
# StrataRPCD Windows build.  Run it after VsDevCmd, so cl.exe is on PATH.  It writes the first flag set that compiles
# the probe to -Out (one line, for CMAKE_CUDA_FLAGS) and exits 0; when none does, it prints what nvcc ran and exits 1.
param(
    [Parameter(Mandatory = $true)][string]$Nvcc,
    [Parameter(Mandatory = $true)][string]$Out
)
$ErrorActionPreference = "Continue"
$d = Join-Path $env:RUNNER_TEMP "nvcc-dialect"
New-Item -ItemType Directory -Force $d | Out-Null
$src = Join-Path $d "probe.cu"
$obj = Join-Path $d "probe.obj"
# a nested namespace (C++17: device.hpp), a concept and a lambda with template parameters (C++20: iq_kernels.cu)
[System.IO.File]::WriteAllText($src, @'
namespace a::b { __host__ __device__ inline int one() { return 1; } }
template <class T> concept word = sizeof(T) <= 8;
template <word T> __device__ T same(T x) { return x; }
__global__ void k(int* p) { auto g = []<class T>(T x) { return x; }; *p = g(same(a::b::one())); }
int main() { return 0; }
'@)

$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
Write-Output ("cl.exe: " + $(if ($cl) { $cl.Source } else { "not on PATH" }))
if ($cl) { & cl.exe 2>&1 | Select-Object -First 1 | ForEach-Object { Write-Output "  $_" } }
Get-ChildItem env: | Where-Object { $_.Name -like "NVCC*" -or $_.Name -eq "CUDAHOSTCXX" } |
    ForEach-Object { Write-Output "env $($_.Name)=$($_.Value)" }

$tries = @("-std=c++20", "-std=c++20 -Xcompiler=/Zc:__cplusplus", "-std=c++20 -allow-unsupported-compiler",
           "-std=c++20 -Xcudafe=--c++20", "")
$found = $null
foreach ($t in $tries) {
    $flags = @($t -split " " | Where-Object { $_ })
    if (Test-Path $obj) { Remove-Item $obj }
    $o = & $Nvcc -c $src -o $obj -arch=sm_89 @flags 2>&1 | Out-String
    $ok = ($LASTEXITCODE -eq 0) -and (Test-Path $obj)
    Write-Output ("probe [" + $(if ($t) { $t } else { "no flag" }) + "]: " + $(if ($ok) { "compiles" } else { "fails" }))
    if (-not $ok) { ($o -split "`r?`n" | Where-Object { $_ } | Select-Object -First 4) | ForEach-Object { Write-Output "    $_" } }
    if ($ok -and $null -eq $found) { $found = $t }
}

if ($null -eq $found) {
    Write-Output "nvcc --dryrun -std=c++20 (what it runs; long lines cut):"
    & $Nvcc --dryrun -c $src -o $obj -arch=sm_89 -std=c++20 2>&1 | ForEach-Object {
        $s = "$_"; Write-Output ("  " + $s.Substring(0, [Math]::Min(600, $s.Length)))
    }
    Write-Output "::error::no flag set makes this nvcc compile C++20 (see the probe lines above)"
    exit 1
}
[System.IO.File]::WriteAllText($Out, $found)
Write-Output "CMAKE_CUDA_FLAGS for the build: [$found]"
exit 0
