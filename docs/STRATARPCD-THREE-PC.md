# Optional A + B + C, alongside the existing A + C preset

This is an experimental three-PC mode. Its speed and GPU correctness have not yet been measured on these PCs.
Keep the working two-PC preset for comparison. Both modes use the same new protocol-3 binary; update every
participating PC together. Model packs stay unchanged.

[Download the Windows engine package](https://github.com/Protonburn35/StrataRPC/actions/runs/40853673787/artifacts/12889408046)
(407 MB; [successful build](https://github.com/Protonburn35/StrataRPC/actions/runs/40853673787)). The binary reports
`strata-dsv41 0.1.52 build bf4c05624`. This package passed the Windows build and CPU/launcher checks, not a
three-GPU inference benchmark.

| PC | Address | Three-PC work |
| --- | --- | --- |
| A: RTX 4090, 32 GB RAM | 192.168.50.206 | API, GPU vision, embedding, Engram, layers 0-19 |
| B: Ryzen AI 9 HX 370, RTX 4060 Laptop 8 GB, 32 GB RAM | 192.168.50.129 | layers 20-23; forwards to C |
| C: RTX 5070 Ti, 32 GB RAM | 192.168.50.143 | reconstructs shared source 20, then layers 24-39, output head and sampling |

DeepSeek's later layers share layer 20's compressed KV and attention candidates. An ordinary cut at 24 would
lose them. B therefore forwards both its original input and its output. C repeats layer 20 from that original
input, then replaces its residual streams with B's output before layer 24. Layer 24 computes its own indexer
picks. No growing KV cache crosses the link. C loads 17 layers instead of 20; B loads four. The repeated layer
adds work, so extra GPU/RAM capacity does not establish a speedup. BF16 remains the user's chosen wire format.

The opt-in flag is `--layer-stage-middle-end 24` on all three PCs. The model validates the dependency layout;
other supported indexer boundaries are 28, 32 and 36. Start with the supplied 24 preset. B forwards prompts
in at most 2048-token requests, bounding its retained input and C's paired-row buffer. The three-PC path
processes every prompt row rather than using the native prompt-skip optimization. This can cost prefill time.
All six experts remain selected. There is no DSpark, forced temperature, expert pruning, or new weight quantization.
The extra BF16 activation boundary and execution on different GPUs can change floating-point results and sampled
answers. Pass `-Wire fp32` to all three launchers for numerical comparisons; normal sampling remains supported in both modes.

## Prepare the laptop once

Use this fork's source/setup, in `C:\StrataRPCD`. Install **IST-DAS 2.25 bpw**, not the setup's 2.00 default.
In Command Prompt, from the source folder:

```cmd
cd /d C:\StrataRPCD
START-HERE.bat --family deepseek41 --model IST-DAS-2.25bpw --no-start
```

Each PC still needs the complete 2.25 shards and shared assets locally. A pack is an index into those GGUFs;
it is not an independent copy of the weights. Setup's stock engine does not establish three-PC support:
install the new StrataRPCD engine package into `C:\StrataRPCD\engine-rpcd` on all three PCs afterwards.
The RTX 4060 Laptop uses the package's existing sm_89 build, as the 4090 does.

## Configs and startup

The existing `strata-stratarpcd-coordinator.json` and `strata-stratarpcd-worker.json` remain the **A+C** presets.
The new **A+B+C** presets, also included in the engine package, are:

- A: `strata-stratarpcd-abc-a.json`
- B: `strata-stratarpcd-abc-b.json`
- C: `strata-stratarpcd-abc-c.json`

They use context 1048576, BF16, GPU 0, six experts, no DSpark, GPU vision on A, and manual sessions on all PCs.
They refer to `C:\Strata-data\packs\deepseek-v4.1-flash-ist-das-2.25bpw` and A's API listens on port 8081
on LAN (`host: 0.0.0.0`). The coordinator writes a separate `strata-stratarpcd-abc-coordinator.runtime.json`;
it does not overwrite either input preset. Edit input presets, not generated runtime files.

Stop the engines before switching modes. In each new PowerShell window set the **same existing RPC token**:

```powershell
cd C:\StrataRPCD
$env:STRATA_RPC_TOKEN = "<your existing shared token>"
Get-ChildItem .\engine-rpcd -Recurse | Unblock-File
```

Start **C first**:

```powershell
cd C:\StrataRPCD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1 -Config .\engine-rpcd\strata-stratarpcd-abc-c.json -Listen 192.168.50.143:39501 -Stats
```

Wait for READY, then start **B**:

```powershell
cd C:\StrataRPCD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1 -Config .\engine-rpcd\strata-stratarpcd-abc-b.json -Listen 192.168.50.129:39501 -Stats
```

B connects to C, then prints READY. Finally start **A**:

```powershell
cd C:\StrataRPCD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-coordinator.ps1 -Config .\engine-rpcd\strata-stratarpcd-abc-a.json -Stats
```

The API endpoint remains `http://192.168.50.206:8081/v1`. The laptop can keep being a chat/OpenCode client
while also running B. Use the API on A, not the native worker port.

For **A+C**, stop all three engines, then use the original two presets and commands in
[STRATARPCD-FIRST-RUN.md](STRATARPCD-FIRST-RUN.md). B is not needed. C then runs layers 20-39 as before.

## Firewall additions

Run these once in **administrator PowerShell**. Keep C's existing allowance for A, for two-PC operation.

On B, allow A to reach B's worker:

```powershell
cd C:\StrataRPCD
New-NetFirewallRule -DisplayName "StrataRPCD A to B" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 39501 -RemoteAddress 192.168.50.206 -Profile Any
```

On C, also allow B to reach C's worker:

```powershell
cd C:\StrataRPCD
New-NetFirewallRule -DisplayName "StrataRPCD B to C" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 39501 -RemoteAddress 192.168.50.129 -Profile Any
```

A's existing API rule must continue to admit the laptop on 8081. RPC authentication still uses the shared token.

## Manual saved states

The existing `/slots/0?action=save|restore` API remains on A. C publishes its local file first, then B, then A.
All three files share a snapshot ID; only descriptors cross RPC. A restore reconnects and loads C, B, then A.
Connection failures or partial restores reset the chain. Keep all three files.

The three-PC presets use separate folders:

- A: `C:\StrataRPCD\sessions-abc\<name>.bin`
- B: `C:\StrataRPCD\sessions-abc-b\<id>.srpc`
- C: `C:\StrataRPCD\sessions-abc-c\<id>.srpc`

Two-PC and three-PC states are incompatible and explicitly refused across modes. Existing protocol-2 two-PC
session files retain their envelope format and can still be read with the same two-PC model/settings.
Manual snapshots retain the existing RAM/disk guards and 8 GiB per-file cap. They are not chat transcripts:
continue the same client history and attachments. Automatic disk saving is not enabled by this change.

If using the user's separate `Chat-Sessions.ps1` helper **on A**, point its local file checks at the new folder:

```powershell
cd C:\Strata\Restore-Scripts
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Chat-Sessions.ps1 -ApiUrl http://192.168.50.206:8081 -ChatsDirectory C:\StrataRPCD\sessions-abc
```

Adjust only that helper's installed location if different. On the laptop it still needs a Windows share of A's
snapshot folder for its local list/check operations; the API itself accepts remote save/restore requests.

## First hardware checks

Compare A+C and A+B+C on the same prompt, context, model, sampling settings and generated length. Start with
a short text chat and an image/follow-up. Then save, switch chat, restore, restart all engines and restore again.
Check reused prompt tokens and keep logs from all three PCs. Test a long prompt only after these checks pass.
The declared 1M context does not prove a full 1M-token snapshot fits the existing memory/file guards.

With `-Stats`, A's decode line gives total feed time and the time waiting on B+C; B's gives its feed time and
the time waiting on C. Compare those along with GPU-resident experts, CPU expert work, disk read latency and
RAM usage. No network-bandwidth bottleneck or output-speed gain is assumed.
