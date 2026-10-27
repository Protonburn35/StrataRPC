# StrataRPCD: DeepSeek-V4.1-Flash on two PCs

StrataRPCD runs DeepSeek-V4.1-Flash (IST-DAS 2.00 bpw) split by layers across two PCs, the way StrataRPC runs
Qwen3.8-Flash-Next.

| PC | Role | What it runs |
|---|---|---|
| A (RTX 4090) | coordinator | the API, the embedding table, the Engram, layers 0-19 |
| C (RTX 5070 Ti) | worker | layers 20-39, the head, sampling |

- **What crosses the network:** for each token, one row of 4 x 5120 numbers plus 8 more (80 KiB in FP32). The KV
  cache never crosses.
- **Quality:** with the default FP32 rows the answers are the same as on one PC, apart from the experts'
  summing order.
- **Full quality:** both PCs use the model's 6 experts per word (no "Faster" mode).

This is the first version. It has been built and its link has been tested without a GPU. It has **not** run on the two
PCs yet: you test it first.

## Once on each PC (A and C)

1. **Get the source.** Download the `stratarpcd` branch as a ZIP
   (https://github.com/Protonburn35/StrataRPC/archive/refs/heads/stratarpcd.zip) and extract it so that `setup.py` is at
   `C:\StrataRPCD\setup.py`. This is separate from `C:\Strata`; nothing there changes.
2. **Get the engine.** Download the `StrataRPCD-engine-win-x64-cuda13-sm89-sm120` artifact from the newest green
   "StrataRPCD Windows engine" run (https://github.com/Protonburn35/StrataRPC/actions). Extract it into
   `C:\StrataRPCD\engine-rpcd\`. It holds `strata-dsv41.exe`, the CUDA DLLs and the two start scripts.
3. **Download and prepare the model** with mainline's setup, pointed at this engine. In PowerShell:

   ```powershell
   cd C:\StrataRPCD
   $env:STRATA_DSV41_EXE = "C:\StrataRPCD\engine-rpcd\strata-dsv41.exe"
   .\START-HERE.bat --family deepseek41 --model IST-DAS-2.00bpw --vision no --no-start
   ```

   `--vision no` skips the picture reader (`mmproj-BF16.gguf`, 1 GB), which the two-PC split does not use. Setup would
   otherwise put it at `C:\StrataRPCD\models\deepseek-v4.1-flash-shared\mmproj-BF16.gguf` and build it into the pack.
   Either way the split works: the handshake ignores the vision weights.

   - **Download:** setup fetches 257 GB into `C:\StrataRPCD\models\` (the 138 GB model, the 111 GB Engram table and
     the 8 GB drafter).
   - **Pack:** it builds the pack under `C:\StrataRPCD\packs\` and writes `strata-deepseek-v4.1-flash-ist-das-2.00bpw.json`.
   - **Test:** it then has this one PC answer a 16-word test question, which checks the new engine on its own.
   - **If you already downloaded the GGUF files:** add `--gguf-dir <the folder with them>`.
   - **To skip the test:** set `$env:STRATA_DSV41_NO_SMOKE = "1"` before running setup.
4. **On C only, allow the worker's port** through the firewall for your home network. Run this in an
   **administrator** PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "StrataRPCD worker" -Direction Inbound -Protocol TCP -LocalPort 39501 -Action Allow -Profile Private
   ```

## Each time

Pick a long random secret and use the same one on both PCs, in every window you start them from.

**On C (worker) first.** Loading takes several minutes; wait for `READY layer-stage worker`:

```powershell
cd C:\StrataRPCD
$env:STRATA_RPC_TOKEN = "replace-with-a-long-random-secret"
powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1
```

**Then on A (coordinator):**

```powershell
cd C:\StrataRPCD
$env:STRATA_RPC_TOKEN = "replace-with-a-long-random-secret"
powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-coordinator.ps1 -Worker 192.168.50.143:39501 -Port 8081
```

- **A's engine** loads its half, then connects to C. If C is still loading, A waits for it, up to 30 minutes.
- **The API** is then at `http://127.0.0.1:8081/v1` (OpenAI) and `http://127.0.0.1:8081/v1/messages` (Anthropic), as
  with Strata.
- **Ctrl+C** stops either side. If you restart C alone, A connects again at the next new chat.

### Switches (both scripts)

| Switch | Default | What it does |
|---|---|---|
| `-Wire bf16` | `fp32` | Sends the rows in BF16: half the bytes on the link, but the numbers change slightly (lossy). Use the same on both PCs. |
| `-Stats` | off | Writes each prompt's time, the MB sent and the time spent waiting on C into the log. On A it also logs every 256 tokens. |
| `-Context N` | setup's (8192) | The context. It must be the same on both PCs. |
| `-Split 20` | 20 | The worker's first layer. 20 is the only clean split for V4.1, so leave it. |
| `-DryRun` | | Prints the command and stops. |
| `-Listen` (C) | `0.0.0.0:39501` | The worker's address and port. |
| `-Port` (A) | setup's | The API port. |

## What to test, in order

1. **Each PC alone:** setup's 16-word test passed on both.
2. **Both PCs:** a short chat on A's API. Check that the answer reads well.
3. **Speed:** a 2,000-4,000-token prompt and a long answer, with `-Stats` on both PCs. Note the prompt's tokens/s and
   the answer's tokens/s from A's server log.
4. **Stop:** press Stop during a long prompt, then send another message. Both sides should go on without reloading.
5. **Optional:** the same with `-Wire bf16` on both PCs. Compare the speed and whether the answers change.

Send back:
- A's log (`strata-deepseek-v4.1-flash-ist-das-2.00bpw.log` in `C:\StrataRPCD`)
- C's `stratarpcd-worker-*.log`
- the numbers above

## When something goes wrong

| Message | Meaning |
|---|---|
| `wrong STRATA_RPC_TOKEN` | The secret differs between the two windows. |
| `--layer-stage-split differs`, `--ctx differs`, `--layer-stage-wire differs` | The two PCs were started with different settings. |
| `the two PCs load different models` | The packs differ, or `--experts` differs. Run setup with the same model on both. |
| `waiting for the worker at ...` | C is still loading, or the address, port or firewall is wrong. |
| `a layer stage runs on one GPU per PC` | That PC sees more than one GPU. Pass `--gpu N` through the config. |
| An answer that stops early, with `sampling on the worker failed` in A's log | The link to C dropped. The next new chat connects again. |

## Not in this version

- DSpark (it would be off on these PCs anyway), pictures, and saved conversation slots.
- The laptop as a third PC (next phase).
- Overlapping the network with the GPUs during a prompt.
