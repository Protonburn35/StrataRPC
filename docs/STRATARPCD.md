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
3. **Download the model** with the Hugging Face CLI (or let setup in step 4 download it).
   - Every PC gets the same files in the same folder.
   - `hf download` creates the folder itself.
   - The revisions are the ones setup pins. A file from another revision fails setup's SHA-256 check, and setup then
     downloads it again.

   ```powershell
   $d = "C:\StrataRPCD\models\deepseek-v4.1-flash-IST-DAS-2.00bpw"
   hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF --revision cb02422c900df489cea5833e8a92171224059b3c --include "*IQ2_XXS*" "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" --local-dir $d
   hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --revision f924b94f19c1fea02c06a65c8475b0a45e2b88b1 --include "UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf" "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" "mmproj-BF16.gguf" --local-dir $d
   Move-Item "$d\UD-Q8_K_XL\DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf" $d
   Get-ChildItem $d -Filter *.gguf | Format-Table Name, Length
   ```

   | File | Bytes |
   |---|---:|
   | `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf` | 39,075,714,624 |
   | `...-00002-of-00004.gguf` | 39,388,449,504 |
   | `...-00003-of-00004.gguf` | 39,246,891,616 |
   | `...-00004-of-00004.gguf` | 20,527,837,312 |
   | `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110,595,290,816 |
   | `DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf` (only its metadata is read) | 5,780,589 |
   | `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | 7,976,914,240 |
   | `mmproj-BF16.gguf` (pictures; the split does not use them yet) | 972,208,288 |

   - **Copying instead of downloading:** a finished folder can be copied to another PC instead of downloading it again.
4. **Prepare the model** with mainline's setup, pointed at this engine. It finds the files above, checks them, and
   downloads only what is missing. In PowerShell:

   ```powershell
   cd C:\StrataRPCD
   $env:STRATA_DSV41_EXE = "C:\StrataRPCD\engine-rpcd\strata-dsv41.exe"
   .\START-HERE.bat --family deepseek41 --model IST-DAS-2.00bpw --no-start
   ```

   - **Where the files end up:** setup first moves `C:\StrataRPCD\models` into Strata's shared data folder,
     `C:\Strata-data` (it prints `model files from C:\StrataRPCD moved to C:\Strata-data`). On the same drive this is
     instant and copies nothing. The Qwen setup's files there are not touched. The pack goes to `C:\Strata-data\packs\`.
   - **"Download: 138 GB to fetch" is not a download:** files from `hf download` lack setup's `.done` marks, so setup
     counts them first. In step 5 it finds each file complete (by its size on Hugging Face), checks its SHA-256 (about
     5 minutes for all of them) and marks it. It downloads only a file that is missing, short, or fails the check.
   - **The RAM warning:** with 32 GB of RAM setup warns that one PC alone is slow and asks `Install it anyway?`.
     Answer `y`: the split is what makes it fit.
   - **Config:** setup writes `C:\StrataRPCD\strata-deepseek-v4.1-flash-ist-das-2.00bpw.json`, which the start
     scripts read.
   - **Test:** it then has this one PC answer a 16-word test question, which checks the new engine on its own.
   - **To skip the test:** set `$env:STRATA_DSV41_NO_SMOKE = "1"`.
   - **Pictures:** setup builds the picture reader into the pack. The two-PC split leaves it off for now.
5. **On C only, allow the worker's port** through the firewall for your home network. Run this in an
   **administrator** PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "StrataRPCD worker" -Direction Inbound -Protocol TCP -LocalPort 39501 -Action Allow -Profile Private
   ```

   - **Limiting it to your PCs:** add `-RemoteAddress <A's IP>,192.168.50.129` (A and the laptop, which may connect
     to C in the three-PC version), or `-RemoteAddress LocalSubnet` for any device on the home network. A changed IP
     is then blocked (A shows `waiting for the worker at ...`); a DHCP reservation in the router keeps the IPs fixed.
     To change the list: `Get-NetFirewallRule -DisplayName "StrataRPCD worker" | Set-NetFirewallAddressFilter -RemoteAddress <IPs>`.
   - **Reusing StrataRPC's rule instead:** StrataRPC (Qwen) uses port 39001. If C already allows that port, start C with
     `-Listen 0.0.0.0:39001` and A with `-Worker 192.168.50.143:39001`. Only one worker can hold the port, so stop the Qwen
     worker first. A rule that allows a program (`strata.exe`) rather than a port does not cover `strata-dsv41.exe`.
   - **To see what C's rules allow:**

     ```powershell
     Get-NetFirewallRule -Direction Inbound | Where-Object DisplayName -like "*Strata*" | ForEach-Object { "{0}: port {1}, program {2}" -f $_.DisplayName, ($_ | Get-NetFirewallPortFilter).LocalPort, ($_ | Get-NetFirewallApplicationFilter).Program }
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

- **Starting C from A over SSH:** `ssh` into C, type `powershell` if the session opens in cmd, then run the three lines
  above in it (add `-Listen 192.168.50.143:39501` to listen on C's network address only). Keep that window open:
  closing it stops the worker. If the worker reports no CUDA device there, start it from C's own desktop instead.
- **The secret:** any long random text, the same on both PCs. One way to make one:
  `[guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")`. Setup does not need it; the start scripts do.
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

## Adding the 2.25 bpw set

The 2.25 set (155 GB of model) has better quality than 2.00: about 7.3% above the original model's perplexity instead of
15.5% (code and agent work 2-3% instead of 5-9%). It shares the Engram, the drafter and the vision file with 2.00.

1. **Only where `hf download` put the shared files beside the 2.00 shards** (PC A): setup looks for them in
   `C:\Strata-data\models\deepseek-v4.1-flash-shared` and would download them again. Hard links put them there without a
   copy (no extra disk space; 2.00 keeps working):

   ```powershell
   $from = "C:\Strata-data\models\deepseek-v4.1-flash-IST-DAS-2.00bpw"
   $to = "C:\Strata-data\models\deepseek-v4.1-flash-shared"
   New-Item -ItemType Directory -Force $to | Out-Null
   foreach ($n in "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf", "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf", "mmproj-BF16.gguf") {
       if (-not (Test-Path (Join-Path $to $n))) { New-Item -ItemType HardLink -Path (Join-Path $to $n) -Target (Join-Path $from $n) | Out-Null }
   }
   ```

   A PC whose setup downloaded the files itself already has them there.
2. **On both PCs:** run setup with `--model IST-DAS-2.25bpw` (the same three lines as for 2.00, with `--yes`). It
   downloads only the 2.25 shards and writes a second config.
3. **Starting it:** with two configs, pass `-Config C:\StrataRPCD\strata-deepseek-v4.1-flash-ist-das-2.25bpw.json` (or
   `...-2.00bpw.json`) to both start scripts. Both PCs must run the same set.

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
