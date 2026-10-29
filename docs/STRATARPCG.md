# StrataRPCG: GLM-5.3-Flash over two PCs

StrataRPCG runs **GLM-5.3-Flash** (pfeifferj's GSQ-RCO 3.5-bit GGUF, 137.1 GB) on two PCs with Project Maya's GLM
engine, split by layers:

| PC | Address | Role | Runs |
|---|---|---|---|
| **4090 PC** | 192.168.50.206 | coordinator (the server) | the web page and API, the embedding, pictures, layers 0-23 |
| **5070 Ti PC** | 192.168.50.143 | worker | layers 24-44, the output head and sampling |

For each token, the 4090 runs its layers and sends one 32 KB row (BF16) to the 5070 Ti. The 5070 Ti runs the rest and
sends back the next token. With `-HeadOnCoordinator`, the output head and the sampling move to the 4090 instead. The
5070 Ti then sends back the final row (64 KB as FP32), about half a millisecond more per token on gigabit Ethernet.
Prompts are sent in chunks. The 4090 computes the next chunk while the previous one is on its way.

The branch is `stratarpcg`. StrataRPC (Qwen) and StrataRPCD (DeepSeek) are unchanged and stay in their own folders.

> **Status:** the engine, protocol and scripts are built and tested without a GPU (a CUDA 13.2 compile for the 4090
> and the 5070 Ti, protocol tests over loopback, launcher tests). **The split itself has not yet run on real GPUs.**
> The first start on your PCs is its first real test. Please send both PCs' logs if anything goes wrong.

## Once, on both PCs

1. **Get the branch into `C:\StrataRPCG`:**

   ```powershell
   git clone -b stratarpcg https://github.com/Protonburn35/StrataRPC C:\StrataRPCG
   ```

   - **Updating later:** see [Updating](#updating) below (no Git needed).

2. **Run `START-HERE.bat`** in `C:\StrataRPCG` (double-click, or from a terminal). It runs Project Maya's setup for
   this model:
   - **Checks the PC.** Setup compiles the engine on each PC, so it needs Visual Studio 2022 Build Tools ("Desktop
     development with C++") and the CUDA toolkit (13.x for the 5070 Ti). If one is missing, setup says so and prints
     the command that installs it.
   - **Compiles the engine** for that PC's GPU (10-30 minutes the first time).
   - **Asks before it downloads** the model (137.1 GB) and the picture files (1.1 GB, from Maya's repository). They go
     to `C:\Strata-data\models-glm`, next to your other Strata data.
   - **Builds the model's pack** and writes its config, `C:\StrataRPCG\maya-gsq-rco-3.5bit.json`. It does not start
     the model.
   - **Defaults:** a 512K context (FP16 cache) and no drafter. Anything you add wins over the defaults, e.g.
     `START-HERE.bat --context 262144`.
   - **Memory warning:** if setup warns that one PC alone has less memory than the model, carry on. In the split,
     each PC loads only its own layers.
   - **Downloading once:** after the first PC finishes, copy its `C:\Strata-data\models-glm` folder to the same place on
     the other PC *before* running `START-HERE.bat` there. Setup checks the files it finds (SHA-256, a few minutes) and
     downloads only what is missing.

3. **The 5070 Ti PC's firewall:** nothing to do. StrataRPCD's rule `StrataRPCD worker` (inbound TCP 39501, private
   network) already lets the 4090 in, and StrataRPCG uses the same port. Only one worker can hold the port, so stop
   StrataRPCD's worker before starting this one. To check the rule, run this in PowerShell:

   ```powershell
   Get-NetFirewallRule -Direction Inbound | Where-Object DisplayName -like "*Strata*" | ForEach-Object { "{0}: port {1}" -f $_.DisplayName, ($_ | Get-NetFirewallPortFilter).LocalPort }
   ```

## Updating

Do this on **both PCs** for each new version. The folder is deleted and made again from the download, so every
version starts clean.

1. **Stop StrataRPCG** on that PC (Ctrl+C in its windows).
2. **Back up what you want to keep from `C:\StrataRPCG`.** The model, its pack and their checksums live in
   `C:\Strata-data\models-glm` and are kept. Everything in `C:\StrataRPCG` is deleted:
   - your saved chats (`sessions-host` on the 4090, `sessions-worker` on the 5070 Ti);
   - the logs;
   - any change you made to a config (`maya-*.json`), such as an API key. Setup writes the configs again.
3. **Delete the folder, download the version and put it in place.** In PowerShell, `$c` picks the version:
   `stratarpcg` is the newest, or put a commit id from `HANDOFF.md` for that exact one.

   ```powershell
   Remove-Item C:\StrataRPCG -Recurse -Force
   $ProgressPreference = "SilentlyContinue"
   [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
   $c = "stratarpcg"
   $zip = "$env:TEMP\stratarpcg-$c.zip"
   $dir = "$env:TEMP\stratarpcg-$c"
   Invoke-WebRequest "https://github.com/Protonburn35/StrataRPC/archive/$c.zip" -OutFile $zip
   Unblock-File $zip
   Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
   Expand-Archive $zip -DestinationPath $dir
   $src = (Get-ChildItem $dir -Directory | Select-Object -First 1).FullName
   robocopy $src C:\StrataRPCG /E /NFL /NDL /NJH /NP
   ```

4. **Set up and compile.** In a fresh folder, setup:
   - makes the Python environment again;
   - downloads llama.cpp's source;
   - compiles the engine and the picture encoder from scratch.

   It finds the model and its pack in `C:\Strata-data\models-glm`, so nothing large is downloaded. Expect longer
   than an in-place update (roughly 30-60 minutes).

   ```powershell
   cd C:\StrataRPCG
   .\START-HERE.bat --yes --download-model --no-start
   ```

   It should end with `engine compiled: C:\StrataRPCG\build\maya.exe`. `build\MAYA-BUILD.json`'s `"src"` is the
   version's fingerprint, which `HANDOFF.md` lists for each version.
5. **Start as usual** (the 5070 Ti first; see the next section).

- **Keeping the folder instead:** leave out the `Remove-Item C:\StrataRPCG` line. The copy then overwrites only the
  branch's files and keeps your configs, environment, build and saved chats. Step 4 then only recompiles what changed.

## Each time

Both windows need the **same secret** in `STRATA_RPC_TOKEN`. Your existing StrataRPCD secret is fine. Never share it
or commit it anywhere.

**On the 5070 Ti PC first.** Loading takes several minutes. Wait for `READY layer-stage worker`:

```powershell
cd C:\StrataRPCG
$env:STRATA_RPC_TOKEN = "<your long secret>"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-worker.ps1
```

**Then on the 4090 PC:**

```powershell
cd C:\StrataRPCG
$env:STRATA_RPC_TOKEN = "<your long secret>"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-coordinator.ps1
```

The page opens at **http://127.0.0.1:8081**. The OpenAI-style API is at `http://127.0.0.1:8081/v1`.

- **Starting the 4090 first is fine.** The 4090 loads its layers, then waits up to 30 minutes for the worker, printing
  `waiting for the worker at 192.168.50.143:39501` every minute.
- **A wrong secret or different settings stop it at once**, with the reason (see the table below).
- **To stop:** Ctrl+C in each window.
- **Logs:** the worker writes `C:\StrataRPCG\stratarpcg-worker-<date>.log`; the 4090 writes
  `C:\StrataRPCG\maya-gsq-rco-3.5bit-stratarpcg.log`.
- **To see what a script would run without starting anything:** add `-DryRun`.

## Options

Add these to the commands above. **Both** means use the same value on both PCs; the 4090 refuses a worker whose
settings differ.

| Option | Where | What it does |
|---|---|---|
| `-HeadOnCoordinator` | both | Runs the output head and sampling on the 4090 instead of the 5070 Ti. Frees VRAM on the 5070 Ti for experts, uses VRAM on the 4090. Costs 64 KB more per token on the network. |
| `-Split 24` | both | The worker's first layer (default 24: the 4090 runs 0-23, the 5070 Ti runs 24-44). A lower number moves layers to the 5070 Ti. |
| `-Sessions` | both | Turns on saved chats (below). |
| `-NoVision` | 4090 | Text only. Pictures are on by default. |
| `-Context 262144` | both | Overrides setup's context for this start. |
| `-Wire fp32` | both | Sends the rows exactly instead of as BF16 (64 KB a token instead of 32 KB). For checking only. |
| `-Stats` | either | Writes each prompt's speed and every 256 tokens' timings to the log. |
| `-Worker IP:port` | 4090 | The worker's address (default `192.168.50.143:39501`). |
| `-Listen IP:port` | 5070 Ti | Where the worker listens (default `0.0.0.0:39501`). |
| `-Port 8081`, `-ApiHost 127.0.0.1` | 4090 | The page and API's port and address. `-ApiHost 0.0.0.0` lets other devices on your network use it. |
| `-Gpu N` | either | Which GPU, as `nvidia-smi` numbers them (default 0). |
| `-Config path` | either | Setup's config, if the folder holds more than one GLM config. |

Example, the head on the 4090 with saved chats:

```powershell
# 5070 Ti PC
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-worker.ps1 -HeadOnCoordinator -Sessions
# 4090 PC
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stratarpcg\start-coordinator.ps1 -HeadOnCoordinator -Sessions
```

## Pictures

The 4090 reads pictures with Maya's picture encoder (the files setup downloaded) and puts them into the prompt. The
5070 Ti needs nothing for them: it only ever sees the rows coming out of the 4090's layers. Attach pictures on the
page, or send OpenAI-style `image_url` content to the API. `-NoVision` turns this off.

## Saved chats

This uses the same API as StrataRPC: save the conversation the model holds, and restore it later without re-reading
the whole prompt. Start **both** PCs with `-Sessions`.

Save, after a reply:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8081/slots/0?action=save" -ContentType "application/json" -Body '{"filename":"chat1.bin"}'
```

Restore, then send the same conversation (plus your new message) as usual:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8081/slots/0?action=restore" -ContentType "application/json" -Body '{"filename":"chat1.bin"}'
```

- **Where the files go:** each save writes two files. The 5070 Ti writes its half first, to
  `C:\StrataRPCG\sessions-worker\<32-character id>.gslot`. The 4090 then writes `C:\StrataRPCG\sessions-host\chat1.bin`,
  which records that id. A restore needs both. Keep them together and delete them together.
- **Restore checks first.** It refuses a file saved with another model, split, head setting, wire or context, or one
  whose other half is missing or damaged. If a restore fails partway, both PCs start that chat over cleanly.
- **The client still sends the whole conversation.** The restore only means the model does not have to read it again.
  A conversation with pictures must send the same pictures.
- **Disk space:** files grow with the conversation's length. A save needs 4 GB free on each PC; otherwise it is
  refused with HTTP 507.
- **Names:** a plain file name only (no folders, no `..`).

## When something goes wrong

| Message | What it means |
|---|---|
| `set STRATA_RPC_TOKEN ...` | That window has no secret. Set it and start again. |
| `layer-stage handshake ... failed: wrong STRATA_RPC_TOKEN` | The two windows have different secrets. |
| `--layer-stage-split differs`, `the context differs`, `--layer-stage-wire differs`, `--layer-stage-head differs` | The two PCs were started with different options. |
| `the two PCs load different models` | The packs differ. Run `START-HERE.bat` with the same model on both PCs. |
| `waiting for the worker at ...` for a long time | The worker is still loading, or its address, port or firewall is wrong. |
| `the layer-stage worker cannot listen on ...` | Another program (e.g. StrataRPCD's worker) holds port 39501. |
| `a layer stage runs one GPU on each PC` | The config has a multi-GPU split. Use `-Gpu N`. |
| `disk sessions are off: start both PCs with --layer-stage-sessions` | Saved chats need `-Sessions` on both PCs. |
| A reply stops early, with a `layer-stage` error in the 4090's log | The link to the 5070 Ti dropped. The next request reconnects and reads its prompt again. |

## Not in this version

- **One GPU per PC.** No multi-GPU split inside a PC.
- **One conversation at a time.** No `--parallel`.
- **No drafting.** This file has no MTP block, and DFlash2 is not used in the split.
- **No automatic saved conversations.** Maya keeps several recent chats ready on one PC; that is off in the split.
  The current chat is still reused turn to turn. Switching chats re-reads the prompt unless you restore a saved chat.
- **No overlap on the worker during a prompt.** The 5070 Ti does not receive the next chunk while computing the
  current one. The 4090 does overlap; this half is a possible later speedup.
