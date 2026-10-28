# StrataRPCD handoff (branch `stratarpcd`)

Read this first. Claude and Sol (GPT/Codex) both work here. Fetch before you start, keep the other's commits, update this
file with every push, and never force-push.

## Start here (2026-10-28): the user's setup and Sol's next two tasks

**The setup the user runs** (both PCs on the user's own local build of this branch, `scripts/stratarpcd/build-local.ps1`
from `C:\StrataRPCD-project`, copied into `C:\StrataRPCD\engine-rpcd`):

- **IST-DAS 2.25 bpw only** (2.00 deleted; Qwen3.8-Flash-Next on StrataRPC `main` is the user's fast model), **1M
  context**, **BF16 rows** (the start scripts' default), default prompt chunks, `-Stats`. C's firewall rule admits A
  (192.168.50.206) only.
- **C (worker, 192.168.50.143, over SSH in PowerShell)**, then **A (coordinator)**, the same `STRATA_RPC_TOKEN` in both:

  ```powershell
  # C
  powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1 -Config C:\StrataRPCD\strata-deepseek-v4.1-flash-ist-das-2.25bpw.json -Context 1048576 -Listen 192.168.50.143:39501 -Stats
  # A
  powershell -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-coordinator.ps1 -Config C:\StrataRPCD\strata-deepseek-v4.1-flash-ist-das-2.25bpw.json -Context 1048576 -Worker 192.168.50.143:39501 -Port 8081 -Stats
  ```

- **Measured** (60,283-token prompt): 123.4 s (488 tok/s) to read it, **6.8 tok/s** answering. Answer speed is what
  the user cares about; it is at this hardware's ceiling without losing quality (see Status). The user's DeepSeek goal:
  a Nintendo Switch static-recompilation PC port.
- **The user's rules still hold:** nothing that lowers quality, every experiment opt-in, both PCs updated together.

### Task 1: vision in the split

- **Now:** `stage_cut` (ds4_model.cu) drops `vision.*` / `aligner.*` on both stages and sets `o.vision = false`, though
  setup built the tower into the pack (`mmproj-BF16.gguf`) and the config passes `--vision on`.
- **Where it runs:** on A, which holds the embedding table. Pictures come in through the serve loop's `GENI` command
  (ds4_main.cpp: the `DSI1` pictures file); the tower's rows replace the token embeddings at the image positions
  (`img_tok`, `image_row(p)`, `in_image(p)` in ds4_model.cu). Keep `vision.*` out of the identity hash, as now.
- **The catch:** the MoE layers route image positions with their own gate bias (`ffn.gate.bias_vl`; see the `img_tok`
  use near the layer setup, ~line 3995). So the worker's layers 20-39 must know which rows are image rows, though it
  never sees the pictures. Send a per-row image flag (or the image spans) with WINDOW and PROMPT rows, and have the
  worker's `in_image()` answer from it. That is a protocol change: bump `kProtocol` (ds4_stage.cpp), add it to the
  GPU-free test, and update both PCs together.
- **Cost:** the tower is ~319 MiB on A (about 33 fewer experts there).
- **Reference:** StrataRPC `main` did vision for the Qwen layer stage in Exp28 (docs/LAYER_STAGE_RPC.md there).
- **Test:** the same picture prompt on one PC (setup's single-PC config) and through the split; the answers should
  agree.

### Task 2: port StrataRPC's optional paired disk sessions ("save chats")

- **The feature (StrataRPC `main`, commit `9bd85e5b` "Add opt-in paired disk sessions for layer-stage RPC", Exp29,
  docs/LAYER_STAGE_RPC.md ~lines 24-55):**
  - **Manual:** `POST /slots/0?action=save|restore` with `{"filename": "chat-1.bin"}` on the host API.
  - **Each PC saves only its own layers' state to its own SSD.** The worker writes `<snapshot-id>.srpc` first; then
    the host publishes its file, which holds the snapshot id. Only the small descriptor crosses Ethernet.
  - **Opt-in:** `--layer-stage-sessions` on both, `--layer-stage-session-dir` on the worker, `slot_save_path` in the
    host's JSON.
- **What this branch already has:** serve/server.py implements `/slots/0?action=save|restore` by writing
  `SAVE <path>` / `RESTORE <path>` to the engine's stdin and waiting for `SAVED` / `RESTORED` / `SERR ...` (around
  server.py lines 2434-2460 and 4194-4250; `slot_save_path`).
- **What is missing:**
  - **strata-dsv41's serve loop (ds4_main.cpp) has no `SAVE` / `RESTORE`.** It only keeps conversations
    automatically (STRATA_DS4_SLOT_KEEP, `ds4-slots`, built on `Model::slot_save` / `slot_load`).
  - **In a stage, `Model::slot_save` / `slot_load` refuse,** and the worker forces `STRATA_DS4_SLOT_KEEP=0`.
- **Plan:**
  1. **`SAVE` / `RESTORE` in ds4_main.cpp's serve loop** on top of `slot_save` / `slot_load`. This works on one PC
     too.
  2. **In a stage:** new DSTG ops (e.g. SNAP_SAVE id / SNAP_LOAD id). The worker saves or loads its half (its layers'
     window rings, compressed KV, index keys, the compressors' partial groups, the position) in its session dir. It
     goes first; then the coordinator writes its file with the snapshot id, the spec / identity hash, the split, the
     context and the position. On restore, both sides check all of them.
  3. **Lift the stage refusal** in `slot_save` / `slot_load` for a stage's own layers (A: no head; C: no embedding).
  4. **Keep it opt-in, off by default,** as in Exp29.
  5. **Tests:** the new ops in ds4_stage_test, and a save / restore round trip.

## Goal

Run DeepSeek-V4.1-Flash (IST-DAS 2.00 bpw) across PCs with layer-stage RPC, the way StrataRPC `main` runs
Qwen3.8-Flash-Next ("qfn") on two PCs. Phase 1 is PC A + PC C. Phase 2 adds the laptop (B) as a third stage, after the
user has tested phase 1.

| PC | Hardware | Role |
|---|---|---|
| A (192.168.50.206) | RTX 4090 24 GB, 7800X3D, 32 GB RAM | coordinator: API, embedding table, Engram, layers 0-19 |
| C (192.168.50.143) | RTX 5070 Ti 16 GB, 9800X3D, 32 GB RAM | worker: layers 20-39, head, sampling |
| B (192.168.50.129) | RTX 4060 Laptop 8 GB, HX 370, 32 GB RAM | phase 2 |

Network: the worker listens on port 39501 (StrataRPC's qfn worker uses 39001). C's inbound firewall rule lists
remote addresses: `-RemoteAddress 192.168.50.206` (A only; the user removed the laptop on 2026-10-28). When planning phase 2, check which PC connects to
which: every listening PC needs an inbound rule, and a rule with an address list must include each PC that connects to it.

The user keeps the whole model in a `StrataRPCD` folder on every PC: the GGUF shards, the Engram table, the DSpark GGUF
and the pack. This is separate from the qfn setup on `main`.

## The user's rules

- Nothing may lower the model's quality. Use `--experts 6` (full quality, not "Faster" mode), keep the FP32 wire as
  the default, and no temperature-0 or other unusual settings.
- Every experiment is an option, off by default.
- 64 GB of RAM is not an option.
- No PR unless asked. No model identifiers in commits.
- End commit messages with `Assisted-by: <agent>`, plus the co-author and session lines your harness asks for.

## Why this branch starts from mainline, not from StrataRPC `main`

StrataRPC `main` forked mainline (niko1221/strata) at a0116b2 (2026-10-12). Mainline has 2,671 commits since, among
them the DeepSeek engine (`src/dsv41`, binary `strata-dsv41`). 22 files changed on both sides, including
`generate.cpp`, `verify.cpp` and `prefill.cpp`. So `stratarpcd` starts from mainline 5e8a4cc (0.2.7) and copies over
only StrataRPC's self-contained link pieces:

- `src/core/rpc_transport.cpp`, `include/strata/core/rpc_transport.hpp`, `tests/core/rpc_transport_test.cpp`:
  copied unchanged.
- `include/strata/core/layer_stage_codec.hpp`, plus `src/core/layer_stage_codec.cpp`: the FP32/BF16/FP16 codec,
  split out of StrataRPC's `layer_stage_rpc.cpp`.

The Qwen layer-stage code (`layer_stage_rpc.cpp`, the `generate.cpp` hooks, Exp26-29) is not on this branch.

## Design (decided; see `src/dsv41/ds4_stage.hpp`)

- **Split at layer 20 (`--layer-stage-split 20`).** In V4.1 the compressed KV sources are layers 2, 8, 14 and 20.
  - Each layer reads the most recent source before it, which must have its compress ratio.
  - Layers 20-39 have ratio 1 and read only layer 20's KV and index keys (mainline's "yoco" layer).
  - The candidate source is layer 20, and the index sources are 2, 8, 14, 20, 24, 28, 32 and 36.
  - Engram runs at layers 1 and 14.
  - So at a split of 20, nothing but the per-token hand-off crosses: no KV cache, no Engram, no candidates.
  - Splits 2, 8 and 14 are also clean for KV, but they put Engram layer 14 on the worker; v1 refuses that.
- **What crosses per token is the existing inter-GPU hand-off.** It is `h | pre | topk | cand` (`hand_xfer`,
  `chunk_xfer`), minus topk and cand, because layer 20 recomputes both.
  - `h` is hc x dim = 4 x 5120 floats. On the wire it uses the negotiated format: FP32 by default (exact), or
    BF16/FP16 (lossy, opt-in via `--layer-stage-wire`).
  - `pre` is 8 floats, always FP32.
  - That is 80 KiB per token in FP32, or 40 KiB in BF16.
- **The head lives on the worker (C), which samples.** Each stage is then a contiguous slice: A runs the embedding plus
  layers 0-19, C runs layers 20-39 plus the head. This mirrors Exp29's worker-owned head. Head on A was discussed as a
  later option (it matters only with DSpark or the laptop).
- **Off in stage mode:** DSpark (it would be auto-off on these PCs anyway), vision, and conversation slots
  (`slot_save` / `slot_load`).
- **Protocol (`ds4_stage.cpp`):** TCP, authenticated with `STRATA_RPC_TOKEN`, a handshake that checks the model
  identity, split, context and wire format, then one request at a time.
  - `WINDOW`: a step or a short feed, rows inline, one round trip.
  - `PROMPT` + `ROWS`: long prompts. The coordinator streams rows in pieces of 256 as its last layer finishes each
    chunk. The worker acknowledges each piece when its first layer takes it, so the two GPUs overlap chunks.
  - `RESET`, `CHECKPOINT`, `RESTORE`, `PIN`: mirror `Model`'s state calls 1:1.
  - `SAMPLE`, `LOGITS`: answered by the worker.
  - Any refusal closes the connection. The coordinator reconnects on its next reset, and both ends restart at
    position 0.
  - A prompt the coordinator stops (Stop button) sends a HALT frame. That connection stays open, and both ends go
    back to a checkpoint.
- **Coordinator integration:** inside `Model` (`ds4_model.cu`), so `ds4_main.cpp`'s serve loop and the Python server
  stay unchanged. `Model::feed` / `step` / `feed_mark` run the local layers and export rows; `Model::sample` / `logits`
  / `checkpoint` / `restore` / `pin` / `reset` forward to the worker.

## Status

| Piece | State |
|---|---|
| Transport + codec copied, CMake `strata_stage_net` library | done |
| Stage protocol `src/dsv41/ds4_stage.{hpp,cpp}` | done |
| GPU-free test `tests/dsv41/ds4_stage_test.cpp` | passes, also under ASan/UBSan |
| Engine stage mode in `ds4_model.cu` | built with nvcc 13.2 + MSVC 14.44 for sm_89 + sm_120 (run 7); not run on a GPU yet |
| `ds4_main.cpp` flags and worker mode | built on Windows (run 7); not run yet |
| Windows workflow `.github/workflows/stratarpcd-windows.yml` | **green: run 7** (https://github.com/Protonburn35/StrataRPC/actions/runs/40593070051), artifact `StrataRPCD-engine-win-x64-cuda13-sm89-sm120` (https://github.com/Protonburn35/StrataRPC/actions/runs/40593070051/artifacts/12776846981, kept 30 days), `strata-dsv41 0.1.52 build a31d1b2cb` |
| User guide `docs/STRATARPCD.md`, `scripts/stratarpcd/start-worker.ps1`, `start-coordinator.ps1` | written; the workflow parses them and dry-runs them under PowerShell 5.1 |

What the engine changes do (see `git log -p src/dsv41/ds4_model.cu`):
- **`stage_cut()`, called in `load()` after the geometry checks:** builds `ident` and cuts `Geo` and `pk.t` to the
  stage's layers.
  - `lb` is the global index of local layer 0; layer names use `il + lb`.
  - DSpark and vision are off; experts per word default to the model's (full quality).
- **Coordinator (`stage_out`):** no head (`head_b` 0, no `head`/`norm.weight`/`d.logits`). Its last layer's rows go
  through `stage_export()` in `forward_serial`, `run_dev`, `run_dev_chunks` and `run_dev_layers`.
- **Worker (`stage_in`):** no embedding table and no Engram. Its first layer's rows come through `stage_import()` from
  `stage_rows` at the same four places. `yoco_layer()` accepts `yl == 0` on a worker, so layers 21-39 keep the
  long-prompt shortcut.
- **Rows** move in pieces of 256 through pinned `sx_h` / `sx_pre`.
- **One GPU per stage is required** (`--gpu N` picks it).
- **Public wrappers at the end of `ds4_model.cu`:**
  - `feed` / `feed_mark` / `step` go through `Impl::stage_feed` on a coordinator: WINDOW for n <= 16 without a
    checkpoint mark, otherwise PROMPT + ROWS.
  - `sample` / `logits` are answered by the worker.
  - `reset` / `checkpoint` / `restore` / `pin` act on both ends; `reset` reconnects after a failure.
  - `slot_save` / `slot_load` are refused.
  - `Model::load` on a coordinator waits up to `STRATA_LAYER_STAGE_WAIT` seconds (1800) for the worker.
- **`ds4_main.cpp`:** `--layer-stage-client/-server/-split/-wire`. `STRATA_RPC_TOKEN` is required. The worker mode
  is `stage_worker()` with `StageBackend` around `Model`.

How to check without a GPU (Linux):
- **Protocol tests:** the cmake commands above.
- **Engine compile check:** `clang++ -x cuda -std=c++17 -fsyntax-only --cuda-path=<a CUDA root> --cuda-gpu-arch=sm_89
  -DSTRATA_VERSION='"t"' -Iinclude -Isrc/dsv41 -Isrc/prefill -Ithird_party/ggml src/dsv41/ds4_model.cu`.
  - The CUDA root used here was assembled from PyPI wheels: `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvcc-cu12` (headers
    and libdevice only) and `nvidia-cuda-cccl-cu12`, plus `cublas_v2.h` from `nvidia-cublas-cu12` and an empty
    `curand_mtgp32_kernel.h`.
  - Without a GPU, this is as far as the engine can be checked.
- **Engine main:** `g++ -std=c++17 -fsyntax-only -DSTRATA_VERSION='"t"' -Iinclude -Isrc/dsv41 src/dsv41/ds4_main.cpp`.

The Windows build (`stratarpcd-windows.yml`):
- **Trigger:** runs on pushes to this branch that touch the engine, unless the commit title has `[skip ci]`.
- **Steps:** it dry-runs the scripts, runs the protocol tests on Windows, builds `strata-dsv41.exe` with CUDA 13.2 for
  sm_89 + sm_120, and uploads the artifact `StrataRPCD-engine-win-x64-cuda13-sm89-sm120`. The artifact holds the exe, the
  cuBLAS DLLs, the scripts, `README-STRATARPCD.md` and this file.

Windows build notes:
- **mainline's CUDA sources need C++20.** Run 1 forced C++17 and stopped in `iq_kernels.cu`. Run 2 gave no dialect,
  so nvcc followed MSVC's default (C++14) and stopped at the first nested namespace (`device.hpp`, in `strata_core`).
- **Run 3 passed `-DCMAKE_CUDA_FLAGS=--std=c++20` and failed exactly like run 2.**
- **Run 4 found why.** `.github/stratarpcd/nvcc-dialect.ps1` (it runs before CMake: it compiles a small C++20 file with
  several flag sets and prints `cl.exe`'s version) showed `cl.exe` 19.29, the VS 2019 toolset (MSVC 14.29) that the VS
  2022 image also has. The conda CUDA environment's activation selects it, and `VsDevCmd` kept it. With it, nvcc prints
  "The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored." This is mainline's
  #3566 (setup.py), which is why mainline builds with VS 2022's own toolset.
- **Run 5 onward:** both Windows steps pick the newest `14.4x` toolset under VS 2022 by name and pass it to `VsDevCmd`
  (`-vcvars_ver=`). The probe still runs first: the build uses the first flag set that compiles, and when none does it
  prints `nvcc --dryrun` (the `cudafe++` line shows the dialect nvcc really passed: `--ms_c++14` was the bad sign) and
  stops a few minutes in.
- **Run 5:** with the 14.4x toolset nvcc passed `--ms_c++20`; only the probe's own test file was wrong (it called a
  host function from a kernel). Fixed in run 6.
- **Run 6 compiled and linked `strata-dsv41.exe`** (the build step took 31 minutes). Only the Package step failed:
  `CONDA_PREFIX` is not set in a pwsh step (micromamba activates the environment for cmd only). Run 7 finds the
  environment under `MAMBA_ROOT_PREFIX\envs\cuda-build` instead.
- **Run 7 is green** (31 minutes of compiling, then packaging and upload). The artifact (407 MB zipped) holds
  `strata-dsv41.exe`, `cublas64_13.dll`, `cublasLt64_13.dll`, both start scripts, `README-STRATARPCD.md`, this file
  and `BUILD.txt`. cudart is linked statically.
- **PC A setup passed (2026-10-27)** with run 7's exe: SHA-256 of the `hf download` files, pack built (with DSpark and
  the vision tower), and setup's single-PC test question answered sensibly on the RTX 4090 (sm_89, stage mode off).
  Setup moved the files from `C:\StrataRPCD\models` to `C:\Strata-data\models\deepseek-v4.1-flash-IST-DAS-2.00bpw`
  (its shared data folder; the user is fine with that) and the pack is in `C:\Strata-data\packs\`. The config is
  `C:\StrataRPCD\strata-deepseek-v4.1-flash-ist-das-2.00bpw.json` (API port 8080). PC C's setup is next (sm_120).
- **The A + C split works (2026-10-28, user report: "working way better than expected")** with the 2.00 set, FP32
  rows, `-Stats` on both. The numbers are not in yet: ask the user for A's log and C's worker log.
- **Bug found by the user's 69,701-token prompt (2026-10-28), fixed in `run_dev_chunks`:** the worker failed every
  prompt longer than one chunk with "the prompt path's expert stream is out of step (layer 0)". On the worker the long
  prompt's skip layer (the model's layer 20) is the device's first layer, and its chunks before the cut are `kv_only`
  (no experts); the stream's first prediction was always (chunk 0, first layer), so the first real expert layer,
  (cut chunk, layer 0), did not match. The first prediction now follows `flow_next`'s skip rule. Mainline (skip at
  layer 20 of 40) and the coordinator (no skip) never take the new branch. A Python model of the prediction against the
  call order (chunk-major and layer-major) reproduced the failure and passes with the fix. Until the user has the new
  build, `STRATA_DS4_PROMPT_SKIP=0` on the worker avoids it (no skip: the same answers, slower long prompts).
- **First measurements (2026-10-28, FP32 rows, `-Stats`, from the user's logs):**

  | Run | Prompt | Answer | Per answer token (waiting on C) |
  |---|---|---|---|
  | 2.00, 8K context | 347 tokens at 23.9 tok/s | 9.02 tok/s over 2,955 tokens | 93-140 ms (60-86) |
  | 2.25, 8K context | 347 tokens at 19.9 tok/s | 6.46 tok/s over 921 tokens | 150-167 ms (~100) |
  | 2.00, 1M context, worker with `STRATA_DS4_PROMPT_SKIP=0` | 69,701 tokens at 216-236 tok/s (295-323 s; 5,712 MB to C; 228-245 s waiting on C) | 5.6-5.7 tok/s at ~70K | 174-182 ms (90-95) |

  - RAM store: A 5.4 GB at 8K but **0 GB at 1M** (see the next item); C 9.7-10.1 GB. Experts in VRAM: A 2,040
    (2.00) / 1,768 (2.25), C 989-1,058 / 958.
  - The 70K prompt's time on C was mostly C's own work: without the skip it runs its 20 layers on every token. The
    transfer (5.7 GB) is about 50 s on gigabit. The user's link speed is not known yet.
- **With the long-prompt fix and the RAM store fix (the user's local build of f7bba148, 1M context, FP32 rows, logs of
  2026-10-28 08:05):** a 60,283-token prompt in 135.4 s (445 tok/s); then 7.57 tok/s over 1,277 answer tokens at ~60K
  context (109-145 ms a token, 62-81 ms of it waiting on C; was 5.6-5.7 with A's store at 0). A's RAM store 7.2 GB.
  - **The worker's split of the prompt's 135.4 s:** 4,940 MB of rows in 41.9 s (**118 MB/s: gigabit Ethernet, full**),
    71.6 s waiting for A (A's 20 layers), 21.9 s C's own work (the skip works: layer 20's KV for most rows, the full
    layers only for the tail). The three add up to the total: nothing overlaps.
  - **Correction (DS4_FLOW_STATS run, 2.00, BF16, 1M, logs 09:42): A does NOT run the prompt layer-major.** Its stream
    line: 60,283 tokens in 93.3 s; 45,085 experts streamed: 38,669 from the files (**320 GB** unbuffered, 3.43 GB/s)
    and 6,416 from RAM (49.8 GB); **the compute waited 44.0 s for reads**; no "again from the layer cache" part. That
    is A's ~5,738 off-GPU experts re-read for each of the 8 chunks. `lm_begin` refuses layer-major because it needs
    `state_b` (every chunk's state on the host: chunk_hand_bytes ~84 KB a row, ~5.5 GB for 64K rows) plus at least one
    expert of layer cache within `ram_avail() - kRamMargin` (8 GB); A has ~12 GB free after its 7.6 GB store.
    C's line: 5,959 experts streamed (44.9 GB from the files, 7.5 GB from RAM: one pass, the skip leaves it only the
    tail), the compute waited 11.5 s for reads.
  - **Forced layer-major works (2.00, BF16, 1M, `DS4_LAYER_CACHE_MB=2048` on A, logs 09:49):** A's stream line: 45,074
    experts streamed, 6,235 from the files (**52.7 GB**, was 320), 6,216 from RAM, **32,623 again from the layer cache
    (layer-major, 8 chunks, 2.1 GB of RAM)**; the compute waited 9.2 s for reads (was 44.0); A's share 77.0 s (was
    93.3). The prompt **93.3 s, 646 tok/s** (was 109.3 s, 552); answers 8.04 tok/s. The state's trips to the host and
    back at every layer cost about 19 s of the 35 s saved (perhaps pageable: a 5.5 GB pinned allocation may fail).
    The forced setting does not bound the state (84 KB a token): unsafe past ~70K tokens on this 32 GB PC.
  - **Deprioritized (user, 2026-10-28: prompt speed is fine, answer speed is what bothers them). Was proposed:
    segment long prompts so layer-major always fits the free RAM** (each segment's
    state within `ram_avail() - kRamMargin`, run layer-major, segments in order; one expert pass per segment instead
    of per chunk; on the worker each segment would get its own skip tail). Open question to the user: did A page
    during the forced run (if not, the 8 GB `kRamMargin` is too conservative there)?
  - **So the overlap is still small** (rows come per chunk, but A's time is reads), and **the real prompt lever is
    layer-major on A**: one pass over its ~47 GB instead of eight would cut A's reads from 320 GB to ~40-50 GB.
    Cheap test, no code: `DS4_LAYER_CACHE_MB=2048` on A forces `lm_begin` past the RAM check (it then allocates the
    5.5 GB state, pinned or pageable, and a 2 GB layer cache per prompt; watch for paging on a 32 GB PC). It does not
    bound the state by RAM, so it is unsafe for very long prompts (200K tokens ~17 GB of state). The proper fix if it
    pays off: layer-major over groups of chunks whose state fits the free RAM (reads once per group, not per chunk).
  - **What does cut the 42 s link:** C on 2.5 GbE (118 MB/s is a 1 Gbps link; A and the switch are 2.5 Gbps; ask the
    user what C's adapter is: `Get-NetAdapter`), ~17 s, lossless; `-Wire bf16` halves it (slightly lossy); both ~8.5 s.
- **2.25 at 1M context, same 60,283-token prompt (logs 08:20):** 146.7 s (411 tok/s): link 41.9 s (118 MB/s), waiting
  for A 83.9 s, C's own 20.9 s; then 6.55 tok/s over 1,723 tokens (130-168 ms a token, 74-93 of it waiting on C).
  Experts in VRAM: A 1,683, C 897.
- **BF16 rows (2.25, 1M, logs 08:55):** the 60,283-token prompt in 123.4 s (**488 tok/s**, was 146.7 s / 411 with
  FP32): 2,471 MB of rows in 20.9 s (118 MB/s), 82.3 s waiting for A, 20.1 s C's own. Answers 6.82 tok/s over 2,390
  tokens (FP32 6.55; within the noise). As predicted: BF16 halves the link and leaves the rest.
- **2.00 with BF16 rows (1M, logs 09:30):** the 60,283-token prompt in 109.2 s (**552 tok/s**; FP32 135.4 s / 445):
  2,471 MB of rows in 20.9 s, 69.5 s waiting for A, 18.8 s C's own; answers **7.94 tok/s** over 2,119 tokens.
- **The two settled configs, measured (split, 1M context, BF16 rows, 60,283-token prompt):**

  | Set | Prompt | Answers | VRAM experts A / C |
  |---|---|---|---|
  | 2.00 (speed) | 109.2 s, 552 tok/s | 7.94 tok/s | 1,942 / 989 |
  | 2.25 (quality) | 123.4 s, 488 tok/s | 6.82 tok/s | 1,683 / 897 |

- **`-Context 262144` instead of 1M (2.25, BF16, logs 09:05):** VRAM experts A 1,683 -> 1,746 (+63), C 897 -> 942
  (+45); the 60K prompt 122.5 s (492 tok/s, 1M: 123.4 s); answers 6.78 tok/s over 2,650 tokens (1M: 6.82). No change:
  ~60 more experts on each card is lost in the noise, so the head on A (~+50-70 on C, the same fewer on A) would
  likely gain about nothing. The user keeps 1M.
- **DSpark and the 4090 alone (2.00, 8K, one PC, logs 09:20):** with `--dspark auto` the engine did not load DSpark
  ("the routed experts (132 GB) are far more than the GPU holds (23 GB)"); answers **8.06 tok/s** over 1,533 tokens,
  1,502 of 15,360 experts in VRAM, CPU lane 35%. `--dspark on`: about half that speed (user report). DSpark is not worth
  adding to the split. **The 4090 alone answers nearly as fast as the split** (2.00, 8K: 9.02 tok/s with 3,098 GPU
  experts): the split's answer gain is ~12%, so its value is mainly prompts and room for 2.25 at 1M. Next comparison
  asked of the user: the 4090 alone on 2.25, 1M, the 60K prompt (`single-4090-2.25.json`, slots off).
- **Answer speed is the user's concern.** Measured levers gave nothing (+45-63 VRAM experts a card: no change; DSpark
  halves the speed). Last cheap, lossless test offered: a bigger RAM store via `DS4_STORE_GB` (the auto size is free
  RAM - 12 GB: A 7.5 GB, C 10 GB; C only runs the worker over SSH): `DS4_STORE_GB=15` on C, `10` on A. Uncertain:
  Windows' standby list already caches the in-place (mmap) expert pages in that RAM. Beyond that only hardware helps
  (more RAM, a bigger card in C's place); Qwen3.8-Flash-Next is the user's fast model.
- **Settled setup (2026-10-28, final):** **2.25 only**, 1M context, BF16 rows (the scripts' default), default prompt
  chunks, `-Stats`. The user deletes 2.00 and uses Qwen3.8-Flash-Next (StrataRPC `main`, `C:\Strata`) as the fast
  model. The user's goal for DeepSeek: a Nintendo Switch static-recompilation PC port (exact AArch64 semantics, long
  disassembly contexts), hence 2.25. The layer-major test (`DS4_LAYER_CACHE_MB`) is to be run on 2.25.
  Answer-speed levers left are all small (the head on A: likely ~0-2% after the 256K test; one round trip a token: <1%). Prompt-side: saved
  conversation slots in the split (each PC saving its half of the cache) would spare re-reading a long prompt after
  switching conversations; the overlap is worth a few seconds. **Biggest prompt lever found: A runs chunk-major
  (re-reads its experts per chunk), see the DS4_FLOW_STATS correction below.**
- **Saved chats, later (user, 2026-10-28):** instead of automatic slots, port the manual "save chat" the user says
  "Astra" implemented (invoked by hand). Unclear which code: possibly mainline Strata's conversation files
  (`src/core/conversation_file.cpp`), possibly a separate project; ask before starting. In the split, both PCs would
  save and load their halves of the cache together.
- **`-PrefillChunk 2048` on C (2.25, 1M, logs 08:35/08:36) against the default 4096:** C's VRAM experts 897 -> 942
  (+45, +0.46 GB; its RAM store 885 -> 915 slots); answers 6.55 -> 6.89 tok/s (+5%, within the run-to-run spread:
  256-step blocks range 120-168 ms in both); the prompt 146.7 -> 156.7 s (C's own part 20.9 -> 31.0 s). Freeing ~0.5
  GB on C is worth a few percent of answer speed at most, so moving the head to A (~0.5-0.7 GB) would gain about as
  little. The default stays 4096.
- **What the 1M context costs in VRAM (8K vs 1M, same set):** the log's `caches` (compressed KV, index keys, window
  rings) are 0.58 GB on A and 0.39 GB on C; with the buffers sized by the context the experts lose 0.81 GB on A
  (2,040 -> 1,942 for 2.00) and 0.61 GB on C (1,058 -> 989): 5-7% of each card's expert slots.
- **RAM store fix (`store_gb_now`):** after the weights are on the GPU, the engine sizes the store again, and when the
  free RAM fell by 4 GB or more it took the PC for a shared-memory one (Strix Halo) and set aside 0.11 MB of RAM per
  token of context beyond 16K. A's own host buffers (8192-token prompt chunks) took 6.4 GB, so at 1M it set aside about
  113 GB and the store went to 0. The rule now also needs an integrated GPU (`cudaDeviceProp::integrated`). C's drop was
  3.4 GB, just under the limit.
- **Worker `-Stats` line:** each prompt now also says the MB of rows, the time receiving them (first byte to last, so
  MB/s is the link), the time waiting for the coordinator (before a frame's first byte) and the worker's own time.
- **Local builds:** `scripts/stratarpcd/build-local.ps1` builds the engine on a PC (VS 2022 Build Tools + CUDA 13.2.1,
  docs/STRATARPCD.md "Building the engine yourself"). The user keeps the source in `C:\StrataRPCD-project` and copies
  `build-rpcd\engine-rpcd\*` into `C:\StrataRPCD\engine-rpcd`. The workflow's build step runs the same script (with
  the conda toolkit), so a green run tests it. It re-marks changed sources by content hash, because a re-extracted ZIP
  keeps old file times. **The user's first local build took 7.5 minutes** (2026-10-28, from commit f7bba148, both
  archs), and it worked. Build 10 (the first workflow run through the script) failed after 24 minutes: micromamba's
  `init-shell: cmd.exe` sets cmd's AutoRun to the conda activation, which selects VS 2019's toolset, and the build's own
  cmd.exe children ran it before each compile. The workflow now removes that AutoRun before the build; the script warns
  when a PC has one. **Build 11 (8ce0e5b0) is green through the script.**
- **C's adapter is 1 Gbps (user, 2026-10-28):** no 2.5 GbE. **The user made BF16 rows the start scripts' default**
  ("for simplicity", after noting that vLLM sends pipeline stages' hidden states in BF16); this is the user's own
  exception to "experiments off by default". `-Wire fp32` gives exact rows. The engine's own default (no
  `--layer-stage-wire`) is still fp32; the scripts always pass it. FP16 is offered as the more precise alternative,
  with the codec's refusal above 65,504 (a failed prompt, never silent). Not measured yet: the largest value at the
  split, which would say whether FP16 is safe.
- **The user chose the 2.25 set (2026-10-28)**; later the same day: keep 2.00 too, as the speed set. DSpark and vision stay off in the split.
- **The user is adding the 2.25 set** (docs/STRATARPCD.md, "Adding the 2.25 bpw set"). On A the shared files were
  downloaded with `hf download` beside the 2.00 shards, and setup only looks for them in
  `models/deepseek-v4.1-flash-shared` (`dsv41_find_shared`): hard links put them there. With two configs the start
  scripts need `-Config`.
- **Next step: the user runs it on A and C** (docs/STRATARPCD.md, "What to test, in order"). Nothing in stage mode has
  run on a GPU yet, so the first logs are the first real test of `ds4_model.cu`'s stage hooks.
- **If the dialect problem comes back:** the next thing to try is NVIDIA's installer CUDA 13.0 (as mainline's
  `packaging/build_dsv41_windows.bat` uses) instead of the conda packages; the network installers are at
  `https://developer.download.nvidia.com/compute/cuda/<13.x.y>/network_installers/cuda_<13.x.y>_windows_network.exe`.
- **When the build fails,** the step prints the first failed command in full, so the flags nvcc got are visible.
- **`strata-dsv41` now links `strata_stage_net`** (it did not before; that would have been a link error after
  compiling).

Known limits of this first version (measure before changing):
- **The worker takes a chunk's rows one 256-row piece at a time.** It waits for each piece's copy to the GPU, which
  is queued behind the previous chunk's work. So during a prompt, the network transfer does not overlap the worker's
  compute.
- **The coordinator's device thread sends each chunk before it starts the next.** The fix for both is a receiver or
  sender thread with a host buffer.
- **Each decode step is two round trips** (WINDOW, then SAMPLE); folding SAMPLE into WINDOW would make it one.

## Plan for the engine changes (`src/dsv41/ds4_model.cu`, mainline line numbers at 5e8a4cc)

1. **`Options`:** add `stage_begin` / `stage_end` (global layer range), plus `stage_client`, `stage_token` and
   `stage_wire`.
2. **Before slicing, in `load()` right after `g.from(pk)`:** compute the model identity from the full geometry and
   tensor table. Then slice `Geo`:
   - `nl = end - begin` and `ratios` to the range.
   - `kv_src` and `ix_src` shifted by `-begin`; `cand_src` too.
   - Disable DSpark (`n_mtp = 0`, `d_targets` cleared) and vision.
   - On the worker: clear `eg_layers` and set `pk.eg_layers = 0`.
   - Validate the cut: the worker's first layer must be a KV source and an index source. Refuse Engram layers
     >= split, and a candidate source before the split that worker layers use.
3. **Layer names:** `l.name = "layers." + to_string(il + begin)` (line ~2129). The Engram names use `g.eg_layers`,
   which stays global on the coordinator.
4. **Profile:** read rows `[begin, end)` of the global `expert-profile.bin` (seek to `begin * n_exp * 4`). Save by
   rewriting only those rows.
5. **Head (coordinator has none):**
   - Skip `head_b`, the head and `norm.weight` load (~3930) and `d.logits` (~2421) on the coordinator.
   - Skip the embedding table (~2533) on the worker.
6. **Window path:**
   - `forward()` skips `h_emb` on the worker.
   - `forward_serial` (~8929): import instead of `embed`/`init_pre`; export instead of `head_rows_dev`.
   - Same in `run_dev` (~9020).
7. **Chunk path:**
   - `run_dev_chunks` (~7887): at `il == 0`, import; after the last layer, export (coordinator) instead of the
     head-group block.
   - `run_dev_layers` (~7780, layer-major): the same at `li == 0` and `last_l`.
   - `yoco_layer()` (~8458) must accept `yl == 0` on the worker, so it keeps the long-prompt shortcut (layers 21-39
     only on the last 20 x 127 + 128 rows). Also check `forward_chunks`' `keep` / `a`.
8. **Import/export helpers:** move rows in pieces of 256 through two small pinned buffers on the stage's device
   (`cudaMemcpyAsync` on `d.st`, synchronized per piece).
9. **Public wrappers at the end of `ds4_model.cu`:**
   - Coordinator `feed` / `feed_mark`: for n <= 16, local forward then `Client::window`. Longer runs use
     `prompt_begin`, with the export callback calling `prompt_rows`, then `prompt_end` (halt if the local run was
     stopped).
   - `sample` and `logits` forward to the worker.
   - `reset` reconnects if the link is down.
   - `restore` returns false when the link is down or the positions disagree, so the serve loop resets.
   - `slot_save` returns 0.
10. **Worker:** `ds4_main.cpp --layer-stage-server ADDR:PORT --layer-stage-split 20` loads the worker stage and runs
    `stage::Server::serve_forever` with a `Backend` that wraps `Model` (its `feed` sets the import `RowFn` and calls
    `Model::feed` / `feed_mark`).

## Expectations (estimates, not measured)

- Decode at full quality: 5-6 tok/s on A + C; 6.5-8 with the laptop.
- Prompt reading: a roughly fixed 20-25 s per chunk of up to 8K tokens.
- See the conversation summary in LlamaGLM's HANDOFF.md for the reasoning.
