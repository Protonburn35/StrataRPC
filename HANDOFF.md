# StrataRPCG handoff (branch `stratarpcg`)

Read this first. Claude and Sol (GPT/Codex) both work here. Fetch before you start, keep the other's commits, update
this file with every push, and never force-push. Doc-only commits carry `[skip ci]`. Never commit
`STRATA_RPC_TOKEN`, raw logs or other private data.

## Coordinator RAM budget fixed (2026-10-29, Claude)

Sol's priority below is implemented. **Not yet measured on the PCs.** The user compiles updates locally on both PCs
and decided a GitHub-built engine package is **not needed**. The Windows workflow stays a compile check
(sm_89 + sm_120) and does not package anything; Sol's "Delivery request" section below is superseded.

- **Cause, as Sol found:** the coordinator's RAM tier was sized as one part of a split over the whole model. Shared
  over all 42 MoE layers, the coordinator's 21 layers got half the eligible budget. The other half was set aside
  for layers that live on the other PC and never use it.
- **Fix:** `Glm5Model::set_stage_layers(l0, l1)` tells the model which layers this PC loads; `l1 = 0` means
  "to the last layer", as in `load_pack`.
  - `generate.cpp` calls it on the coordinator (`0, split`) and on the worker (`split, 0`) before `load_pack`.
  - In the RAM-tier block of `src/core/glm_fast_path.cu`, `remaining_layers` then counts only this PC's MoE layers.
    The last part of this PC's range takes everything left (`l1_ >= local_end`).
  - Without a stage (`stage_l0_ = -1`) the arithmetic is exactly as before: one GPU, local multi-GPU splits and the
    parallel load are unchanged.
  - Kept unchanged: the 6 GiB headroom, Windows' physical/commit checks, staging reserves, the pinning-cap
    fallback, the global geometry and layer indexing.
- **Expected effect:**
  - **4090:** the eligible budget roughly doubles, from about 3.7-3.9 GiB to about 7.4-7.8 GiB under the same
    startup conditions, subject to pinning/commit limits.
  - **5070 Ti:** unchanged; it already took its whole budget.
  - `STRATA_GLM_RAM_GB` / `MAYA_GLM_RAM_GB` in a stage is now this PC's whole RAM-tier budget, no longer a whole-model
    figure that the coordinator halved.
- **Validation:** the CUDA engine builds (Linux, sm_89); `glm_stage_test` passes. The Windows workflow re-checks
  sm_89 + sm_120 on this push.
- **Measure next:** the same prompt, sampling, context, split, head placement and pictures as before. Compare:
  - the 4090's `glm fast: CUDA0 RAM tier ... GB pinned, ... slots` line and whether a pinning-cap line appears;
  - disk loads per token and the disk-miss service time;
  - the VRAM hit rate;
  - decode and prefill speed.
- **Install route the user chose:** download the commit's ZIP, copy it over `C:\StrataRPCG` (the user backs the folder
  up first) and run `START-HERE.bat --yes --download-model --no-start` on each PC. That recompiles the engine and
  reuses the model, pack, picture encoder and configs.

## Latest hardware results and Claude's next fix (2026-10-29, Codex)

**Read this section first; it supersedes the older "first hardware run pending" status below.** The user is
nearly out of Codex usage and asks Claude to fix the coordinator's RAM budgeting next. This update documents
the findings only; no engine/config/launcher change or new performance improvement has been implemented.
**Work alone: no sub-agents.** Preserve the other branches and the user's quality/settings requirements.

### What now works and how the user starts it

- Setup finished on both Windows PCs, including the native engine and `build-vision/bin/maya-vision.exe`.
  The 117.5 GB GSQ-RCO 3.0-bit model passed SHA-256 verification and packing. Hugging Face connection timeouts
  affected the separate vision download; retries recovered on A, and copying A's complete `vision` folder to
  C over SCP was suggested after C exhausted its retries. Both later logs show completed setup.
- Two-PC GLM inference now runs on the user's hardware. Engine source is `e940f482` (optional profiler-header
  fix), with subsequent handoff-only commits. Model packing reports 45 trunk layers and **`nextn 0`**:
  this GGUF has no MTP head, and the stage path also disables drafting. There is no MTP launcher/env switch
  that makes it work here; compatible weights and split-engine drafting support would both be required.
- Both PCs use split **24**, **524288 context**, FP16 cache, BF16 wire, `-Sessions -Stats` and the same existing
  RPC token, which must remain out of the repository. Sampling in the supplied log is **temperature 1.0,
  top_p 0.95**. Keep these settings for comparisons; do not introduce quality-reducing defaults.
- C: `scripts/stratarpcg/start-worker.ps1 -Listen 192.168.50.143:39501 -Sessions -Stats`.
  A: `scripts/stratarpcg/start-coordinator.ps1 -Worker 192.168.50.143:39501 -ApiHost 192.168.50.206 -Port 8081
  -Sessions -Stats`. Run through `powershell.exe -NoProfile -ExecutionPolicy Bypass -File` from
  `C:\StrataRPCG`, setting the existing token in each window first. Start C first. API is currently bound
  to **192.168.50.206:8081**, not the launcher's loopback default. Vision is automatic; G has `-NoVision`,
  not D's `-Vision` switch. Stop the old D processes before G uses the same ports/GPUs.
- To move the output head to A, add **`-HeadOnCoordinator` on both PCs**. A failed startup in the log correctly
  refused differing head settings; a later start returned to the worker-head layout. Worker Winsock 10054
  messages accompanied coordinator shutdowns; those messages alone do not establish a network problem.
- Vision encoding executed and returned its borrowed VRAM. The supplied excerpts do not establish answer
  correctness or paired save/restore success. Every recorded generation ended with `cancel=True`; ordinary
  end-of-turn termination has not been demonstrated. The tokenizer prints EOT/EOM membership warnings;
  the native GLM path explicitly stops on ids 154820/154827/154829, so inspect before treating those warnings
  as a broken stop condition. No full 512K-context hardware validation has been supplied.

### Measurements from the user's logs (observations, not a controlled benchmark)

The reviewed inputs are the user's coordinator log `maya-gsq-rco-3.0bit-stratarpcg.log` and the earlier worker
terminal capture. Do not commit the raw files. Values below use GiB for the engine's binary-sized pools.

| Head location | A VRAM expert pool | A RAM expert cache | Reported decode | A disk-miss service time |
| --- | --- | --- | --- | --- |
| C / 5070 Ti | 16.25 GiB, 1872 slots | 3.68 GiB, 416 slots | 5.82 tok/s | 57.70 ms/token |
| A / 4090 | 15.62 GiB, 1802 slots | 3.83 GiB, 432 slots | 5.77 tok/s | 60.97 ms/token |

- The long runs round to 5.8 tok/s either way. Prompts, output lengths and cache state differed; this does
  **not** establish a meaningful head-placement advantage. Keep the worker-head default unless an identical
  prompt comparison demonstrates otherwise. A separate short, 81-token request reported 4.83 tok/s.
- Moving the head to A grew C's VRAM expert pool from **8.78 to 9.41 GiB**, **1142 to 1223 slots**.
  C's RAM expert cache stayed **10.74 GiB / 1397 slots**. Both C starts explicitly report
  `a pinning cap: 11.25 GB more pin, the RAM tier sized to it`: CUDA refused a larger pinned allocation.
  This is a separate allocation constraint; raising a RAM setting alone may not overcome it.
- A's streamed context cache uses **3.22 GiB pinned RAM**, C's **2.68 GiB**, separately from expert caches.
  The warm-up's "20.7 GB" is the combined expert tiers, including VRAM; it is not 20.7 GB of host RAM.
- A reports **35.67-47.74 expert loads from disk per generated token**, **57.70-76.03 ms/token** servicing
  disk misses. The counter counts experts loaded, not individual OS read calls. An SSD need not reach its
  advertised sequential throughput for these misses to cost decode time. This supports testing a larger
  RAM cache; it does not mean all that time is removable or promise a particular speedup.
- A later prompt totals about **152K tokens** across the recorded prefill calls. Following it, A's RAM
  expert cache is **3.92 GiB / 443 slots**, decode **4.90 tok/s**, VRAM hit **63.17%**, disk-miss service
  **76.03 ms/token**. Shorter runs' VRAM hit rates are 64-72%; these are A's local counters, not cluster totals.
- During vision, A lends **7.96 GiB** of its expert pool and takes it back afterward. One call moves 560
  experts to colder slots and drops 335; another drops 840. More RAM could help preserve displaced experts.
  This borrowing is temporary, not a permanently reserved 7.96 GiB vision allocation.
- A's ordinary 256-step windows spend roughly **80-94 ms/token waiting on C**. That includes C's computation.
  **Do not label network bandwidth a bottleneck without measuring it.** The user explicitly rejects that
  assumption. The available timings do not establish a bandwidth limit.

### Priority: fix A's stage-local RAM expert-cache budget

**Strong finding from code inspection, supported by the small measured cache; no patched hardware result yet.**
`src/program/generate.cpp` loads A with `load_pack(..., 0, stage_split)` and C with
`load_pack(..., stage_split, 0)`. Each process has one local GPU, but the GLM geometry retains the whole model's
45 layers. In `src/core/glm_fast_path.cu`, the RAM-tier sizing block (around lines 1530-1602 at `e940f482`)
initializes `remaining_layers` to all model MoE layers and, when `l1_ < g.n_layers`, grants only
`remaining * n_moe / remaining_layers`. That sharing is for a local multi-GPU split whose other parts later
consume the same process's budget. In RPC mode the other layers live on a different PC and never consume
A's remainder. At split 24 with the normal three dense leading layers, A has 21 of 42 MoE layers: half the
eligible budget. C reaches the final layer and already receives its own full eligible budget.

- A's **3.68-3.92 GiB** cache suggests a corrected eligible budget around **7.4-7.8 GiB** under comparable
  startup conditions: approximately **another 4 GiB**, subject to allocation/commit limits. The earlier
  chat's 14 GiB -> 7 GiB example was illustrative, not a measurement. A's log has no C-style pinning-cap
  warning; it does not prove every additional GiB will successfully pin.
- Make the budget explicitly local to an RPC stage, while preserving the existing budget-sharing behavior
  for in-process multi-GPU loading. Do not truncate global geometry to the local layer count: absolute
  layer indexing, protocol identity and other engine paths still need it. Keep automatic 6 GiB headroom,
  Windows' physical/commit checks, staging reserves and the allocation fallback/probing behavior intact.
- `MAYA_GLM_RAM_GB` feeds this same shared-budget calculation: do not assume its value is already a direct
  per-stage allocation on A. Preserve/document the intended explicit-budget semantics rather than giving
  the user an unchecked override that merely masks the fraction or exhausts pinnable memory.
- Validate stage-local budgeting for coordinator/worker, single-GPU full-model and local multi-GPU paths.
  Deliver the fixed engine through GitHub as requested below, then compare the **same prompt, sampling, context, split/head and image
  state**. Record each PC's allocated RAM cache, disk misses/service time, VRAM hit rate, decode and prefill.
  The expected benefit is primarily **decode**; prefill may improve, but its larger expert working set and
  transfers remain. No speedup has been measured yet. Keep the model arithmetic and full expert routing.

### Delivery request: downloadable Windows engine, avoid another local compile

The user explicitly prefers **not to compile the next update again on either PC**. GitHub Actions can build
it: [Windows run 40930293645](https://github.com/Protonburn35/StrataRPC/actions/runs/40930293645) already compiled
and linked the engine with MSVC 2022 / CUDA 13.2 for **both sm_89 and sm_120**. The obstacle is packaging, not
GitHub's ability to compile. `.github/workflows/stratarpcg-windows.yml` currently stops after tests, build and
CLI checks and has no artifact upload. No downloadable G engine was created by that successful run.

- With the RAM fix, extend the workflow to package/upload a versioned Windows engine ZIP usable on both PCs:
  `maya.exe`, its actual runtime DLL dependencies, source commit/build information and precise install steps.
  D's packaging/build-local script is a reference, but verify G's dependencies rather than copying its list
  blindly. Test the packaged CLI with dependencies from the package so the build toolkit's PATH cannot hide
  missing DLLs; retain the existing launcher/protocol checks and both CUDA architectures.
- Prefer an update that reuses the user's model, pack, Python environment, existing vision encoder and
  per-PC settings. The RAM fix alone should not require another model download, pack conversion or vision
  compilation. Check CPU-target portability as well as the CUDA architectures.
- A currently loads `C:\StrataRPCG\build\maya.exe` through `maya.py`; C's launcher runs that configured exe
  directly. Inspect Maya's engine freshness/rebuild checks when designing installation: the downloaded
  binary must actually be selected and must not immediately trigger an unwanted local rebuild. Include
  whatever matching source/build metadata or supported prebuilt-engine selection is needed; preserve local
  builds as an option. Verify both PCs' config paths instead of assuming a ZIP overwrite is sufficient.
- Build, monitor and link the resulting artifact when implementing that change. Clearly distinguish a
  compile/package check from GPU inference validation. Local recompilation is the user's accepted fallback
  only if a usable GitHub-built package cannot be delivered. This handoff-only update does not implement
  packaging or launch another workflow.

### StrataRPCD comparison: do not apply this fix blindly there

Read-only inspection of the restored two-PC D source **`78dece87`** found **no same proportional-budget bug**.
In `src/dsv41/ds4_model.cu`, `stage_cut()` trims the tensor table and changes `g.nl` to the local layer count
before placement; `store_gb()` / `store_gb_now()` size the expert store from that PC's available RAM, leaving
about **12 GB** (decimal) by default, without reserving a fraction for the remote layers. This is source
inspection, not a fresh D hardware benchmark or a claim that its reserve is optimal.

D previously had a **different, already-fixed** cache-sizing bug: a large drop in free RAM from ordinary host
buffers was mistaken for an integrated GPU, subtracting a huge context allowance and zeroing A's store at
1M context. Its `store_gb_now()` now requires `cudaDeviceProp::integrated` before applying that UMA allowance.
The fix is present in the restored source; the D handoff records A's recovered store around 7.2 GB and prior
user results. Leave D and `main` unchanged for the G budget fix. No new D changes were made in this review.

## Windows setup fix (2026-10-29, Codex)

The user's first setup on A reached the CUDA build but stopped on `glm_fast_path.cu`:
`fatal error C1083: Cannot open include file: 'cuda_profiler_api.h'`. This is an optional profiling component,
not a missing compiler. The log confirms A already has CUDA **13.2.78**, VS 2022 Build Tools **14.44.35207**,
driver **621.01** and a working Python **3.13.5** environment. Python packages and pinned llama.cpp downloaded
successfully. Model download had not yet been reached. The missing Git warning was not the fatal error.

- `glm_fast_path.cu` now detects profiler-API and NVTX headers independently. Normal inference and host timing
  build without those extras. If the headers are installed, the existing profiler capture and markers remain.
  Explicit `MAYA_GLM_NSYS` capture is refused clearly when profiler-API support was not compiled in.
- Windows CI no longer installs the optional profiler-API/NVTX packages, so its full engine build checks the
  basic toolkit path used by the PCs. CUDA 13.2, MSVC and sm_89 + sm_120 remain the build targets.
- **Local validation:** the complete changed CUDA translation unit compiled for sm_89 + sm_120 with both optional
  headers absent; the HIP-name mapping check passes. No model arithmetic, sampling or inference setting changed.
  **Windows CI passed:** [run 40930293645](https://github.com/Protonburn35/StrataRPC/actions/runs/40930293645),
  source `e940f482`. Windows PowerShell 5.1 launchers, MSVC protocol tests, full CUDA build for sm_89 + sm_120
  and the engine's CLI check all passed without installing the optional profiling packages. The full Python
  [workflow also passed](https://github.com/Protonburn35/StrataRPC/actions/runs/40930293684).
  GPU inference remains untested here.
- **User follow-up:** replacing that one source file let A compile and link `C:\StrataRPCG\build\maya.exe`
  successfully with its NVIDIA-installed toolkit, then begin downloading the 117.5 GB model. At that point setup,
  the 5070 Ti build and GPU inference were pending; see the newer hardware results above. That log reports Git absent, so this
  build uses the separate SwiGLU prompt kernel instead of the optional fused MMQ patch.
- **Recovery:** replace only `src/core/glm_fast_path.cu` with this branch's updated file and rerun
  `START-HERE.bat --yes --download-model --no-start`. Existing build objects are reused. No compiler/Python
  reinstallation is needed for this error, and the user's model files need not be removed.

## Previous continuation before hardware testing (2026-10-29, Codex)

The user asked to resume this branch after Claude. Work alone. This section records the continuation before
the user's first GPU runs; see the newer hardware results above. This cloud host has no GPU or access to the
home PCs. Do not claim an optimization's throughput improvement before measuring it on the PCs.

- Claude's Windows validation **passed**: [run 40913650336](https://github.com/Protonburn35/StrataRPC/actions/runs/40913650336),
  source `a4744854`. The MSVC protocol tests, Windows PowerShell 5.1 launcher checks and full CUDA build for
  sm_89 + sm_120 all succeeded. That workflow is a compile check, with no downloadable engine package.
- The general Python workflow [failed](https://github.com/Protonburn35/StrataRPC/actions/runs/40913650381)
  in `FailedRestartKeepsContext`: after a restart reaches EOF before READY, the process may not have exited yet,
  and `alive()` incorrectly said it was ready to use. A failed restart now explicitly marks it ended, while
  retaining the previous context, info and request routes so the next request can retry.
- The regression now holds the failed child alive after closing stdout, eliminating the exit-status race, and
  checks a subsequent successful retry. It fails against the previous code and passes with the fix.
- Local checks pass: GLM-stage and transport tests (2), PowerShell launcher checks (20), tools suite
  (594 tests, 31 skipped), server suite (541 tests, 11 skipped) and HIP-name mapping. The old cloud test
  environment needed CI's `jsonschema>=4.18,<5` dependency installed before the server suite could pass.
  No C++ engine, model, sampling, split or GPU-memory settings changed.
- **GitHub recheck passed:** [run 40921417875](https://github.com/Protonburn35/StrataRPC/actions/runs/40921417875),
  source `f4a7041a`. Backend-name mapping, the full server suite and installer/tools suite all passed.
  The follow-up documentation commit only records this result; the original successful Windows build above
  still covers the unchanged C++ engine and start scripts.
- For the first run, keep Claude's 512K FP16-cache, split 24, BF16-wire and worker-head defaults. Use `-Stats`
  on both PCs. Check short text first, then a picture and follow-up, then paired save/restore with `-Sessions`
  on both. Compare decode using the same prompts and sampling before trying a different split/head location.
  Use `-ApiHost 192.168.50.206` on the coordinator if LAN API access is wanted; its default remains loopback.

## What this branch is (2026-10-29, Claude)

GLM-5.3-Flash split by layers over the user's two PCs, built on **Project Maya v1.1.38** (`cf6472cc`, mw00/project-maya)
rather than mainline Strata or llama.cpp: Maya's GLM engine is the fastest path the user has for this model. The branch
therefore carries Maya's history. StrataRPC (`main`, Qwen) and StrataRPCD (`stratarpcd`, DeepSeek) stay as they are.
The user's folder is `C:\StrataRPCG`. The user guide is `docs/STRATARPCG.md`.

**User decisions** (do not change without asking):
- **Model:** pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF, 3.0-bit file (117.5 GB, revision `892aabe2`, sha256
  `a60586ce…d63d7b`). Maya's vision files come from peasantsmith/GLM-5.3-Flash-Maya-GGUF `vision/`.
- **Context:** 512K, FP16 cache. 256K only if measurably faster.
- **PCs:** the 4090 PC (192.168.50.206) is coordinator and server, currently using `-ApiHost 192.168.50.206`
  on port 8081 (launcher default `127.0.0.1`). The 5070 Ti PC
  (192.168.50.143) is the worker on port 39501, reusing StrataRPCD's firewall rule.
- **Defaults:** BF16 wire rows. The head stays on the worker; the head on the 4090 is an option.
- **Standing rules:** no reduced intelligence (no temperature 0 or odd sampling in defaults). Experiments are opt-in
  and off by default. The user does not want 64 GB RAM setups suggested. No PR unless asked.
- **LlamaGLM:** the user plans to delete that branch and does not care about it. `stratarpcd`'s HANDOFF still points to
  LlamaGLM's handoff (around line 498); drop that pointer the next time that handoff is updated anyway.

## Design

- **Cut:** the coordinator loads layers `[0, split)` and keeps `token_embd` on the host. The worker loads
  `[split, 45)`. The default split is 24.
  - Both use `Glm5Model::load_pack(..., gpu, 0, split)`: one part, one GPU each.
  - Between PCs travels the mHC residual, hc × n_embd = 16384 values: 32 KB in BF16, 64 KB in FP32.
- **Head placement:** `--layer-stage-head worker|coordinator`.
  - **Worker (default):** the worker samples and returns a token id.
  - **Coordinator:** the worker returns the final residual as FP32 (64 KB). The coordinator runs `head_prep` and
    samples as usual.
  - `Glm5Model::set_head_here` overrides Maya's "the head is on the part with `l1_ == n_layers`" rule.
  - With the head remote, the coordinator's `sample_token` and `logit_bias_table` forward the request to the worker:
    sampling settings, penalty history and bias slot (`stage_sample`, `stage_bias`).
- **Protocol:** `include/strata/core/glm_stage.hpp`, `src/core/glm_stage.cpp` (magic `GSTG`, protocol 1).
  - **Hello:** the token plus a Spec (pack identity hash, n_layers, split, dim, hc, vocab, context, head placement,
    sessions, wire). A mismatch is refused with its reason.
  - **Ops:** Reset, SnapSave/Restore, CkptSave/Restore, Step, PromptBegin/Rows/End, Sample, Bias, Logits, Save/Load.
  - **Transport:** reuses StrataRPCD's `rpc_transport` and `layer_stage_codec` (copied unchanged).
  - **Refusals and reconnects:** a refusal closes the link, except a refused Save/Load. `Client::refused()` tells
    "turned away" apart from "nobody listening". The coordinator waits up to `STRATA_GLM_STAGE_WAIT_S` (default 1800 s)
    for a worker that is still loading.
- **Decode:**
  - `fast_token` calls `stage_hand_over` before `head_prep` (`src/core/glm_stage_model.cu`).
  - The worker's side is `stage_step`: it runs `fast_layers(p, true)`, then either the head's argmax / sampling or a
    D2H of R.
  - Maya's snapshots and checkpoints (`snapshot_save/restore`, `ckpt_*`) are mirrored to the worker; a restore asks the
    worker first. A link epoch change drops the coordinator's state, so the next request re-reads its prompt.
- **Prompts** (`src/core/glm_prefill.cu`):
  - `prefill()` lends its buffers, then sends `prompt_begin`. After each chunk, `prefill_run` hands that chunk's
    residual rows to a sender thread (`StageSender`, `stage_send_chunk`) while the coordinator computes the next chunk.
  - The worker carves its own prompt buffers (`stage_prompt_begin`), uploads each chunk and runs `prefill_half`.
  - The minimum chunk is 32. `STRATA_GLM_STAGE_CHUNK` caps it.
- **Vision:** image rows replace token embeddings at layer 0 (`image_row`), which is on the coordinator. The worker
  needs nothing for pictures. Maya's on-demand encoder (VLEND/VRECLAIM) runs on the 4090.
- **Saved chats** (`--layer-stage-sessions`, opt-in on both PCs): the server's `POST /slots/0?action=save|restore`
  (ported from StrataRPC, `serve/server.py` `slot_action` / `session_file`) sends `SAVE`/`RESTORE <path>` to the
  engine.
  - **Save:** the worker writes `<session dir>/<32hex>.gslot` first. The coordinator then writes the named file, which
    holds the tokens, the picture hash and the worker file's id.
  - **Format:** both files are an SRPGSES1 header followed by Maya's `slot_save` stream (`generate.cpp`, namespace
    `glm_session`).
  - **Restore:** the coordinator checks identity, split, head, wire, context and the prefix hash, restores the worker,
    then itself. A failure part-way resets both PCs.
  - **Free space:** saves need `STRATA_GLM_SESSION_MIN_FREE_MIB` free (default 4096).
- **Off in a layer stage:** `--parallel`, a multi-GPU split, DFlash2 and MTP (`STRATA_GLM_NO_MTP=1`; this file has no
  MTP block anyway), and Maya's automatic conversation slots (`STRATA_GLM_SLOTS=0`).
- **Setup:** `START-HERE.bat` runs `START-MAYA.bat --setup --model GSQ-RCO-3.0bit --no-start --context 524288
  --draft mtp --models-dir ..\Strata-data\models-glm`. Anything the user passes wins.
  - `maya.py` has the `GSQ-RCO-3.0bit` MODELS entry, including vision.
  - `tools/iq_pack.py --compat-bf16` now keeps glm5-next's small F32 tensors F32 when their values are not exactly
    BF16. Those tensors are `exp_probs_b`, `ssm_a`, `ssm_dt.bias` and the hc base/scale; the GSQ-RCO file's large F32
    tensors were checked to be exact BF16.
- **Start scripts:** `scripts/stratarpcg/start-worker.ps1` and `start-coordinator.ps1` take setup's `maya-*.json`,
  drop what a stage cannot take and add the stage flags.
  - The worker runs the engine directly and tees its output to `stratarpcg-worker-<date>.log`.
  - The coordinator writes `<config>-stratarpcg.json` (GPU, port, host, `slot_save_path`, vision on or off) and starts
    `maya.py --config` with it.

## Validation so far

- **Built:** the full CUDA engine (`maya` target) with CUDA 13.2 for sm_89 on Linux; `maya --help` lists every
  `--layer-stage-*` flag.
- **Protocol:** `glm_stage_test` (fake backends over loopback: FP32, BF16, halts, refusals, state, head on the
  coordinator, sampling, sessions) and `rpc_transport_test` pass, including under ASan and UBSan.
- **Python:** the server suite (541 tests, including `serve/test_slots.py`) passes. The tools suite passes apart from
  `test_ds4_engram_ram`, which fails in a root container (`CAP_IPC_LOCK`) and is untouched here.
- **Launchers:** `tests/stratarpcg/test_launchers.ps1` (20 checks) passes under PowerShell 7.
- **Windows CI:** `.github/workflows/stratarpcg-windows.yml` compiles `maya.exe` with MSVC 2022 + CUDA 13.2 for sm_89
  and sm_120 as setup does, runs the protocol tests with MSVC and the launcher test under Windows PowerShell 5.1.
  Nothing is packaged: each PC compiles its own engine in setup.
- **Hardware follow-up:** the newer section above records successful split inference, VRAM fit and observed
  speeds from the user's logs. Numerical parity, paired saved-chat restores and full 512K-context behavior
  have not been demonstrated by those logs.

## Next

1. **Measure the RAM-budget fix** (top section) with the same prompt on both PCs.
2. **No GitHub engine package:** the user compiles locally (download the commit's ZIP, copy it over
   `C:\StrataRPCG`, run `START-HERE.bat --yes --download-model --no-start` on each PC). Keep the Windows workflow as a
   compile check only.
3. **Tune `-Split`** from each PC's expert-pool lines (VRAM vs RAM experts). With `-HeadOnCoordinator`, the 4090 loses
   the head's VRAM to experts and the 5070 Ti gains it.
4. **Possible speedups, opt-in first:** overlap the worker's receive of chunk c+1 with its compute of chunk c (it
   receives and computes in turn now); prefetch the worker's first-layer experts during the 4090's half of decode.
