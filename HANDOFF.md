# StrataRPCG handoff (branch `stratarpcg`)

Read this first. Claude and Sol (GPT/Codex) both work here. Fetch before you start, keep the other's commits, update
this file with every push, and never force-push. Doc-only commits carry `[skip ci]`. Never commit
`STRATA_RPC_TOKEN`, raw logs or other private data.

## What this branch is (2026-10-29, Claude)

GLM-5.3-Flash split by layers over the user's two PCs, built on **Project Maya v1.1.38** (`cf6472cc`, mw00/project-maya)
rather than mainline Strata or llama.cpp: Maya's GLM engine is the fastest path the user has for this model. The branch
therefore carries Maya's history. StrataRPC (`main`, Qwen) and StrataRPCD (`stratarpcd`, DeepSeek) stay as they are.
The user's folder is `C:\StrataRPCG`. The user guide is `docs/STRATARPCG.md`.

**User decisions** (do not change without asking):
- **Model:** pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF, 3.0-bit file (117.5 GB, revision `892aabe2`, sha256
  `a60586ce…d63d7b`). Maya's vision files come from peasantsmith/GLM-5.3-Flash-Maya-GGUF `vision/`.
- **Context:** 512K, FP16 cache. 256K only if measurably faster.
- **PCs:** the 4090 PC (192.168.50.206) is coordinator and server, API at `127.0.0.1:8081`. The 5070 Ti PC
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
- **Not yet run on GPUs.** The split's numerics, VRAM fit, speed and saved chats on real hardware are untested.

## Next

1. **First run on the PCs** (docs/STRATARPCG.md). Get both logs with `-Stats`. Check that answers are sane and compare
   decode speed with the user's DeepSeek split.
2. **Tune `-Split`** from each PC's expert-pool lines (VRAM vs RAM experts). With `-HeadOnCoordinator`, the 4090 loses
   the head's VRAM to experts and the 5070 Ti gains it.
3. **Possible speedups, opt-in first:** overlap the worker's receive of chunk c+1 with its compute of chunk c (it
   receives and computes in turn now); prefetch the worker's first-layer experts during the 4090's half of decode.
