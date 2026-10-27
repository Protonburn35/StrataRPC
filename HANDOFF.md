# StrataRPCD handoff (branch `stratarpcd`)

Read this first. Claude and Sol (GPT/Codex) both work here. Fetch before you start, keep the other's commits, update this
file with every push, and never force-push.

## Goal

Run DeepSeek-V4.1-Flash (IST-DAS 2.00 bpw) across PCs with layer-stage RPC, the way StrataRPC `main` runs
Qwen3.8-Flash-Next ("qfn") on two PCs. Phase 1 is PC A + PC C. Phase 2 adds the laptop (B) as a third stage, after the
user has tested phase 1.

| PC | Hardware | Role |
|---|---|---|
| A | RTX 4090 24 GB, 7800X3D, 32 GB RAM | coordinator: API, embedding table, Engram, layers 0-19 |
| C (192.168.50.143) | RTX 5070 Ti 16 GB, 9800X3D, 32 GB RAM | worker: layers 20-39, head, sampling |
| B (192.168.50.129) | RTX 4060 Laptop 8 GB, HX 370, 32 GB RAM | phase 2 |

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
| Engine stage mode in `ds4_model.cu` (load only own layers, import/export rows) | **next** |
| `ds4_main.cpp` flags and the worker main loop | todo |
| Windows CUDA workflow building `strata-dsv41.exe` + package | todo |
| User guide (StrataRPCD folder, model download, pack, commands, test plan) | todo |

Run the GPU-free tests (Linux or Windows, no CUDA needed):

```sh
cmake -S . -B build-stage -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_STAGE_TESTS=ON
cmake --build build-stage --target ds4_stage_test rpc_transport_test
ctest --test-dir build-stage -R "ds4_stage_test|rpc_transport_test" --output-on-failure
```

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
