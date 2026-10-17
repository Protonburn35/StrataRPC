# Persistent layer-stage RPC (experimental)

The optional --layer-stage-client / --layer-stage-server mode splits one native model across two Strata processes.
The coordinator owns layers 0..K-1; the worker owns K..last, including those layers' recurrent state and KV. The
coordinator sends residual rows at prompt-chunk boundaries and verifier-window boundaries. The activation format is
negotiated as FP32 (default), BF16, or FP16; token IDs, verifier picks, and MTP drafts remain lossless. The worker
keeps its CUDA context, layer weights, expert cache, prefill path, verifier, and session state alive for the life of the
process. The transport is the bounded native TCP protocol implemented in layer_stage_rpc.cpp; Ray is not in the
activation data path.

This is a prototype for performance and capacity experiments, not a benchmarked speedup. It is deliberately limited
to one serial request at a time, a static expert profile, a native model pack, --spec >= 2,
and the same explicit fixed --prefill N, model, runtime configuration, and --layer-stage-split K on both PCs. Both
machines need their own local model files and Strata installation. The worker command is a protocol server, not an
HTTP generation endpoint. The client remains the API endpoint. Prefix continuation is opt-in; without it,
every request starts at position zero and rereads the complete prompt. Optional worker-owned native MTP is supported
for greedy and sampled requests. Images are optional from Exp28. Temperature, top-k, top-p, min-p, and seed are applied on the terminal worker
for each request. Batching, penalties, logit bias, cvec overrides, probabilistic or coupled draft sampling,
and expert-cache RPC are not supported. The worker also requires the non-finite-logit
guard; do not set STRATA_NAN_GUARD=0 on it.

## Exp28: optional vision

Upgrade the engine on **both PCs**. Exp28 uses layer-stage protocol 5 and includes the CUDA `strata-vision.exe`
encoder. Vision stays off unless enabled. Existing text-only configs continue to work after both engines are updated.
This release adds vision only; distributed session files are not enabled here.

To enable images with the existing Windows launchers:

1. Add `"--vision"` to the `args` array in **both** the host and worker JSONs. Keep the existing FP16 KV,
   context, fixed prefill chunk, split, MTP and sampling settings.
2. On the **host**, add `"--vram-reserve-mib", "2048"` to `args` (replace an existing reserve value). This
   leaves space for the encoder's image-processing buffers; it reduces room for cached experts. The encoder
   starts before the main engine, so its loaded weights are also included in the available-memory measurement.
3. Add this top-level `vision` object to the **host** JSON. Use its own shard 1 as `model`; change `IQ3_S` to
   `IQ3_XXS` for that quant. The engine folder below is the existing folder, with the new release extracted into it.

```json
"vision": {
  "exe": "C:\\Strata\\engine-rpc-exp24\\strata-vision.exe",
  "mmproj": "C:\\Strata-data\\models\\mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": "C:\\Strata\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "gpu": true,
  "max_tokens": 1024
}
```

The host needs the matching mmproj from the ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO-GGUF repository. Reuse an existing
copy if present. The worker does not need the mmproj or an encoder process. Set `gpu` to `false` to encode images
on the host CPU instead; the engines on both PCs still need `--vision`. Restart the worker first, then the host.

The existing API accepts OpenAI image_url parts, Anthropic image blocks, and browser image attachments. The host
inserts image embeddings before layer 0. The usual stage activations cross the network, along with one lossless
absolute image-position table per request (about 3 MiB at 256K context). No image file or KV cache is sent to the
worker. Both stages use the same positions for attention, decode and MTP. Sending a text prompt after an image
restores text positions on both stages. Prefix reuse checks image grids/embedding hashes as well as tokens on the
coordinator; changed images therefore cannot reuse an old image's state merely because their token counts match.

Validation includes protocol tests for missing/invalid positions, updates during a pending verifier window,
image-prefix checkpoint continuation, and image-to-text transitions. GPU image quality and speed on the real
4090/5070 Ti pair still require hardware testing; a successful build is not that validation.

## Exp27: interrupt an active reply without reloading

Sending another message, pressing Stop, or disconnecting the HTTP response cancels the current request. The
coordinator now finishes its current remote prompt chunk or verify/commit window, reports `DONE ... cancel`, and
keeps the engine, model weights, expert cache, and RPC connection alive for the next request. It skips a new MTP
draft when Stop is already known. A completed decode window retains a confirmed prefix; an interrupted pipelined
prefill retains only its previously acknowledged checkpoints and may need some replay on the next request.

Stopping during prefill can wait for the active GPU chunk and the outstanding pipeline handoff to finish. It does
not forcibly interrupt a CUDA kernel. Transport deadlines remain enabled; genuine I/O errors are still errors.
Previously the request's Stop flag interrupted socket I/O and reached a fatal decode error path as
`RPC operation cancelled`, unloading the host model.

There are no JSON or launcher changes. Exp27 uses the same layer-stage protocol 4 as Exp26. For an existing Exp26
pair, upgrading the **4090 host engine** is sufficient for this fix; the Exp26 worker remains compatible. Updating
both engines is also supported. Regression tests interrupt prefill, verification, and commit, then resume and rewind
on the same connection/executor; the older immediate-I/O cancellation tests remain in place.

## Exp26 upgrade and sampling

Replace the release engine files on **both PCs** and restart both processes. Exp26 uses layer-stage protocol 4;
older peers cannot connect to it. Existing model files, prepared packs, RPC token, split, FP16 KV settings, and
Python/API installation can stay in place. Keep `--layer-stage-reuse` on both roles to enable the checkpoint fix.
This release also fixes a coordinator flag that previously forced even exact live continuations to replay.
Worker MTP catches up prompt/decode windows that do not request a draft, retains their last residual, and rebuilds
the final MTP cell with the actual next token on resume. This avoids holes or stale lookahead in the draft cache.

Temperature is a host/API request setting; there is no worker JSON temperature to keep in sync. For example, put
this top-level object in the **host** run JSON for defaults (request values override it):

```json
"sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0 }
```

Use `"temperature": 0` for greedy output. A nonzero request `seed` selects repeatable per-position sampling;
seed zero retains Strata's existing random-seed behavior. Top-k retains the existing sampler limit of 64 candidates.
Sampler settings are a small control message once per request; vocabulary logits stay on the worker.

After the first completed turn, look for `LAYER_STAGE_REUSE reused_tokens=...` in the host log. A positive count
means both stages agreed to reuse that many tokens. `replayed_tokens` counts matching tokens after the retained
checkpoint; `new_tokens` counts the changed/appended suffix. Reasoning stripped by a frontend can still require
replaying the answer from the stable prompt checkpoint. Checkpoints add CPU memory and synchronization overhead;
there is no measured two-GPU speed claim for this change. Automated TCP tests cover checkpoint rewind, branch
invalidation, sampling updates, and end-of-output commit bounds; GPU numerical equivalence needs a model run.

## Windows two-PC setup

Extract the Windows release ZIP into C:\Strata\engine on both PCs. Keep each PC's model shards, expert pack, profile,
and Python/API installation in the paths used by its run config. Both PCs need their own local model files. Pick one
unused TCP port and permit it through the 5070 Ti PC's Windows Firewall on the trusted LAN. Set the same secret in
PowerShell on both PCs for each terminal; use a long random value:

    $env:STRATA_RPC_TOKEN = "replace-with-a-long-random-secret"

For the attached Qwen3.8 Flash Next IQ3_XXS setup, this is a complete baseline worker config. Save it on the 5070 Ti as
C:\Strata\strata-iq3_xxs-worker.json. Replace paths only if the files live elsewhere; the profile must be the same
static profile used by the 4090. --prefill 2048 is a fixed chunk shared by both roles. The Ray worker config must not
put --gpu in its args: Strata would override Ray's CUDA_VISIBLE_DEVICES assignment. With one GPU reserved, the worker
sees its assigned card as local device zero. The example keeps optional MTP, vision, and --kv-resident settings off.
Set
--max-context to the context you intend to test and ensure the selected split fits both GPUs.

    {
      "exe": "C:\\Strata\\engine\\strata.exe",
      "cwd": "C:\\Strata",
      "args": [
        "--pack", "C:\\Strata-data\\packs\\iq3_xxs",
        "--native", "C:\\Strata\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "--ple-gguf", "C:\\Strata\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
        "--expert-profile", "C:\\Strata\\data\\expert-profile.bin",
        "--expert-cache", "auto",
        "--prefill", "2048",
        "--spec", "4",
        "--max-context", "65536",
        "--kv", "fp16",
        "--mmap-experts"
      ]
    }

Start the worker with the bundled executable directly; the normal Python API wrapper expects an HTTP-capable READY
line, while the worker intentionally exposes only the layer-RPC socket. K below must be the same first worker layer on
both PCs (24 is only an example):

    cd C:\Strata
    .\engine\strata.exe --serve --gpu 0 --pack C:\Strata-data\packs\iq3_xxs --native C:\Strata\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf C:\Strata\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile C:\Strata\data\expert-profile.bin --expert-cache auto --layer-stage-server 0.0.0.0:39001 --layer-stage-split 24 --prefill 2048 --spec 4 --max-context 65536 --kv fp16 --mmap-experts

The Ray supervisor uses the worker JSON's model args and adds --serve, --layer-stage-server, and
--layer-stage-split. For a direct launch, add those role flags to the same model args. Keep the worker terminal open
while it serves RPC requests.

On the 4090, use the following compatible API config as a starting point. It keeps the host as the API endpoint and
uses the same model files, fixed prefill chunk, speculative width, context, KV type, profile, and split as the worker.
Set the address and K to match the worker. Temperature zero is required. This baseline keeps optional MTP, reduced
precision transport, and conversation reuse off.

    {
      "exe": "C:\\Strata\\engine\\strata.exe",
      "cwd": "C:\\Strata",
      "args": [
        "--pack", "C:\\Strata-data\\packs\\iq3_xxs",
        "--native", "C:\\Strata\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "--ple-gguf", "C:\\Strata\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
        "--expert-profile", "C:\\Strata\\data\\expert-profile.bin",
        "--expert-cache", "auto",
        "--prefill", "2048",
        "--spec", "4",
        "--max-context", "65536",
        "--kv", "fp16",
        "--mmap-experts",
        "--layer-stage-client", "192.168.1.50:39001",
        "--layer-stage-split", "24"
      ],
      "gpu": 0,
      "host": "127.0.0.1",
      "port": 8081,
      "tokenizer": "C:\\Strata-data\\packs\\iq3_xxs\\tokenizer",
      "model_name": "qwen3.8-flash-next-iq3_xxs",
      "log": "C:\\Strata\\strata-iq3_xxs.log",
      "sampling": { "temperature": 0 }
    }

Start the 4090's normal API wrapper after setting STRATA_RPC_TOKEN in its environment:

    cd C:\Strata
    Start-Process -FilePath .\START-HERE.bat

The client's model-specific args must match the worker's. Set STRATA_RPC_TOKEN in the environment inherited by the
worker and API server. K is the first layer assigned to the 5070 Ti. Start with a split that leaves the 5070 Ti enough
VRAM for its suffix weights, KV/state, verifier scratch, and expert cache; then tune K and compare against a single-4090
run. Strata checks the model/config fingerprint, geometry, split, verify limit, and prefill chunk at connect time. A
mismatch refuses the connection. The RPC secret is sent over plaintext TCP, so use a trusted LAN or VPN and never
expose this port to the public internet.

Layer-stage vision requires `--vision` on both endpoints (Exp28 or newer). The API wrapper forwards per-request sampling fields to
the engine. Temperature 1 is supported, as are top-k, top-p, min-p, and a seed. Penalties, logit bias, and
cvec overrides still return an ERR. Temperature 0 remains the default when the request has no sampling settings.

The worker reserves its owned prefill scratch before sizing the expert cache and disables cache borrowing, because
this persistent worker does not use the normal per-request cache loan/refill lifecycle. That can leave fewer resident
experts on the 5070 Ti, but it keeps prompt scratch outside the expert cache. Real-model cache integrity after a
multi-chunk prefill still needs the validation described below.

## Optional transport, reuse, and MTP features

Add the same options to both roles, then restart both processes:

- `--layer-stage-wire bf16` or `--layer-stage-wire fp16` halves activation bytes on the link versus the FP32 default.
  BF16 has a wider value range; FP16 can represent smaller values more precisely but rejects overflow. Either format
  changes activation values and can change model outputs. Start with BF16 on the 1 GbE link and compare tokens against
  FP32 before relying on it.
- `--layer-stage-reuse` retains the live conversation and coordinated checkpoints on both PCs. A later request
  resumes the longest retained prefix whose tokens match. Prompt checkpoints include the stable turn boundary;
  a rolling decode checkpoint is saved about every 1,024 generated tokens. This also permits reuse when a client
  removes reasoning text or changes the end of the previous answer, although tokens after the matching checkpoint
  must be read again. `--prompt-cache` is bounded to 3–8 slots in this mode, with a minimum of three even if the
  config previously requested one. Snapshots use system RAM for recurrent state and indexer tails; existing KV
  pools remain in place. No KV cache is transferred over Ethernet. Retention and token identities are checked on
  both stages. A refused resume resets both stages and replays the prompt. Transport disconnects invalidate the
  remote session; normal request cancellation finishes the active exchange. This is one active conversation; multi-chat parking remains disabled.
- `--layer-stage-mtp --mtp C:\\Strata-data\\mtp` moves the native MTP drafter to the terminal worker stage. Both PCs
  need the same MTP directory and MTP options so the connection can verify the model files and draft settings; only
  the worker loads MTP weights onto the GPU. The worker returns draft tokens and probabilities in the accepted-prefix
  commit response, so drafting adds no extra network round trip. `--spec-min-p` can cap the draft chain. Drafts stay
  greedy, while the target verifier can sample at temperature 1 (or other supported values). Drafts are accepted
  only when they match the target sample; higher temperatures can reduce acceptance and output throughput. Do not
  enable `STRATA_SPEC_PROB`, `STRATA_SPEC_COUPLED`, or `--coupled-draft` with remote MTP.

For example, append these matching flags to both the worker and coordinator configs to try reduced wire precision and
prefix reuse:

    "--layer-stage-wire", "bf16",
    "--layer-stage-reuse"

To try worker-owned MTP, append these matching flags instead (or combine them with the transport/reuse flags):

    "--mtp", "C:\\Strata-data\\mtp",
    "--layer-stage-mtp",
    "--spec-min-p", "0.0"

The MTP directory must exist on both PCs and contain matching files; only the worker uploads its weights to the GPU.
Any custom draft-vocabulary file must also match. Use `--spec-min-p` values between 0 and 1; a higher threshold can
shorten the draft chain. These features are independent switches and can be combined, but exact-prefix reuse with MTP
should be checked with repeated greedy requests before use.

Set `STRATA_LAYER_STAGE_STATS=1` in the coordinator environment to print each RPC operation's transmitted and received
bytes, elapsed time, cumulative byte totals, and negotiated activation format. These counters help separate network
time from the worker's compute and prompt work; they do not run a benchmark.

## What crosses the link

For the reviewed Qwen geometry (n_embd=2560, hc=4), prompt boundaries carry hc*n_embd values per token:
40 KiB/token in FP32 or 20 KiB/token in BF16/FP16, so a 2,048-token chunk is 80 MiB or 40 MiB respectively. The
verifier handoff carries (hc*n_embd + n_embd + hc) values per row: 51,216 bytes in FP32 or 25,608 bytes in BF16/FP16,
plus token IDs. The remote suffix computes and retains its own KV; KV tensors are never
read by the other PC. At 1 GbE, one 80 MiB FP32 prefill handoff has a serialization floor around 0.67 seconds, while
the 40 MiB BF16/FP16 handoff is about 0.34 seconds, at the theoretical 125 MB/s. Actual time includes protocol, copies,
waiting, and compute. There is no benchmark yet, and this mode may reduce host work or memory pressure while still
making prefill slower due to the link.

The smaller existing expert-cache RPC has fewer activation bytes but leaves all model layers and KV on the host. The
layer-stage mode can also reduce the host's suffix-layer GPU weights, session state, expert cache, and suffix expert
reads. It can reduce SSD/page-cache traffic only when those suffix expert pages were actually being read by the host;
measure host faults and disk reads before attributing a speedup to that effect.

## Protocol and safety

The protocol is single-session per connection, ordered, authenticated, and bounded. It includes reset, sampling,
checkpoint, resume, prompt prefill, verify, accepted-prefix commit, abort, and health operations. A checkpoint is
acknowledged only at a committed position, and rewinding discards checkpoints beyond that position. Duplicate operations replay their response only when the
full request identity matches; replay state is scoped to one connection. A disconnect aborts and clears the worker's
active session. The worker accepts clients sequentially; do not run multiple coordinators against one worker process.

Normal serve-request cancellation is cooperative at chunk/window boundaries and keeps the connection open. The
transport also supports explicit immediate cancellation for callers that intend to discard the connection; those
callers must reconnect and reset before sending fresh work. A timed-out state-changing request is never retried
because the remote outcome may be ambiguous. The server permits an idle connection before the first operation-header
byte, then applies one total deadline to the rest of the header, payload, execution, and reply.

## Optional Ray supervisor

This is native Strata inference over Strata's native TCP protocol. Ray only places, starts, and monitors the worker; it
does not run vLLM, provide vLLM's scheduler or batching, or inherit vLLM's performance. Ray's object store is not in
the activation path. The worker can run without Ray using the direct command above.

If Ray is installed and a cluster is running on the trusted LAN, start Ray's head on the 4090 PC and join the 5070 Ti
as a GPU worker. Then run the supervisor on the 4090 from C:\Strata. The worker config path is read on the 5070 Ti
and must be that PC's local path. With the release ZIP extracted to C:\Strata\engine, the script path below matches
the package layout:

    $env:STRATA_RPC_TOKEN = "replace-with-the-same-long-random-secret"
    .\.venv\Scripts\python.exe .\engine\layer_stage_ray.py --ray-address auto --worker-node-ip 192.168.1.50 --worker-config C:\Strata\strata-iq3_xxs-worker.json --split 24 --bind 0.0.0.0:39001

The script bounds actor placement and startup by --ready-timeout, prints the two client args to add to the 4090 run
config, streams worker logs, and makes a bounded attempt to stop the worker when you press Ctrl+C or startup fails. It
preserves Ray's one-GPU CUDA_VISIBLE_DEVICES assignment and refuses a conflicting worker-config override. Ray
multi-node support on Windows is experimental and has not been tested for this release; if it cannot run on these
Windows PCs, use the direct worker command instead. Ray does not improve TCP throughput or make the layer split faster
by itself.

## Validation before use

The Windows release workflow builds strata.exe, compiles the RPC transport and layer-stage tests, checks the Ray driver
helpers, and validates the layout and checksums of the actual ZIP. The fake-executor protocol tests do not prove
numerical equivalence or expert-cache integrity after a real model prefill, and no two-PC IQ3_XXS run or performance
benchmark has been completed for this release. Before relying on it, compare generated tokens against a same-day
single-GPU path with the same deterministic greedy request, including a multi-chunk prompt and repeated request. Then
measure prefill TPS, output TPS, total latency, both GPUs' VRAM, host RAM/commit/page faults/disk reads, and network
traffic. Keep it experimental until those checks pass.
