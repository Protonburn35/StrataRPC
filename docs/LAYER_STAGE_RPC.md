# Persistent layer-stage RPC (experimental)

The optional --layer-stage-client / --layer-stage-server mode splits one native model across two Strata processes.
The coordinator owns layers 0..K-1; the worker owns K..last, including those layers' recurrent state and KV. The
coordinator sends FP32 residual rows at prompt-chunk boundaries and verifier-window boundaries. The worker keeps its
CUDA context, layer weights, expert cache, prefill path, verifier, and session state alive for the life of the process.
The transport is the bounded native TCP protocol implemented in layer_stage_rpc.cpp; Ray is not in the activation
data path.

This is a prototype for performance and capacity experiments, not a benchmarked speedup. It is deliberately limited
to one serial text request at a time, a static expert profile, greedy sampling, a native model pack, --spec >= 2,
and the same explicit fixed --prefill N, model, runtime configuration, and --layer-stage-split K on both PCs. Both
machines need their own local model files and Strata installation. The worker command is a protocol server, not an
HTTP generation endpoint. The client remains the API endpoint. Prompt caching and conversation parking are disabled,
so every request starts at position zero and rereads the complete prompt. MTP, batching, images, per-request sampling,
penalties, logit bias, cvec overrides, and expert-cache RPC are currently refused. The worker also requires the
non-finite-logit guard; do not set STRATA_NAN_GUARD=0 on it.

## Windows two-PC setup

Extract the Windows release ZIP into C:\Strata\engine on both PCs. Keep each PC's model shards, expert pack, profile,
and Python/API installation in the paths used by its run config. Both PCs need their own local model files. Pick one
unused TCP port and permit it through the 5070 Ti PC's Windows Firewall on the trusted LAN. Set the same secret in
PowerShell on both PCs for each terminal; use a long random value:

    $env:STRATA_RPC_TOKEN = "replace-with-a-long-random-secret"

For the attached Qwen3.8 Flash Next IQ3_XXS setup, this is a complete minimal worker config. Save it on the 5070 Ti as
C:\Strata\strata-iq3_xxs-worker.json. Replace paths only if the files live elsewhere; the profile must be the same
static profile used by the 4090. --prefill 2048 is a fixed chunk shared by both roles. The Ray worker config must not
put --gpu in its args: Strata would override Ray's CUDA_VISIBLE_DEVICES assignment. With one GPU reserved, the worker
sees its assigned card as local device zero. The example omits the ordinary
config's MTP, --spec-min-p, vision, and --kv-resident options because this prototype does not support them. Set
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
Set the address and K to match the worker. Temperature zero is required for the greedy prototype; do not add the
ordinary config's MTP, --spec-min-p, vision, or conversation reuse settings.

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

Layer-stage mode rejects vision even for text-only requests. The API wrapper forwards per-request sampling fields to
the engine, which returns a clear ERR for sampling changes, penalties, logit bias, minimum-probability filtering,
images, or cvec overrides the remote worker cannot yet apply consistently.

The worker reserves its owned prefill scratch before sizing the expert cache and disables cache borrowing, because
this persistent worker does not use the normal per-request cache loan/refill lifecycle. That can leave fewer resident
experts on the 5070 Ti, but it keeps prompt scratch outside the expert cache. Real-model cache integrity after a
multi-chunk prefill still needs the validation described below.

## What crosses the link

For the reviewed Qwen geometry (n_embd=2560, hc=4), prompt boundaries carry hc*n_embd FP32 values per token:
40 KiB/token, or 80 MiB for a 2,048-token chunk. The verifier handoff carries (hc*n_embd + n_embd + hc) FP32 values
per row, 51,216 bytes/row, plus token IDs. The remote suffix computes and retains its own KV; KV tensors are never
read by the other PC. At 1 GbE, one 80 MiB prefill handoff has a serialization floor around 0.67 seconds at the
theoretical 125 MB/s, before protocol, copies, waiting, or compute. There is no benchmark yet, and this mode may
reduce host work or memory pressure while still making prefill slower due to the link.

The smaller existing expert-cache RPC has fewer activation bytes but leaves all model layers and KV on the host. The
layer-stage mode can also reduce the host's suffix-layer GPU weights, session state, expert cache, and suffix expert
reads. It can reduce SSD/page-cache traffic only when those suffix expert pages were actually being read by the host;
measure host faults and disk reads before attributing a speedup to that effect.

## Protocol and safety

The protocol is single-session per connection, ordered, authenticated, and bounded. It includes reset, prompt prefill,
verify, accepted-prefix commit, abort, and health operations. Duplicate operations replay their response only when the
full request identity matches; replay state is scoped to one connection. A disconnect aborts and clears the worker's
active session. The worker accepts clients sequentially; do not run multiple coordinators against one worker process.

Cancellation closes the client connection when an operation is in flight. A following request reconnects and resets a
fresh generation; a timed-out state-changing request is never retried because the remote outcome may be ambiguous.
The server permits an idle connection before the first operation-header byte, then applies one total deadline to the
rest of the header, payload, execution, and reply. Cancellation does not interrupt a CUDA kernel already executing on
the worker; it only stops waiting on the local network operation and invalidates the session safely.

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
