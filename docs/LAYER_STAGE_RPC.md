# Persistent layer-stage RPC (experimental)

The optional `--layer-stage-client` / `--layer-stage-server` mode splits one native model across two Strata processes.
The coordinator owns layers `0..K-1`; the worker owns `K..last`, including those layers' recurrent state and KV. The
coordinator sends FP32 residual rows at prompt-chunk boundaries and verifier-window boundaries. The worker keeps its
CUDA context, layer weights, expert cache, prefill path, verifier, and session state alive for the life of the process.
The transport is the bounded native TCP protocol implemented in `layer_stage_rpc.cpp`; Ray is not in the activation
data path.

This is a prototype for performance and capacity experiments, not a benchmarked speedup. It is deliberately limited
to one serial text request at a time, a static expert profile, greedy sampling, a native model pack, `--spec >= 2`,
and the same explicit fixed `--prefill N`, model, runtime configuration, and `--layer-stage-split K` on both PCs. Both
machines need their own local model files and Strata installation. The worker command is a protocol server, not an
HTTP generation endpoint. The client remains the API endpoint. Prompt caching and conversation parking are disabled,
so every request starts at position zero and rereads the complete prompt. MTP, batching, images, per-request sampling,
penalties, logit bias, cvec overrides, and expert-cache RPC are currently refused.

## Windows two-PC setup

Pick one unused TCP port and permit it through the 5070 Ti PC's Windows Firewall on the trusted LAN. Set the same
secret in PowerShell on both PCs for each terminal; use a long random value:

```powershell
$env:STRATA_RPC_TOKEN = "replace-with-a-long-random-secret"
```

Start the 5070 Ti worker first. Replace the paths and common arguments with those from that PC's
`C:\Strata\strata-iq3_xxs.json` run config. Run the bundled engine directly here: the normal Python API wrapper
expects an HTTP-capable `READY` line, while the worker intentionally exposes only the layer-RPC socket.

```powershell
cd C:\Strata
& .\engine\strata.exe --serve --gpu 0 --pack C:\Strata\model --native C:\Strata\native.gguf `
  --layer-stage-server 0.0.0.0:39001 --layer-stage-split K --prefill N --spec 4 `
  --max-context 65536 --kv fp16 --expert-profile C:\Strata\profile.bin --expert-cache auto
```

Include all the worker's model-specific args from its run config in that command. `--serve`, `--layer-stage-server`,
and `--layer-stage-split K` are added for this role. The worker stays in that terminal while it serves RPC requests.

On the 4090, add `--layer-stage-client 192.168.1.50:39001` and the same `--layer-stage-split K` to the `args` array
in its run config, then start Strata normally:

```powershell
cd C:\Strata
Start-Process -FilePath .\START-HERE.bat
```

Keep the normal `host` and `port` settings in this run config; the 4090 remains the HTTP API endpoint. The client's
model-specific args include the same `--prefill N`, `--spec`, context, KV, native kernels, and static profile as the
worker. Set `STRATA_RPC_TOKEN` in the environment inherited by the worker and API server. `K` is the first layer
assigned to the 5070 Ti. Start with a split that leaves the 5070 Ti enough VRAM for its suffix
weights, KV/state, verifier scratch, and expert cache; then tune `K` and compare against a single-4090 run. Strata
checks the model/config fingerprint, geometry, split, verify limit, and prefill chunk at connect time. A mismatch
refuses the connection. The RPC secret is sent over plaintext TCP, so use a trusted LAN or VPN and never expose this
port to the public internet.

The setup command shown earlier enabled vision. Remove `--vision` and any vision-related args from both run configs
for this prototype; layer-stage mode currently rejects vision even for text-only requests.

Set the coordinator run config's sampling default to greedy (`"sampling": {"temperature": 0}`) and send requests with
greedy sampling. The API wrapper forwards per-request sampling fields to the engine, which returns a clear `ERR` for
sampling, penalties, logit bias, minimum-probability filtering, images, or cvec overrides the remote worker cannot yet
apply consistently.

## What crosses the link

For the reviewed Qwen geometry (`n_embd=2560`, `hc=4`), prompt boundaries carry `hc*n_embd` FP32 values per token:
40 KiB/token, or 80 MiB for a 2,048-token chunk. The verifier handoff carries `(hc*n_embd + n_embd + hc)` FP32 values
per row, 51,216 bytes/row, plus token IDs. The remote suffix computes and retains its own KV; KV tensors are never
read by the other PC. At 1 GbE, one 80 MiB prefill handoff has a serialization floor around 0.67 seconds at the
theoretical 125 MB/s, before protocol, copies, waiting, or compute. There is no benchmark yet, and this mode may
reduce host work or memory pressure while still making prefill slower due to the link.

The smaller existing expert-cache RPC has fewer activation bytes but leaves all model layers and KV on the host. The
layer-stage mode can also reduce the host's suffix-layer GPU weights, session state, expert cache, and suffix expert
reads. It can reduce SSD/page-cache traffic only when those suffix expert pages were actually being read by the host;
measure host faults and disk reads before attributing a speedup to that effect.

The protocol is single-session per connection, ordered, authenticated, and bounded. It includes reset, prompt prefill,
verify, accepted-prefix commit, abort, and health operations. Duplicate last operations replay their prior response to
avoid applying a state transition twice. A disconnect aborts and clears the worker's active session. The worker accepts
clients sequentially; do not run multiple coordinators against one worker process.

## Optional Ray supervisor

The release also includes `layer_stage_ray.py`, an optional Ray actor supervisor. Ray starts the worker on the selected
node and monitors its output; the worker still sends activations directly to the coordinator over Strata's native TCP
protocol. Ray's object store is deliberately not in the inference data path, since it would add serialization and
copies for these FP32 tensors. The layer-stage worker can run without Ray using the direct command above.

If Ray is already installed and a cluster is running on the trusted LAN, start Ray's head on the 4090 PC and join the
5070 Ti as a GPU worker. Then run the supervisor on the 4090 from the Strata folder (the worker config path is read on
the 5070 Ti, so it must be that PC's local path):

```powershell
$env:STRATA_RPC_TOKEN = "replace-with-the-same-long-random-secret"
.\.venv\Scripts\python.exe tools\layer_stage_ray.py `
  --ray-address auto --worker-node-ip 192.168.1.50 `
  --worker-config C:\Strata\strata-iq3_xxs.json --split K --bind 0.0.0.0:39001
```

The script waits for `LAYER_STAGE_READY`, prints the two client args to add to the 4090 run config, streams worker
logs, and stops the worker when you press Ctrl+C. Ray setup and platform support are external to this release; in
particular, Ray availability on Windows depends on the Ray version and Python environment. If Ray cannot run on the
Windows PCs, use the direct worker command instead. Ray does not improve the TCP link throughput or make the layer
split faster by itself.

## Validation before use

The Windows release workflow builds `strata.exe`, compiles the RPC transport and layer-stage tests, and runs those
protocol tests. Those tests do not prove numerical equivalence or performance for a real model split. Before relying
on this experimental mode, compare the generated tokens against the single-GPU path with the same deterministic
greedy request, then measure prompt prefill TPS, output TPS, total latency, both GPUs' VRAM/use, host RAM/commit/page
faults/disk reads, and network throughput. No benchmark is bundled or run automatically.
