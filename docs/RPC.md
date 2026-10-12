# Expert-cache RPC (experimental)

Strata's RPC helper lets a host use a second Strata process as an expert-cache tier. The server loads the same expert
pack on its local GPU, accepts one client, and fills its available VRAM with the highest-ranked pairs the host does not
already keep. During decode verify windows, the host sends an activation batch and routed expert IDs for a layer; the
server computes the rows for its resident experts while the host CPU pool computes other misses. By default the server
returns each remote row. With `--remote-expert-opt`, it applies router weights on the server and returns one weighted
sum per token, reducing return traffic when several remote experts serve one token.

This RPC speaks Strata's expert-kernel protocol. It is not wire-compatible with llama.cpp's ggml RPC backend, which
serves ggml graph and tensor operations. Strata's expert cache is static for the life of the connection. Both machines
need a compatible CUDA Strata build and identical expert bytes. The server does not receive model files over the
network, and it does not host KV state or run attention. A server pack with `experts.bin` needs only that file and its
layout metadata. The normal IQ3_XXS setup may omit `experts.bin`; in that case, keep the local model GGUF shards on the
server and pass shard 1 with `--native`. The server reads expert tensors from those shards, so a full second model
install is convenient but its dense weights, PLE data, and MTP files are not used by this mode.
At connection setup the peers compare per-layer expert blob sizes and hashes of up to eight sampled cached experts;
that catches a wrong pack quickly but is not a cryptographic proof that every expert matches.

## Start the server

On the 5070 Ti machine, set the same private token that the client will use, then start the GPU process. The pack path
must point to the local IQ3_XXS expert pack.

```powershell
$env:STRATA_RPC_TOKEN = "<a-long-random-secret>"
.\strata.exe --rpc-server 0.0.0.0:50552 `
  --pack C:\Strata-data\packs\iq3_xxs `
  --native C:\Strata\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
```

`--native` is required when the pack has no `experts.bin`; it can also be supplied when the file exists. Use the same
IQ3_XXS shards and pack as the host. Provision those files once with Strata setup or copy them from the host; the RPC
engine does not download model data at startup. For the attached host configuration, shard 1 is
`C:\Strata\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` and shard 2 is beside it.

Use an explicit private-LAN address for the listener. The server takes its cache size from free VRAM, preserving at
least 512 MiB for CUDA and its work buffers. It accepts one host connection and exits when that client disconnects.
The authentication token is sent in clear text; use this only on a trusted LAN or a VPN, never on an exposed network.

## Connect the 4090 host

On the host, set the same token and add `--expert-cache-rpc <5070-Ti-IP>:50552` to the Strata engine args. Keep the
normal local `--expert-cache auto` and `--expert-profile <profile.bin>` settings so the local 4090 keeps the hottest
experts and the remote card takes the next ranked experts. The remote helper requires the host's expert pool and a
profile. It supports one RPC helper, with no same-process helper GPU, peer tier, or layer split. Use
`--remote-expert-opt` for the reduced-result path; omit it to compare raw row transfers. The option changes floating-
point summation order, so exact output equality is not promised.

To give the remote PC complete expert blocks for selected layers, add e.g. `--expert-cache-rpc-layers 20-39`. The
host removes every expert from those layers from its own cache ranking and repacks its available slots over the other
layers. The server must fit all experts for every selected layer or startup fails; select fewer layers if it cannot.
This moves expert computation only. Attention, dense weights, per-layer state, and KV remain on the host, and the
prompt path still computes all experts locally. A selected expert block is not a remote layer stage.

The remote GPU participates in decode verify windows, including speculative windows. Prompt prefill remains entirely
on the host: this RPC path does not reduce SSD reads or paging during the prompt and should not raise prefill speed.
The host still maps its expert source, and the server's copy of the weights does not free host system RAM. Repacking
the host VRAM cache can reduce decode-time reads for its remaining layers; it does not remove expert files from the
host or address prefill paging. Per-request reports include RPC bytes sent and received; those counters exclude TCP,
IP, Ethernet framing, and retransmissions.

## Capacity and network limits

For this model family the repository reports about 13.7 KiB of KV state per context token in the streaming path. That
is roughly 2.1 GiB at 160K and 3.5 GiB at 256K, before small allocator and rope-table overheads; the exact amount
depends on the selected KV format and model geometry. A 5070 Ti has enough raw VRAM capacity for a 3.5 GiB cache,
though any weights, work buffers, and Windows display use reduce the remaining space. The 4090 cannot directly
address that VRAM over Ethernet. If it kept running attention, remote KV access would require sending KV blocks over
the link repeatedly and would be far slower than local VRAM. The viable design is to keep the KV on the 5070 Ti and
run attention for those layers there too; then the cache stays local and only layer-boundary activations cross the
network. That needs a remote layer-stage protocol with persistent per-conversation state, which this expert-cache RPC
does not implement.

The RPC helper instead spends the 5070 Ti's 16 GiB on expert weights. A useful capacity estimate is the GPU's actual
free memory after Windows reserves display and driver allocations; it is not the card's full advertised VRAM. The
server reports the number and bytes of experts it accepted at startup. This adds expert-cache capacity on the 5070 Ti;
it does not reduce the 4090's local cache allocation, mapped expert source, dense weights, or KV state. During decode,
the selected remote rows no longer need to be read and computed by the host, which may reduce its expert-file reads
for those rows. Prompt prefill still reads and computes its experts on the host. The returned decode activations cross
the network, so the result depends on how much host CPU and disk time the RPC can hide.

Ethernet still adds a serial dependency at each remote-owned MoE layer. With a five-row verify window (`--spec 4`),
each layer sends about 50 KiB of FP32 activations. Returning two to four raw expert rows per token adds roughly
100–200 KiB per layer, or about 7–12 MiB across 48 layers per window. The 1 Gb/s line rate is 125 MB/s before TCP
overhead. The user observed around 200 Mb/s while a model was active in an earlier setup; that is about 25 MB/s, a
context-specific observation rather than a benchmark of this RPC. At that rate, raw-row traffic alone would take
roughly 0.3–0.5 seconds per five-row window if it traversed the link serially.

With `--remote-expert-opt`, the return side is one 50 KiB weighted sum per layer per five-row window. The host still
sends about 50 KiB per layer, so all 48 layers move about 4.8 MiB in both directions together per window (roughly
0.96 MiB per accepted token), before protocol overhead. At 25 MB/s that is about 0.2 seconds per window as a
bandwidth-only floor, or about 25 tokens/s if every byte is serial. If complete expert blocks are assigned to only
20 layers, the estimate is about 2 MiB per window, or 0.08 seconds at that rate (about 62 tokens/s before compute and
round-trip latency). The pool overlaps some local CPU work with remote GPU work, so actual totals depend on hit
coverage and scheduling; these calculations are not speed claims.

Each selected layer still requires one sequential request/response. For a 20-layer assignment, a 1 ms LAN RTT adds
about 4 ms per accepted token at a five-row window; all 48 layers would add about 9.6 ms/token. Measure RTT and the
RPC counters on both PCs. The raw-row arm is useful as a correctness/reference comparison; the reduced-result arm
and complete-block assignment are the network-conscious experiments.

On a 1 GbE link, a contiguous remote layer stage with its own KV remains the more network-efficient direction for
decode and long-context capacity: it would exchange the verifier handoff once per stage boundary rather than making
one request per remote-owned MoE layer. In this model the handoff is 12,804 FP32 values (about 50 KiB/token); a
2,048-token chunk is about 100 MiB per boundary. That is about 0.84 seconds at the 1 Gb/s line rate, or about
4.2 seconds at the user's earlier 200 Mb/s observation, before synchronization or compute. A persistent stage must
also keep QSA/GDN state, KV, PLE history, and speculative commit/rollback state on the remote GPU. This runtime is not
implemented by the expert-cache RPC modes described above.
