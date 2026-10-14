# Persistent layer-stage RPC (experimental API)

This is a native C++ service contract for a future persistent remote layer stage. It transfers FP32 boundary activations
over bounded TCP frames and keeps the worker's recurrent session state behind one persistent connection. The wire
operations are reset, prompt prefill, verify, accepted-prefix commit, abort, and health. Each connection has ordered
operation IDs, a model/configuration and layer-range handshake, a shared secret, replayed acknowledgements for a
duplicate last operation, and bounded I/O deadlines.

The current release includes the protocol, a generic `LayerStageExecutor` interface, and callbacks in `Prefill` and
`Verifier`. It does **not** include an in-tree model-owning executor, `strata.exe` launch options, or a Ray actor adapter.
The loopback test uses a fake executor to verify the protocol and state ordering. This API cannot yet run a model split
between two PCs; continue using the expert-cache RPC mode for a usable cross-PC path. There is no `--ray` option.

An executor implementation is responsible for keeping its CUDA context, weights, expert cache, session KV/GDN state,
prefill path, and verifier alive. It must reset or abort the session on disconnect and implement `commit(keep)` with
the same accepted-prefix semantics as `Verifier::commit`. The coordinator sends the prompt's boundary residual rows
and the verify window's residual, pending write, and injection values. Raw device pointers never cross the network.
Each connection is single-session and serial; concurrency requires a separate executor and connection per session.

With `hc=4` and `n_embd=2560`, a prefill boundary is 40 KiB per prompt token and a verify boundary is about 50 KiB per
row. A 2,048-token prefill chunk therefore sends about 80 MiB in one direction. At an ideal 1 Gb/s, that payload
alone takes about 0.67 seconds; at 25 MB/s it takes about 3.36 seconds before compute, copies, and framing. A five-row
verify window is about 250 KiB one way. These are wire-volume estimates, not benchmarks. A useful implementation
needs bounded overlap between local computation and transfers, then must be measured against one-PC execution.

The connection uses authenticated plaintext TCP, so bind it to a trusted LAN or VPN. FP32 payloads require compatible
native IEEE-754 and byte order. The wire handshake compares the supplied model/configuration fingerprints and stage
geometry; the caller is responsible for deriving fingerprints that uniquely identify the actual weights and runtime
configuration.
