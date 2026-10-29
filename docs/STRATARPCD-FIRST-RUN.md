# Vision and manual saved chats: A + C

Both PCs need this build. Its layer-stage protocol is version 3; it refuses an older peer.
This remains the A+C preset. Optional A+B+C configs and commands are in
[STRATARPCD-THREE-PC.md](STRATARPCD-THREE-PC.md); stop the engines before changing modes.
The supplied configs use IST-DAS 2.25 bpw, 1,048,576 tokens of context, BF16 stage transfers,
split 20, all six experts, default prompt chunks, and no DSpark. They explicitly enable vision
and manual disk sessions. Ordinary launchers keep both new features off unless requested.

Stop A and C before replacing their engines. Extract the artifact's files into
`C:\StrataRPCD\engine-rpcd` on both PCs, alongside the existing Python source/setup in
`C:\StrataRPCD`. Keep the model packs. The 2.25 pack must already exist at
`C:\Strata-data\packs\deepseek-v4.1-flash-ist-das-2.25bpw` on each PC.

The artifact includes `strata-stratarpcd-coordinator.json` and `strata-stratarpcd-worker.json`.
Copy the coordinator JSON to `C:\StrataRPCD` on A, and the worker JSON there on C. Use each PC's
own file. These presets assume CUDA GPU 0, A's API on port 8081, and C at 192.168.50.143:39501.

Start C first, with the same existing RPC secret in both windows:

```powershell
cd C:\StrataRPCD
$env:STRATA_RPC_TOKEN = "YOUR_EXISTING_SHARED_SECRET"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-worker.ps1 -Config .\strata-stratarpcd-worker.json -Context 1048576 -Wire bf16 -Split 20 -Listen 192.168.50.143:39501 -Vision -Sessions -Stats
```

Wait for C's `READY layer-stage worker`, then start A:

```powershell
cd C:\StrataRPCD
$env:STRATA_RPC_TOKEN = "YOUR_EXISTING_SHARED_SECRET"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\engine-rpcd\start-coordinator.ps1 -Config .\strata-stratarpcd-coordinator.json -Context 1048576 -Wire bf16 -Split 20 -Worker 192.168.50.143:39501 -Port 8081 -Vision -Sessions -Stats
```

The coordinator script regenerates its runtime JSON. Passing the supplied coordinator file explicitly keeps
these settings. No firewall change is needed when C already admits A on port 39501.

## Pictures

A's existing native GPU encoder reads pictures once, using roughly 319 MiB of VRAM plus host RAM.
C loads only its layers' image-routing biases; it runs no encoder. The coordinator sends image spans,
never pixels, to C. OpenAI image inputs, Anthropic image blocks, and the existing browser attachments
use the same server interface. The 2.25 pack on A must include `vision.*` / `aligner.*`, prepared from
the matching `mmproj-BF16.gguf`. A pack without that tower cannot enable split vision.

First check a small picture and ask a specific question about it. Then send another turn with the same
picture, switch to a text-only chat, and return to the picture. Compare against the single-PC engine
on the same model and prompt. GPU image/routing parity remains a hardware check.

## Save and restore a chat's model state

Run these on A after an answer finishes. The server serializes the command with inference.
Saving is manual: the automatic conversation-slot cache stays off.

```powershell
cd C:\StrataRPCD
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8081/slots/0?action=save' -ContentType 'application/json' -Body '{"filename":"chat-1.bin"}'
```

After switching chats or restarting the engines, restore before continuing that conversation:

```powershell
cd C:\StrataRPCD
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8081/slots/0?action=restore' -ContentType 'application/json' -Body '{"filename":"chat-1.bin"}'
```

A writes `C:\StrataRPCD\sessions\chat-1.bin`. C first writes its own immutable
`C:\StrataRPCD\sessions-worker\<snapshot-id>.srpc`; A's file references that ID. Keep both files.
Only the reference crosses the network. Model identity, split, context, wire format, vision mode,
position, token-prefix hash, native slot identity and file checksum are checked before use.
Restore reconnects to C, so it also works after a worker restart. A partial restore failure resets both
live halves; it cannot serve a mixture of two chats' caches.

These files save model state, including checkpoints and image-position/hash metadata. Keep the actual
chat history and attachments in your chat client, and send that same history when continuing. They
do not export a readable transcript or retain the client's request sampling settings. Use any normal
temperature. Restoring the cache does not rewind the serve loop's random-number counter or
reproduce an earlier sampled answer byte for byte.

Restores stage each PC's own file in RAM, with a 256 MiB margin; files over 8 GiB are refused. A full
1M-token state has not been hardware-tested here. Saves reserve an additional 64 MiB of disk space.
Files remain until deleted. Replacing A's filename leaves the former worker file, as does a failed
host save after C has published. To clear all snapshots, stop both engines and remove files from
both dedicated session folders. Protect them like the chats they contain.

## First checks

1. Ordinary text inference still works with the same prompt and sampling settings.
2. A picture is interpreted correctly, including a follow-up and a change of picture.
3. Save a short text chat, run an unrelated chat, restore, and continue the original with its history.
4. Repeat for an image chat and after restarting both engines. Look for reused prompt tokens in A's log.
5. Check a long saved chat after the short tests pass. Keep A's log and C's worker log if restore refuses.

Host-only protocol, file-integrity, launch-config and CUDA compilation checks do not establish GPU
restore correctness or performance. Those measurements require the two PCs.
