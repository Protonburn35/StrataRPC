#include "strata/core/rpc_transport.hpp"

#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

namespace {
using namespace strata::core::rpc;

bool set_socket_buffer(Socket socket, int option, int buffer_bytes, std::string& err) {
#if defined(_WIN32)
    if (setsockopt((SOCKET)socket, SOL_SOCKET, option, (const char*)&buffer_bytes, sizeof(buffer_bytes)) != 0) {
        err = "could not constrain RPC loopback socket buffers (Winsock " + std::to_string(WSAGetLastError()) + ")";
        return false;
    }
#else
    if (setsockopt((int)socket, SOL_SOCKET, option, &buffer_bytes, sizeof(buffer_bytes)) != 0) {
        err = "could not constrain RPC loopback socket buffers";
        return false;
    }
#endif
    return true;
}

bool constrain_socket_buffers(Socket sender, Socket receiver, std::string& err) {
    constexpr int buffer_bytes = 8192;
    return set_socket_buffer(sender, SO_SNDBUF, buffer_bytes, err) &&
           set_socket_buffer(receiver, SO_RCVBUF, buffer_bytes, err);
}

void shutdown_socket(Socket socket) {
#if defined(_WIN32)
    (void)::shutdown((SOCKET)socket, SD_BOTH);
#else
    (void)::shutdown((int)socket, SHUT_RDWR);
#endif
}

bool open_loopback(Socket& listener, Socket& client, Socket& peer, std::string& err,
                   int receive_buffer_bytes = 0) {
    const uint32_t seed = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t port = 0;
    for (uint32_t i = 0; i < 64 && listener == kInvalidSocket; ++i) {
        const uint32_t candidate = 20000 + (seed + i * 137) % 40000;
        err.clear();
        if (listen("127.0.0.1:" + std::to_string(candidate), listener, err)) port = candidate;
    }
    if (listener == kInvalidSocket) return false;
    // Set the listener's receive buffer before the TCP handshake. On Windows this prevents
    // receive-window autotuning from advertising a large default window to the sender.
    if (receive_buffer_bytes > 0 && !set_socket_buffer(listener, SO_RCVBUF, receive_buffer_bytes, err)) {
        close(listener);
        listener = kInvalidSocket;
        return false;
    }
    if (!connect("127.0.0.1:" + std::to_string(port), client, err) || !accept(listener, peer, err) ||
        !set_nodelay(client, true, err) || !set_nodelay(peer, true, err)) {
        close(client); close(peer); close(listener);
        client = peer = listener = kInvalidSocket;
        return false;
    }
    close(listener);
    listener = kInvalidSocket;
    return true;
}

bool partial_or_stalled_peer(bool send_prefix, bool cancel, bool disconnect, std::string& err) {
    Socket listener = kInvalidSocket, client = kInvalidSocket, peer = kInvalidSocket;
    if (!open_loopback(listener, client, peer, err)) return false;
    std::thread server([&] {
        if (send_prefix) {
            const uint8_t prefix = 0x5a;
            std::string ignored;
            (void)send_all(peer, &prefix, 1, deadline_after(std::chrono::seconds(1)), ignored);
        }
        if (!disconnect) std::this_thread::sleep_for(std::chrono::milliseconds(300));
        close(peer);
    });

    std::atomic<bool> cancelled{false};
    std::thread cancel_thread;
    if (cancel) cancel_thread = std::thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        cancelled.store(true, std::memory_order_relaxed);
    });
    uint32_t frame = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = deadline_after(cancel ? std::chrono::seconds(2) : std::chrono::milliseconds(100),
                                         cancel ? &cancelled : nullptr);
    const bool received = recv_all(client, &frame, sizeof(frame), deadline, err);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (cancel_thread.joinable()) cancel_thread.join();
    server.join();
    close(client);
    if (received) { err = "incomplete RPC frame was accepted"; return false; }
    if (cancel && (err != "RPC operation cancelled" || elapsed > std::chrono::milliseconds(500))) {
        err = "RPC cancellation did not interrupt the socket wait promptly: " + err;
        return false;
    }
    if (!cancel && !disconnect && (err != "RPC I/O deadline exceeded" || elapsed > std::chrono::seconds(1))) {
        err = "stalled RPC peer did not hit the total operation deadline: " + err;
        return false;
    }
    if (disconnect && err != "RPC connection closed") {
        err = "truncated RPC frame did not report a closed connection: " + err;
        return false;
    }
    return true;
}

bool expired_deadlines_reject_immediate_io(std::string& err) {
    Socket listener = kInvalidSocket, client = kInvalidSocket, peer = kInvalidSocket;
    if (!open_loopback(listener, client, peer, err)) return false;
    const uint32_t buffered = 0x12345678u;
    if (!send_all(peer, &buffered, sizeof(buffered), err)) {
        close(client); close(peer); return false;
    }
    IoDeadline expired{std::chrono::steady_clock::now() - std::chrono::milliseconds(1), nullptr, true};
    uint32_t received = 0;
    if (recv_all(client, &received, sizeof(received), expired, err) || err != "RPC I/O deadline exceeded") {
        close(client); close(peer);
        err = "an expired deadline accepted bytes that were already buffered";
        return false;
    }
    err.clear();
    if (send_all(client, &buffered, sizeof(buffered), expired, err) || err != "RPC I/O deadline exceeded") {
        close(client); close(peer);
        err = "an expired deadline accepted an immediately writable socket";
        return false;
    }
    close(client); close(peer);
    return true;
}

bool progressing_transfer_keeps_one_total_budget(std::string& err) {
    Socket listener = kInvalidSocket, client = kInvalidSocket, peer = kInvalidSocket;
    constexpr int kSocketBufferBytes = 8192;
    if (!open_loopback(listener, client, peer, err, kSocketBufferBytes)) return false;
    if (!constrain_socket_buffers(client, peer, err)) {
        close(client); close(peer);
        return false;
    }
    constexpr size_t kChunk = 4096;
    constexpr size_t kPayloadBytes = 128u << 20;
    std::atomic<size_t> received{0};
    std::atomic<bool> prefix_seen{false};
    std::atomic<bool> receiver_ready{false};
    std::thread server([&] {
        std::vector<uint8_t> block(kChunk);
        std::string server_err;
        if (!recv_all(peer, block.data(), block.size(), deadline_after(std::chrono::seconds(3)), server_err)) return;
        received.fetch_add(block.size(), std::memory_order_relaxed);
        prefix_seen.store(true, std::memory_order_release);
        receiver_ready.store(true, std::memory_order_release);
        while (received.load(std::memory_order_relaxed) < kChunk + kPayloadBytes) {
            if (!recv_all(peer, block.data(), block.size(), deadline_after(std::chrono::seconds(3)), server_err)) return;
            received.fetch_add(block.size(), std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    std::vector<uint8_t> payload(kPayloadBytes, 0x6b);
    std::vector<uint8_t> prefix(kChunk, 0xdd);
    if (!send_all(client, prefix.data(), prefix.size(), err)) {
        close(client); server.join(); close(peer); return false;
    }
    const auto barrier_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!prefix_seen.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < barrier_deadline)
        std::this_thread::yield();
    if (!prefix_seen.load(std::memory_order_acquire)) {
        close(client); server.join(); close(peer);
        err = "the progress test peer did not receive its initial bytes";
        return false;
    }
    while (!receiver_ready.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < barrier_deadline)
        std::this_thread::yield();
    if (!receiver_ready.load(std::memory_order_acquire)) {
        close(client); server.join(); close(peer);
        err = "the progress test peer was not ready for payload bytes";
        return false;
    }

    std::atomic<bool> sender_done{false};
    bool sent = false;
    std::string send_err;
    const auto deadline = deadline_after(std::chrono::seconds(1));
    std::thread sender([&] {
        sent = send_all(client, payload.data(), payload.size(), deadline, send_err);
        sender_done.store(true, std::memory_order_release);
    });
    while (!sender_done.load(std::memory_order_acquire) &&
           received.load(std::memory_order_relaxed) <= kChunk &&
           std::chrono::steady_clock::now() < deadline.expires + std::chrono::milliseconds(250))
        std::this_thread::yield();
    const bool observed_progress = received.load(std::memory_order_relaxed) > kChunk;
    sender.join();
    err = send_err;
    const size_t progress = received.load(std::memory_order_relaxed);
    close(client);
    shutdown_socket(peer);
    server.join();
    close(peer);
    if (sent || err != "RPC I/O deadline exceeded" || !observed_progress ||
        progress <= kChunk || progress >= kChunk + kPayloadBytes) {
        err = "a progressing transfer did not stop at its original total deadline (sent=" +
              std::string(sent ? "true" : "false") + ", error='" + send_err + "', received=" +
              std::to_string(progress) + ", observed_progress=" +
              std::string(observed_progress ? "true" : "false") + ")";
        return false;
    }
    return true;
}
} // namespace

int main() {
    using namespace strata::core::rpc;
    std::string err;
    Socket listener = kInvalidSocket;
    const uint32_t seed = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t bound_port = 0;
    for (uint32_t i = 0; i < 64 && listener == kInvalidSocket; ++i) {
        const uint32_t candidate = 20000 + (seed + i * 137) % 40000;
        err.clear();
        if (listen("127.0.0.1:" + std::to_string(candidate), listener, err)) bound_port = candidate;
    }
    if (listener == kInvalidSocket) {
        std::fprintf(stderr, "could not reserve an RPC loopback port: %s\n", err.c_str());
        return 1;
    }

    Socket client = kInvalidSocket;
    if (!connect("127.0.0.1:" + std::to_string(bound_port), client, err)) {
        close(listener);
        std::fprintf(stderr, "could not connect to the reserved RPC loopback port: %s\n", err.c_str());
        return 1;
    }

    Socket peer = kInvalidSocket;
    if (!accept(listener, peer, err) || !set_nodelay(client, true, err) || !set_nodelay(peer, true, err)) {
        close(client);
        close(peer);
        close(listener);
        std::fprintf(stderr, "RPC loopback setup failed: %s\n", err.c_str());
        return 1;
    }
    close(listener);

    std::vector<uint8_t> sent(1u << 20), server_received(sent.size()), client_received(sent.size());
    for (size_t i = 0; i < sent.size(); ++i) sent[i] = (uint8_t)((i * 131u + i / 17u) & 0xffu);
    bool server_ok = false;
    std::thread server([&] {
        std::string server_err;
        server_ok = recv_all(peer, server_received.data(), server_received.size(), server_err) &&
                    server_received == sent &&
                    send_all(peer, sent.data(), sent.size(), server_err);
    });
    const bool exchange_ok = send_all(client, sent.data(), sent.size(), err) &&
                             recv_all(client, client_received.data(), client_received.size(), err);
    server.join();
    close(client);
    close(peer);
    if (!exchange_ok || !server_ok || sent != client_received) {
        std::fprintf(stderr, "RPC loopback exchange failed%s%s\n", err.empty() ? "" : ": ", err.c_str());
        return 1;
    }
    if (!partial_or_stalled_peer(false, false, false, err) ||
        !partial_or_stalled_peer(true, false, false, err) ||
        !partial_or_stalled_peer(true, false, true, err) ||
        !partial_or_stalled_peer(true, true, false, err) ||
        !expired_deadlines_reject_immediate_io(err) ||
        !progressing_transfer_keeps_one_total_budget(err)) {
        std::fprintf(stderr, "RPC bounded I/O regression failed: %s\n", err.c_str());
        return 1;
    }
    std::puts("RPC transport loopback, deadlines, truncation and cancellation passed");
    return 0;
}
