#include "strata/core/rpc_transport.hpp"

#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strata::core::rpc;

bool open_loopback(Socket& listener, Socket& client, Socket& peer, std::string& err) {
    const uint32_t seed = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t port = 0;
    for (uint32_t i = 0; i < 64 && listener == kInvalidSocket; ++i) {
        const uint32_t candidate = 20000 + (seed + i * 137) % 40000;
        err.clear();
        if (listen("127.0.0.1:" + std::to_string(candidate), listener, err)) port = candidate;
    }
    if (listener == kInvalidSocket) return false;
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
        !partial_or_stalled_peer(true, true, false, err)) {
        std::fprintf(stderr, "RPC bounded I/O regression failed: %s\n", err.c_str());
        return 1;
    }
    std::puts("RPC transport loopback, deadlines, truncation and cancellation passed");
    return 0;
}
