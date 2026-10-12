#include "strata/core/rpc_transport.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

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
    std::puts("RPC transport loopback passed");
    return 0;
}
