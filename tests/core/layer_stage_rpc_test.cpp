#include "strata/core/layer_stage_rpc.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strata::core;

class FakeStage final : public LayerStageExecutor {
public:
    bool reset(uint64_t session, uint64_t generation, int64_t position, std::string&) override {
        session_ = session; generation_ = generation; position_ = position; pending_ = 0; ++resets; return true;
    }
    bool prefill(uint64_t session, uint64_t generation, uint64_t, int64_t position, uint32_t rows,
                 const int64_t* tokens, const float* residual, std::string& err) override {
        const int delay = prefill_delay_ms.exchange(0, std::memory_order_relaxed);
        if (delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        if (session != session_ || generation != generation_ || position != position_ || tokens[0] != 10 ||
            residual[0] != 0.25f) { err = "fake stage received wrong prefill ordering or data"; return false; }
        position_ += rows; ++prefills; return true;
    }
    bool verify(uint64_t session, uint64_t generation, uint64_t, int64_t position, uint32_t rows,
                const int32_t* tokens, const float* handoff, int32_t* picks, std::string& err) override {
        const int delay = verify_delay_ms.exchange(0, std::memory_order_relaxed);
        if (delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        if (session != session_ || generation != generation_ || position != position_ || handoff[0] != -0.5f) {
            err = "fake stage received wrong verifier ordering or data"; return false;
        }
        for (uint32_t i = 0; i < rows; ++i) picks[i] = tokens[i] + 1;
        pending_ = rows; ++verifies; return true;
    }
    bool commit(uint64_t session, uint64_t generation, uint64_t, uint32_t keep, std::string& err) override {
        if (session != session_ || generation != generation_ || keep == 0 || keep > pending_) {
            err = "fake stage received an invalid accepted prefix"; return false;
        }
        position_ += keep; pending_ = 0; ++commits; return true;
    }
    bool abort(uint64_t, uint64_t, uint64_t, std::string&) override { ++aborts; pending_ = 0; return true; }
    bool health(std::string&) override { ++health_checks; return true; }

    int resets = 0, prefills = 0, verifies = 0, commits = 0, aborts = 0, health_checks = 0;
    std::atomic<int> prefill_delay_ms{0}, verify_delay_ms{0};
private:
    uint64_t session_ = 0, generation_ = 0;
    int64_t position_ = 0;
    uint32_t pending_ = 0;
};

bool open_listener(rpc::Socket& listener, uint32_t& port, std::string& err) {
    const uint32_t seed = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
    for (uint32_t i = 0; i < 64 && listener == rpc::kInvalidSocket; ++i) {
        const uint32_t candidate = 20000 + (seed + i * 137) % 40000;
        err.clear();
        if (rpc::listen("127.0.0.1:" + std::to_string(candidate), listener, err)) port = candidate;
    }
    return listener != rpc::kInvalidSocket;
}

void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
uint16_t get16(const uint8_t* p) { return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8); }
uint64_t get64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}

void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    const size_t at = out.size(); out.resize(at + 4); put32(out.data() + at, v);
}
void append_u64(std::vector<uint8_t>& out, uint64_t v) {
    const size_t at = out.size(); out.resize(at + 8); put64(out.data() + at, v);
}
void append_raw(std::vector<uint8_t>& out, const void* p, size_t n) {
    const size_t at = out.size(); out.resize(at + n);
    if (n != 0) std::memcpy(out.data() + at, p, n);
}

bool send_wire_frame(rpc::Socket socket, uint16_t type, uint64_t operation, const std::vector<uint8_t>& payload,
                     std::string& err) {
    std::array<uint8_t, 24> header{};
    std::memcpy(header.data(), "STGE", 4);
    put16(header.data() + 4, 1); put16(header.data() + 6, type);
    put64(header.data() + 8, operation); put64(header.data() + 16, payload.size());
    const auto deadline = rpc::deadline_after(std::chrono::seconds(2));
    return rpc::send_all(socket, header.data(), header.size(), deadline, err) &&
           (payload.empty() || rpc::send_all(socket, payload.data(), payload.size(), deadline, err));
}

bool receive_wire_frame(rpc::Socket socket, uint16_t expected_type, uint64_t expected_operation,
                        std::vector<uint8_t>& payload, std::string& err) {
    std::array<uint8_t, 24> header{};
    if (!rpc::recv_all(socket, header.data(), header.size(), rpc::deadline_after(std::chrono::seconds(2)), err)) return false;
    const uint64_t bytes = get64(header.data() + 16);
    if (std::memcmp(header.data(), "STGE", 4) != 0 || get16(header.data() + 4) != 1 ||
        get16(header.data() + 6) != expected_type || get64(header.data() + 8) != expected_operation || bytes > (1u << 20)) {
        err = "test peer received an invalid layer-stage frame header";
        return false;
    }
    payload.resize((size_t)bytes);
    return payload.empty() || rpc::recv_all(socket, payload.data(), payload.size(),
                                             rpc::deadline_after(std::chrono::seconds(2)), err);
}

bool raw_handshake(rpc::Socket socket, const LayerStageSpec& spec, const std::string& token, std::string& err) {
    std::vector<uint8_t> payload;
    append_u64(payload, spec.model_fingerprint); append_u64(payload, spec.config_fingerprint); append_u64(payload, spec.max_context);
    append_u32(payload, spec.n_layers); append_u32(payload, spec.n_embd); append_u32(payload, spec.n_expert);
    append_u32(payload, spec.hc); append_u32(payload, spec.layer_begin); append_u32(payload, spec.layer_end);
    append_u32(payload, spec.max_verify_rows); append_u32(payload, spec.max_prefill_rows); append_u32(payload, spec.capabilities);
    const uint32_t one = 1;
    append_u32(payload, *(const uint8_t*)&one == 1 ? 0x01020304u : 0x04030201u);
    const float fone = 1.0f;
    uint32_t fbits = 0; std::memcpy(&fbits, &fone, sizeof(fbits)); append_u32(payload, fbits);
    append_u32(payload, (uint32_t)token.size()); append_raw(payload, token.data(), token.size());
    if (!send_wire_frame(socket, 1, 0, payload, err)) return false;
    std::vector<uint8_t> response;
    if (!receive_wire_frame(socket, 0x8001u, 0, response, err) || response.size() != 8 ||
        get64(response.data()) != 0) {
        if (err.empty()) err = "test peer received a rejected or malformed layer-stage handshake";
        return false;
    }
    return true;
}

std::vector<uint8_t> session_meta(uint64_t session, uint64_t generation, int64_t position) {
    std::vector<uint8_t> payload;
    append_u64(payload, session); append_u64(payload, generation); append_u64(payload, (uint64_t)position);
    return payload;
}

std::vector<uint8_t> session_identity(uint64_t session, uint64_t generation) {
    std::vector<uint8_t> payload;
    append_u64(payload, session); append_u64(payload, generation);
    return payload;
}

bool raw_status_request(rpc::Socket socket, uint16_t type, uint64_t operation,
                        const std::vector<uint8_t>& request, uint32_t& status,
                        std::vector<uint8_t>* body, std::string& err) {
    if (!send_wire_frame(socket, type, operation, request, err)) return false;
    std::vector<uint8_t> response;
    if (!receive_wire_frame(socket, (uint16_t)(0x8000u | type), operation, response, err) || response.size() < 8) return false;
    status = (uint32_t)response[0] | ((uint32_t)response[1] << 8) | ((uint32_t)response[2] << 16) | ((uint32_t)response[3] << 24);
    const uint32_t message_bytes = (uint32_t)response[4] | ((uint32_t)response[5] << 8) |
                                   ((uint32_t)response[6] << 16) | ((uint32_t)response[7] << 24);
    if (message_bytes > response.size() - 8) { err = "test peer received a truncated response message"; return false; }
    if (body != nullptr) body->assign(response.begin() + 8 + message_bytes, response.end());
    return true;
}

bool connect_raw(const std::string& endpoint, const LayerStageSpec& spec, rpc::Socket& socket, std::string& err) {
    return rpc::connect(endpoint, socket, err) && rpc::set_nodelay(socket, true, err) && raw_handshake(socket, spec, "test-token", err);
}

bool test_replay_identity(const LayerStageSpec& spec, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    FakeStage executor;
    LayerStageServer service("test-token", spec, executor);
    bool server_ok = false;
    std::string server_err;
    std::thread server([&] {
        rpc::Socket accepted = rpc::kInvalidSocket;
        if (rpc::accept(listener, accepted, server_err)) server_ok = service.serve(accepted, server_err);
        else server_err = "accept failed: " + server_err;
        rpc::close(listener);
    });

    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    rpc::Socket client = rpc::kInvalidSocket;
    bool ok = connect_raw(endpoint, spec, client, err);
    uint32_t status = 2;
    auto reset = session_meta(42, 1, 0);
    auto invalid_reset = session_meta(42, 1, 1);
    if (ok) ok = raw_status_request(client, 2, 1, invalid_reset, status, nullptr, err) && status == 1;
    if (ok) ok = raw_status_request(client, 2, 1, reset, status, nullptr, err) && status == 0 && executor.resets == 1;
    if (ok) ok = raw_status_request(client, 2, 1, reset, status, nullptr, err) && status == 0 && executor.resets == 1;
    if (ok) ok = raw_status_request(client, 2, 1, invalid_reset, status, nullptr, err) && status == 1 && executor.resets == 1;

    std::vector<uint8_t> prefill = session_meta(42, 1, 0);
    append_u32(prefill, 1); append_u64(prefill, (uint64_t)spec.n_embd * spec.hc);
    const int64_t prompt_token = 10;
    std::vector<float> residual((size_t)spec.n_embd * spec.hc, 0.0f); residual[0] = 0.25f;
    append_raw(prefill, &prompt_token, sizeof(prompt_token)); append_raw(prefill, residual.data(), residual.size() * sizeof(float));
    if (ok) ok = raw_status_request(client, 3, 2, prefill, status, nullptr, err) && status == 0;

    std::vector<uint8_t> verify = session_meta(42, 1, 1);
    append_u32(verify, 1); append_u64(verify, (uint64_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc));
    const int32_t draft_token = 20;
    std::vector<float> handoff((size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f); handoff[0] = -0.5f;
    append_raw(verify, &draft_token, sizeof(draft_token)); append_raw(verify, handoff.data(), handoff.size() * sizeof(float));
    std::vector<uint8_t> pick_body;
    int32_t picked = 0;
    if (ok) ok = raw_status_request(client, 4, 3, verify, status, &pick_body, err) && status == 0 &&
                 pick_body.size() == sizeof(int32_t);
    if (ok) { std::memcpy(&picked, pick_body.data(), sizeof(picked)); ok = picked == 21; }

    auto commit = session_identity(42, 1); append_u32(commit, 1);
    auto altered_commit = session_identity(42, 1); append_u32(altered_commit, 2);
    if (ok) ok = raw_status_request(client, 5, 4, commit, status, nullptr, err) && status == 0 && executor.commits == 1;
    if (ok) ok = raw_status_request(client, 5, 4, commit, status, nullptr, err) && status == 0 && executor.commits == 1;
    if (ok) ok = raw_status_request(client, 5, 4, altered_commit, status, nullptr, err) && status == 1 && executor.commits == 1;
    if (ok) ok = raw_status_request(client, 5, 4, commit, status, nullptr, err) && status == 0 && executor.commits == 1;
    auto abort = session_identity(42, 1);
    if (ok) ok = raw_status_request(client, 6, 5, abort, status, nullptr, err) && status == 0 && executor.aborts == 1;
    if (ok) ok = raw_status_request(client, 6, 5, abort, status, nullptr, err) && status == 0 && executor.aborts == 1;
    rpc::close(client);
    server.join();
    if (!ok || !server_ok || !server_err.empty()) {
        if (err.empty()) err = server_err.empty() ? "replay identity checks failed" : server_err;
        return false;
    }
    return true;
}

bool test_disconnect_clears_replay(const LayerStageSpec& spec, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    FakeStage executor;
    LayerStageServer service("test-token", spec, executor);
    bool server_ok[2] = {false, false};
    std::string server_err[2];
    std::thread server([&] {
        for (int i = 0; i < 2; ++i) {
            rpc::Socket accepted = rpc::kInvalidSocket;
            if (rpc::accept(listener, accepted, server_err[i])) server_ok[i] = service.serve(accepted, server_err[i]);
            else server_err[i] = "accept failed: " + server_err[i];
        }
        rpc::close(listener);
    });
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    rpc::Socket client = rpc::kInvalidSocket;
    bool ok = connect_raw(endpoint, spec, client, err);
    uint32_t status = 2;
    auto reset1 = session_meta(42, 1, 0);
    if (ok) ok = raw_status_request(client, 2, 1, reset1, status, nullptr, err) && status == 0;
    rpc::close(client);
    client = rpc::kInvalidSocket;
    if (ok) ok = connect_raw(endpoint, spec, client, err);
    if (ok) ok = raw_status_request(client, 2, 1, reset1, status, nullptr, err) && status == 1;
    auto reset2 = session_meta(42, 2, 0);
    if (ok) ok = raw_status_request(client, 2, 1, reset2, status, nullptr, err) && status == 0;
    auto abort = session_identity(42, 2);
    if (ok) ok = raw_status_request(client, 6, 2, abort, status, nullptr, err) && status == 0;
    rpc::close(client);
    server.join();
    if (!ok || !server_ok[0] || !server_ok[1] || !server_err[0].empty() || !server_err[1].empty() ||
        executor.resets != 2 || executor.aborts != 2) {
        if (err.empty()) err = !server_err[0].empty() ? server_err[0] : server_err[1];
        if (err.empty()) err = "disconnect replay state was not invalidated";
        return false;
    }
    return true;
}

bool test_partial_header_timeout_and_reaccept(const LayerStageSpec& spec, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    FakeStage executor;
    LayerStageServer service("test-token", spec, executor, std::chrono::milliseconds(120));
    bool first_ok = true, second_ok = false;
    std::string first_err, second_err;
    std::thread server([&] {
        rpc::Socket first = rpc::kInvalidSocket;
        if (!rpc::accept(listener, first, first_err)) first_err = "first accept failed: " + first_err;
        else first_ok = service.serve(first, first_err);
        rpc::Socket second = rpc::kInvalidSocket;
        if (!rpc::accept(listener, second, second_err)) second_err = "second accept failed: " + second_err;
        else second_ok = service.serve(second, second_err);
        rpc::close(listener);
    });

    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    rpc::Socket stalled = rpc::kInvalidSocket;
    bool ok = connect_raw(endpoint, spec, stalled, err);
    const uint8_t first_byte = 'S';
    if (ok) ok = rpc::send_all(stalled, &first_byte, 1, rpc::deadline_after(std::chrono::seconds(1)), err);
    uint8_t reply = 0;
    if (ok) {
        std::string close_err;
        if (rpc::recv_all(stalled, &reply, 1, rpc::deadline_after(std::chrono::seconds(2)), close_err) ||
            close_err != "RPC connection closed") {
            err = "authenticated partial operation header was not closed at its deadline";
            ok = false;
        }
    }
    rpc::close(stalled);

    LayerStageClient fresh;
    if (ok) ok = fresh.connect(endpoint, "test-token", spec, err) && fresh.reset(42, 1, 0, err) &&
                 fresh.abort(42, 1, err);
    fresh.close();
    server.join();
    if (!ok || first_ok || first_err != "RPC I/O deadline exceeded" || !second_ok || !second_err.empty() ||
        executor.resets != 1 || executor.aborts != 1) {
        if (err.empty()) err = !first_err.empty() ? first_err : (!second_err.empty() ? second_err : "worker did not accept a fresh client after a partial header");
        return false;
    }
    return true;
}

bool test_client_cancellation_and_fresh_request(const LayerStageSpec& spec, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    FakeStage executor;
    LayerStageServer service("test-token", spec, executor, std::chrono::seconds(2));
    bool server_ok[3] = {false, false, false};
    std::string server_err[3];
    std::thread server([&] {
        for (int i = 0; i < 3; ++i) {
            rpc::Socket accepted = rpc::kInvalidSocket;
            if (rpc::accept(listener, accepted, server_err[i])) server_ok[i] = service.serve(accepted, server_err[i]);
            else server_err[i] = "accept failed: " + server_err[i];
        }
        rpc::close(listener);
    });

    LayerStageClient client;
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    bool ok = client.connect(endpoint, "test-token", spec, err) && client.reset(42, 1, 0, err);
    std::atomic<bool> cancelled{false};
    client.set_cancel_flag(&cancelled);
    const int64_t prompt_token = 10;
    std::vector<float> residual((size_t)spec.n_embd * spec.hc, 0.0f); residual[0] = 0.25f;
    executor.prefill_delay_ms.store(250, std::memory_order_relaxed);
    std::thread cancel_prefill([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); cancelled.store(true); });
    const auto prefill_started = std::chrono::steady_clock::now();
    const bool prefill_ok = client.prefill(42, 1, 0, 1, &prompt_token, residual.data(), err);
    const auto prefill_elapsed = std::chrono::steady_clock::now() - prefill_started;
    cancel_prefill.join();
    if (prefill_ok || err != "RPC operation cancelled" || prefill_elapsed > std::chrono::milliseconds(700)) ok = false;
    client.set_cancel_flag(nullptr);

    err.clear();
    if (ok) ok = client.reset(42, 2, 0, err);
    cancelled.store(false, std::memory_order_relaxed);
    client.set_cancel_flag(&cancelled);
    const int32_t draft_token = 20;
    std::vector<float> handoff((size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f); handoff[0] = -0.5f;
    int32_t pick = 0;
    executor.verify_delay_ms.store(250, std::memory_order_relaxed);
    std::thread cancel_verify([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); cancelled.store(true); });
    const auto verify_started = std::chrono::steady_clock::now();
    const bool verify_ok = client.verify(42, 2, 0, 1, &draft_token, handoff.data(), &pick, err);
    const auto verify_elapsed = std::chrono::steady_clock::now() - verify_started;
    cancel_verify.join();
    if (verify_ok || err != "RPC operation cancelled" || verify_elapsed > std::chrono::milliseconds(700)) ok = false;
    client.set_cancel_flag(nullptr);

    err.clear();
    if (ok) ok = client.reset(42, 3, 0, err) && client.prefill(42, 3, 0, 1, &prompt_token, residual.data(), err) &&
                 client.verify(42, 3, 1, 1, &draft_token, handoff.data(), &pick, err) && pick == 21 &&
                 client.commit(42, 3, 1, err) && client.abort(42, 3, err);
    client.close();
    server.join();
    if (!ok || !server_ok[2] || !server_err[2].empty() || executor.resets != 3 || executor.prefills != 2 ||
        executor.verifies != 2 || executor.commits != 1 || executor.aborts != 3) {
        if (err.empty()) err = !server_err[0].empty() ? server_err[0] :
                               (!server_err[1].empty() ? server_err[1] : (!server_err[2].empty() ? server_err[2] :
                                                                         "cancelled stage operation did not recover with a fresh request"));
        return false;
    }
    return true;
}

bool test_malformed_success_closes_client(const LayerStageSpec& spec, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    std::atomic<bool> applied{false};
    bool fake_ok = false;
    std::string fake_err;
    std::thread peer([&] {
        rpc::Socket accepted = rpc::kInvalidSocket;
        if (!rpc::accept(listener, accepted, fake_err)) { fake_err = "accept failed: " + fake_err; return; }
        std::vector<uint8_t> hello;
        if (!receive_wire_frame(accepted, 1, 0, hello, fake_err) || !send_wire_frame(accepted, 0x8001u, 0,
                                                                                      std::vector<uint8_t>(8, 0), fake_err)) {
            rpc::close(accepted); return;
        }
        std::vector<uint8_t> request;
        if (!receive_wire_frame(accepted, 2, 1, request, fake_err)) { rpc::close(accepted); return; }
        applied.store(true, std::memory_order_relaxed);  // model the state change before returning a corrupt reply
        std::vector<uint8_t> malformed(9, 0); malformed[8] = 1;
        fake_ok = send_wire_frame(accepted, 0x8002u, 1, malformed, fake_err);
        rpc::close(accepted);
    });
    LayerStageClient client;
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    bool ok = client.connect(endpoint, "test-token", spec, err);
    const bool reset_ok = ok && client.reset(42, 1, 0, err);
    const bool disconnected = !client.connected();
    client.close();
    peer.join();
    rpc::close(listener);
    if (!ok || reset_ok || !disconnected || !applied.load(std::memory_order_relaxed) || !fake_ok || !fake_err.empty()) {
        if (err.empty()) err = fake_err.empty() ? "client kept a connection after a malformed success reply" : fake_err;
        return false;
    }
    return true;
}
} // namespace

int main() {
    using namespace strata::core;
    LayerStageSpec spec;
    spec.model_fingerprint = 1; spec.config_fingerprint = 2; spec.max_context = 32;
    spec.n_layers = 4; spec.n_embd = 8; spec.n_expert = 4; spec.hc = 2;
    spec.layer_begin = 2; spec.layer_end = 4; spec.max_verify_rows = 4; spec.max_prefill_rows = 4;
    spec.capabilities = 3;

    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    std::string err;
    if (!open_listener(listener, port, err)) {
        std::fprintf(stderr, "could not open a stage-RPC loopback port: %s\n", err.c_str());
        return 1;
    }

    FakeStage executor;
    LayerStageServer service("test-token", spec, executor);
    bool server_ok = false;
    std::string server_error;
    std::thread server([&] {
        rpc::Socket client = rpc::kInvalidSocket;
        if (rpc::accept(listener, client, server_error)) server_ok = service.serve(client, server_error);
        else server_error = "accept failed: " + server_error;
        rpc::close(listener);
    });

    LayerStageClient client;
    if (!client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err)) {
        rpc::close(listener); server.join();
        std::fprintf(stderr, "stage-RPC handshake failed: %s (%s)\n", err.c_str(), server_error.c_str());
        return 1;
    }
    if (client.reset(42, 0, 0, err) || err.empty() || !client.connected()) {
        client.close(); server.join();
        std::fprintf(stderr, "worker did not reject an invalid reset while keeping the connection usable: %s\n", err.c_str());
        return 1;
    }
    err.clear();
    if (client.reset(42, 1, 1, err) || err.empty() || !client.connected()) {
        client.close(); server.join();
        std::fprintf(stderr, "worker did not reject a nonzero fresh-session position: %s\n", err.c_str());
        return 1;
    }
    err.clear();
    if (!client.reset(42, 1, 0, err)) { client.close(); server.join(); std::fprintf(stderr, "stage reset failed: %s\n", err.c_str()); return 1; }

    const int64_t prompt_tokens[] = {10, 11};
    std::vector<float> residual(2 * spec.n_embd * spec.hc, 0.0f);
    residual[0] = 0.25f;
    if (!client.prefill(42, 1, 0, 2, prompt_tokens, residual.data(), err)) {
        client.close(); server.join(); std::fprintf(stderr, "ordered stage prefill failed: %s\n", err.c_str()); return 1;
    }

    const int32_t draft_tokens[] = {20, 21, 22};
    const uint32_t verifier_floats = 3 * (spec.hc * spec.n_embd + spec.n_embd + spec.hc);
    std::vector<float> handoff(verifier_floats, 0.0f);
    handoff[0] = -0.5f;
    int32_t picks[3]{};
    if (!client.verify(42, 1, 2, 3, draft_tokens, handoff.data(), picks, err) ||
        picks[0] != 21 || picks[1] != 22 || picks[2] != 23) {
        client.close(); server.join();
        std::fprintf(stderr, "stage verifier result failed: %s\n", err.c_str());
        return 1;
    }
    if (!client.commit(42, 1, 2, err) || !client.health(err) || !client.abort(42, 1, err)) {
        client.close(); server.join(); std::fprintf(stderr, "stage state control failed: %s\n", err.c_str()); return 1;
    }
    // A new request reuses the TCP connection but must start operation numbering at one and advance its generation.
    if (!client.reset(42, 2, 0, err) || !client.health(err) || !client.abort(42, 2, err)) {
        client.close(); server.join(); std::fprintf(stderr, "stage generation reset failed: %s\n", err.c_str()); return 1;
    }
    client.close();
    server.join();

    if (!server_ok || !server_error.empty() || executor.resets != 2 || executor.prefills != 1 || executor.verifies != 1 ||
        executor.commits != 1 || executor.aborts != 2 || executor.health_checks != 2 ||
        client.bytes_sent() == 0 || client.bytes_received() == 0) {
        std::fprintf(stderr, "stage worker lifecycle failed: %s\n", server_error.c_str());
        return 1;
    }
    if (!test_replay_identity(spec, err) || !test_disconnect_clears_replay(spec, err) ||
        !test_partial_header_timeout_and_reaccept(spec, err) ||
        !test_client_cancellation_and_fresh_request(spec, err) ||
        !test_malformed_success_closes_client(spec, err)) {
        std::fprintf(stderr, "layer-stage RPC safety regression failed: %s\n", err.c_str());
        return 1;
    }
    std::puts("layer-stage handshake, reset guards, replay identity, reconnect, cancellation, partial-header timeout, malformed replies, prefill, verify, commit and abort passed");
    return 0;
}
