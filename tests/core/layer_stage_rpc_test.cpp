#include "strata/core/layer_stage_rpc.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
        if (session != session_ || generation != generation_ || position != position_ || tokens[0] != 10 ||
            residual[0] != 0.25f) { err = "fake stage received wrong prefill ordering or data"; return false; }
        position_ += rows; ++prefills; return true;
    }
    bool verify(uint64_t session, uint64_t generation, uint64_t, int64_t position, uint32_t rows,
                const int32_t* tokens, const float* handoff, int32_t* picks, std::string& err) override {
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
    client.close();
    server.join();

    if (!server_ok || !server_error.empty() || executor.resets != 1 || executor.prefills != 1 || executor.verifies != 1 ||
        executor.commits != 1 || executor.aborts != 1 || executor.health_checks != 1 ||
        client.bytes_sent() == 0 || client.bytes_received() == 0) {
        std::fprintf(stderr, "stage worker lifecycle failed: %s\n", server_error.c_str());
        return 1;
    }
    std::puts("persistent layer-stage handshake, prefill, verify, commit, health and abort passed");
    return 0;
}
