#include "strata/core/layer_stage_rpc.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strata::core;

class FakeStage final : public LayerStageExecutor {
public:
    bool reset(uint64_t session, uint64_t generation, int64_t position, std::string&) override {
        checkpoints.clear();
        session_ = session; generation_ = generation; position_ = position; pending_ = 0; ++resets; return true;
    }
    bool resume(uint64_t session, uint64_t old_generation, uint64_t new_generation, int64_t position,
                std::string& err) override {
        if (session != session_ || old_generation != generation_ || new_generation <= old_generation ||
            (position != position_ && std::find(checkpoints.begin(), checkpoints.end(), position) == checkpoints.end()) ||
            pending_ != 0) {
            err = "fake stage rejected a stale resume";
            return false;
        }
        generation_ = new_generation;
        position_ = position;
        checkpoints.erase(std::remove_if(checkpoints.begin(), checkpoints.end(),
                                         [&](int64_t at) { return at > position; }), checkpoints.end());
        ++resumes;
        return true;
    }
    bool set_sampling(const LayerStageSampling& value, std::string&) override {
        sampling = value; ++sampling_updates; return true;
    }
    bool checkpoint(int64_t position, const std::vector<int32_t>& tokens,
                    const std::vector<int64_t>& retained, std::string& err) override {
        if (position != position_ || tokens.size() != (size_t)position || pending_ != 0) {
            err = "fake checkpoint was not at the committed position"; return false;
        }
        checkpoints = retained; ++checkpoint_saves; return true;
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
        last_verify_token_ = tokens[0];
        pending_ = rows; ++verifies; return true;
    }
    bool commit(uint64_t session, uint64_t generation, uint64_t, uint32_t keep, bool draft_next, float min_p,
                LayerStageMtpReply* reply, std::string& err) override {
        const int delay = commit_delay_ms.exchange(0, std::memory_order_relaxed);
        if (delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        if (session != session_ || generation != generation_ || keep == 0 || keep > pending_) {
            err = "fake stage received an invalid accepted prefix"; return false;
        }
        if (draft_next && (reply == nullptr || min_p < 0.0f || min_p > 1.0f)) {
            err = "fake stage received invalid MTP commit fields"; return false;
        }
        if (reply != nullptr) {
            reply->count = draft_next ? std::min<uint32_t>(mtp_draft_count, kLayerStageMaxMtpDrafts) : 0;
            for (uint32_t i = 0; i < reply->count; ++i) {
                reply->tokens[i] = last_verify_token_ + 10 + (int32_t)i;
                reply->probabilities[i] = 0.75f - (float)i * 0.1f;
            }
        }
        position_ += keep; pending_ = 0; ++commits; return true;
    }
    bool abort(uint64_t, uint64_t, uint64_t, std::string&) override { ++aborts; pending_ = 0; return true; }
    bool health(std::string&) override { ++health_checks; return true; }

    int resets = 0, resumes = 0, prefills = 0, verifies = 0, commits = 0, aborts = 0, health_checks = 0;
    LayerStageSampling sampling;
    std::vector<int64_t> checkpoints;
    int sampling_updates = 0, checkpoint_saves = 0;
    uint32_t mtp_draft_count = 2;
    std::atomic<int> prefill_delay_ms{0}, verify_delay_ms{0}, commit_delay_ms{0};
private:
    uint64_t session_ = 0, generation_ = 0;
    int64_t position_ = 0;
    uint32_t pending_ = 0;
    int32_t last_verify_token_ = 0;
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
    put16(header.data() + 4, kLayerStageProtocolVersion); put16(header.data() + 6, type);
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
    if (std::memcmp(header.data(), "STGE", 4) != 0 || get16(header.data() + 4) != kLayerStageProtocolVersion ||
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
    append_u32(payload, (uint32_t)spec.wire_format);
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
    append_u64(prefill, (uint64_t)spec.n_embd * spec.hc * layer_stage_wire_element_bytes(spec.wire_format));
    const int64_t prompt_token = 10;
    std::vector<float> residual((size_t)spec.n_embd * spec.hc, 0.0f); residual[0] = 0.25f;
    append_raw(prefill, &prompt_token, sizeof(prompt_token)); append_raw(prefill, residual.data(), residual.size() * sizeof(float));
    if (ok) ok = raw_status_request(client, 3, 2, prefill, status, nullptr, err) && status == 0;

    std::vector<uint8_t> verify = session_meta(42, 1, 1);
    append_u32(verify, 1); append_u64(verify, (uint64_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc));
    append_u64(verify, (uint64_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc) *
                           layer_stage_wire_element_bytes(spec.wire_format));
    const int32_t draft_token = 20;
    std::vector<float> handoff((size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f); handoff[0] = -0.5f;
    append_raw(verify, &draft_token, sizeof(draft_token)); append_raw(verify, handoff.data(), handoff.size() * sizeof(float));
    std::vector<uint8_t> pick_body;
    int32_t picked = 0;
    if (ok) ok = raw_status_request(client, 4, 3, verify, status, &pick_body, err) && status == 0 &&
                 pick_body.size() == sizeof(int32_t);
    if (ok) { std::memcpy(&picked, pick_body.data(), sizeof(picked)); ok = picked == 21; }

    auto commit = session_identity(42, 1); append_u32(commit, 1); append_u32(commit, 0); append_u32(commit, 0);
    auto altered_commit = session_identity(42, 1); append_u32(altered_commit, 2); append_u32(altered_commit, 0); append_u32(altered_commit, 0);
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

bool test_resume_identity(const LayerStageSpec& spec, std::string& err) {
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

    LayerStageClient client;
    bool ok = client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err);
    const int64_t prompt = 10;
    const int32_t draft = 20;
    const float residual[16] = {0.25f};
    std::vector<float> handoff((size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f);
    handoff[0] = -0.5f;
    int32_t pick = 0;
    if (ok) ok = client.reset(42, 1, 0, err) && client.prefill(42, 1, 0, 1, &prompt, residual, err) &&
                 client.verify(42, 1, 1, 1, &draft, handoff.data(), &pick, err) &&
                 client.commit(42, 1, 1, err);
    const int32_t committed[] = {10, 20};
    LayerStagePrefixIdentity prefix = layer_stage_prefix_identity(committed, 2);
    LayerStagePrefixIdentity wrong = prefix;
    wrong.hash_a ^= 1;
    if (ok && (client.resume(42, 2, 2, wrong, err) || !client.connected())) {
        err = "worker accepted a mismatched committed-token identity or broke the connection";
        ok = false;
    }
    err.clear();
    if (ok) ok = client.resume(42, 2, 2, prefix, err) &&
                 client.verify(42, 2, 2, 1, &draft, handoff.data(), &pick, err) &&
                 client.commit(42, 2, 1, err) && client.abort(42, 2, err);
    client.close();
    server.join();
    if (!ok || !server_ok || !server_err.empty() || executor.resets != 1 || executor.resumes != 1 ||
        executor.prefills != 1 || executor.verifies != 2 || executor.commits != 2 || executor.aborts != 1) {
        if (err.empty()) err = server_err.empty() ? "exact layer-stage continuation failed" : server_err;
        return false;
    }
    return true;
}

// Exercise a prompt checkpoint, decode checkpoint, rewind/branch, and eviction over real TCP.
// Sampling must follow each request (including a switch back to greedy), not worker startup defaults.
bool test_checkpoint_and_sampling(LayerStageSpec spec, std::string& err) {
    const int32_t outputs[] = {20, 99, 21, 22};
    if (layer_stage_output_commit_count(outputs, 4, 10, {99}) != 2 ||
        layer_stage_output_commit_count(outputs, 4, 1, {99}) != 1 ||
        layer_stage_output_commit_count(outputs, 4, 10, {}) != 4 ||
        layer_stage_output_commit_count(outputs, 1, 10, {99}) != 1 ||
        layer_stage_output_commit_count(outputs, 4, 0, {}) != 0) {
        err = "speculative commit passed EOS or the output budget"; return false;
    }
    spec.capabilities |= kLayerStageCapabilitySampling;
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
        rpc::close(listener);
    });
    LayerStageClient client;
    bool ok = client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err) &&
              client.reset(42, 1, 0, err);
    LayerStageSampling sampled;
    sampled.temperature = 1.0f; sampled.top_p = 0.9f; sampled.min_p = 0.05f;
    sampled.top_k = 40; sampled.seed = UINT64_C(0xfedcba9876543210);
    if (ok) ok = client.set_sampling(42, 1, sampled, err) && executor.sampling_updates == 1 &&
                 executor.sampling.temperature == 1.0f && executor.sampling.top_p == 0.9f &&
                 executor.sampling.min_p == 0.05f && executor.sampling.top_k == 40 &&
                 executor.sampling.seed == sampled.seed;
    const int64_t prompt[] = {10, 11};
    const int32_t full[] = {10, 11, 20, 21};
    const auto prompt_id = layer_stage_prefix_identity(full, 2);
    const auto full_id = layer_stage_prefix_identity(full, 4);
    std::vector<float> residual(2 * spec.n_embd * spec.hc, 0.0f); residual[0] = 0.25f;
    std::vector<float> handoff(2 * (spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f);
    handoff[0] = -0.5f;
    int32_t picks[2]{};
    if (ok) ok = client.prefill(42, 1, 0, 2, prompt, residual.data(), err) &&
                 client.checkpoint(42, 1, 2, prompt_id, {2}, err) &&
                 client.verify(42, 1, 2, 2, full + 2, handoff.data(), picks, err);
    // A checkpoint must never capture a speculative state before commit.
    if (ok) ok = !client.checkpoint(42, 1, 2, prompt_id, {2}, err) &&
                 !client.set_sampling(42, 1, sampled, err);
    err.clear();
    if (ok) ok = client.commit(42, 1, 2, err) &&
                 client.checkpoint(42, 1, 4, full_id, {2, 4}, err);
    auto wrong = prompt_id; wrong.hash_b ^= 1;
    if (ok) ok = !client.resume(42, 2, 2, wrong, err);
    err.clear();
    if (ok) ok = client.resume(42, 2, 2, prompt_id, err) && executor.checkpoints == std::vector<int64_t>{2};
    LayerStageSampling greedy;
    if (ok) ok = client.set_sampling(42, 2, greedy, err) && executor.sampling_updates == 2 &&
                 executor.sampling.temperature == 0.0f && executor.sampling.top_k == 0 &&
                 executor.sampling.top_p == 1.0f && executor.sampling.min_p == 0.0f && executor.sampling.seed == 0;
    // A different continuation replaces the discarded branch. Its old checkpoint cannot return.
    const int32_t branch[] = {30, 31};
    const int32_t branch_full[] = {10, 11, 30, 31};
    const auto branch_id = layer_stage_prefix_identity(branch_full, 4);
    if (ok) ok = client.verify(42, 2, 2, 2, branch, handoff.data(), picks, err) && picks[0] == 31 &&
                 client.commit(42, 2, 2, err) &&
                 !client.checkpoint(42, 2, 4, full_id, {2, 4}, err);
    err.clear();
    if (ok) ok = client.checkpoint(42, 2, 4, branch_id, {4}, err) &&
                 !client.resume(42, 3, 2, prompt_id, err);
    err.clear();
    if (ok) ok = client.resume(42, 3, 4, branch_id, err) &&
                 client.reset(42, 4, 0, err) && executor.checkpoints.empty() && client.abort(42, 4, err);
    client.close();
    server.join();
    if (!ok || !server_ok || !server_err.empty() || executor.checkpoint_saves != 3 || executor.resumes != 2) {
        if (err.empty()) err = server_err.empty() ? "checkpoint rewind/branch or request sampling failed" : server_err;
        return false;
    }
    for (int field = 0; field < 4; ++field) {
        auto bad = sampled;
        if (field == 0) bad.temperature = std::numeric_limits<float>::quiet_NaN();
        if (field == 1) bad.top_p = 1.1f;
        if (field == 2) bad.min_p = -0.1f;
        if (field == 3) bad.top_k = -1;
        if (layer_stage_valid_sampling(bad)) { err = "accepted invalid sampling settings"; return false; }
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

// A user replacing an active reply must not tear down a verify/commit transaction. Stop at its boundary,
// then start another generation on the SAME connection and executor, with no reset/reload of the worker.
bool test_coordinator_stop_keeps_connection(const LayerStageSpec& spec, std::string& err) {
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
        rpc::close(listener);
    });
    LayerStageClient client;
    std::atomic<bool> stop{false};
    client.set_cancel_flag(&stop, LayerStageCancelMode::finish_window);
    bool ok = client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err) &&
              client.reset(42, 1, 0, err);
    auto interrupted = [&](auto&& operation) {
        stop.store(false);
        std::thread cancel([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); stop.store(true); });
        const bool completed = operation();
        cancel.join();
        return completed && stop.load() && client.connected();
    };
    const int64_t prompt = 10;
    const int32_t tokens[] = {10, 20, 20};
    const float residual[16] = {0.25f};
    std::vector<float> handoff(spec.hc * spec.n_embd + spec.n_embd + spec.hc, 0.0f);
    handoff[0] = -0.5f;
    int32_t pick = 0;
    executor.prefill_delay_ms.store(150);
    if (ok) ok = interrupted([&] { return client.prefill(42, 1, 0, 1, &prompt, residual, err); }) &&
                 client.checkpoint(42, 1, 1, layer_stage_prefix_identity(tokens, 1), {1}, err) &&
                 client.resume(42, 2, 1, layer_stage_prefix_identity(tokens, 1), err);
    executor.verify_delay_ms.store(150);
    if (ok) ok = interrupted([&] { return client.verify(42, 2, 1, 1, tokens + 1, handoff.data(), &pick, err); }) &&
                 pick == 21 && client.commit(42, 2, 1, err) && // STOP remains set: commit still must finish
                 client.resume(42, 3, 2, layer_stage_prefix_identity(tokens, 2), err);
    if (ok) ok = client.verify(42, 3, 2, 1, tokens + 2, handoff.data(), &pick, err);
    executor.commit_delay_ms.store(150);
    if (ok) ok = interrupted([&] { return client.commit(42, 3, 1, err); }) &&
                 client.resume(42, 4, 3, layer_stage_prefix_identity(tokens, 3), err) && client.health(err);
    // An edited prompt can rewind to the earlier jointly saved checkpoint after interruption, too.
    if (ok) ok = client.resume(42, 5, 1, layer_stage_prefix_identity(tokens, 1), err) && client.abort(42, 5, err);
    client.set_cancel_flag(nullptr);
    client.close();
    server.join();
    if (!ok || !server_ok || !server_err.empty() || executor.resets != 1 || executor.resumes != 4 ||
        executor.prefills != 1 || executor.verifies != 2 || executor.commits != 2 || executor.aborts != 1) {
        if (err.empty()) err = server_err.empty() ? "coordinator STOP dropped its resident stage or continuation" : server_err;
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

bool test_activation_codecs(std::string& err) {
    auto encode_decode = [&](LayerStageWireFormat format, const std::vector<float>& input,
                             std::vector<float>& output) {
        std::vector<uint8_t> wire;
        if (!layer_stage_encode_activations(format, input.data(), input.size(), wire, err)) return false;
        output.resize(input.size());
        return layer_stage_decode_activations(format, wire.data(), wire.size(), output.data(), output.size(), err);
    };
    std::vector<float> out;
    if (!encode_decode(LayerStageWireFormat::fp32, {1.0f, -2.5f, 1.0e-30f}, out) ||
        out != std::vector<float>{1.0f, -2.5f, 1.0e-30f}) {
        if (err.empty()) err = "FP32 activation codec changed reference values";
        return false;
    }
    if (!encode_decode(LayerStageWireFormat::bf16,
                       {1.0f, -2.5f, 1.0e35f, 1.0e-35f, 1.00390625f, 1.01171875f}, out) ||
        out[0] != 1.0f || out[1] != -2.5f || !std::isfinite(out[2]) || out[2] == 0.0f ||
        out[3] == 0.0f || out[4] != 1.0f || out[5] != 1.015625f) {
        if (err.empty()) err = "BF16 activation rounding/range check failed";
        return false;
    }
    if (!encode_decode(LayerStageWireFormat::fp16,
                       {1.0f, -2.5f, 65504.0f, 5.960464477539063e-8f,
                        1.00048828125f, 1.00146484375f}, out) ||
        out[0] != 1.0f || out[1] != -2.5f || out[2] != 65504.0f ||
        out[3] != 5.960464477539063e-8f || out[4] != 1.0f || out[5] != 1.001953125f) {
        if (err.empty()) err = "FP16 activation rounding/subnormal/range check failed";
        return false;
    }
    for (LayerStageWireFormat format : {LayerStageWireFormat::bf16, LayerStageWireFormat::fp16}) {
        std::vector<uint8_t> wire;
        if (layer_stage_encode_activations(format, std::vector<float>{std::numeric_limits<float>::infinity()}.data(),
                                           1, wire, err)) {
            err = "activation codec accepted infinity";
            return false;
        }
        err.clear();
        if (layer_stage_encode_activations(format, std::vector<float>{std::numeric_limits<float>::quiet_NaN()}.data(),
                                           1, wire, err)) {
            err = "activation codec accepted NaN";
            return false;
        }
        err.clear();
    }
    std::vector<uint8_t> wire;
    const float too_large = 65520.0f;
    if (layer_stage_encode_activations(LayerStageWireFormat::fp16, &too_large, 1, wire, err)) {
        err = "FP16 activation codec accepted finite range overflow";
        return false;
    }
    err.clear();
    float decoded = 0.0f;
    if (layer_stage_decode_activations(LayerStageWireFormat::fp16, nullptr, 0, &decoded, 1, err)) {
        err = "activation codec accepted a truncated payload";
        return false;
    }
    err.clear();
    return true;
}

bool test_reduced_precision_rpc(const LayerStageSpec& base, LayerStageWireFormat format, std::string& err) {
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    LayerStageSpec spec = base;
    spec.wire_format = format;
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
    LayerStageClient client;
    bool ok = client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err);
    const int64_t token = 10;
    const float residual[] = {0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int32_t draft = 20;
    std::vector<float> handoff((size_t)(base.hc * base.n_embd + base.n_embd + base.hc), 0.0f);
    handoff[0] = -0.5f;
    int32_t pick = 0;
    if (ok) ok = client.reset(42, 1, 0, err) && client.prefill(42, 1, 0, 1, &token, residual, err) &&
                 client.verify(42, 1, 1, 1, &draft, handoff.data(), &pick, err) && pick == 21 &&
                 client.commit(42, 1, 1, err) && client.abort(42, 1, err);
    client.close();
    server.join();
    if (!ok || !server_ok || !server_err.empty() || executor.prefills != 1 || executor.verifies != 1 ||
        executor.commits != 1 || executor.aborts != 1) {
        if (err.empty()) err = server_err.empty() ? "reduced-precision wire transaction failed" : server_err;
        return false;
    }
    return true;
}

bool test_mtp_commit_response(const LayerStageSpec& base, std::string& err) {
    LayerStageSpec spec = base;
    spec.capabilities |= kLayerStageCapabilityMtp;
    rpc::Socket listener = rpc::kInvalidSocket;
    uint32_t port = 0;
    if (!open_listener(listener, port, err)) return false;
    FakeStage executor;
    executor.mtp_draft_count = 3;
    LayerStageServer service("test-token", spec, executor);
    bool server_ok = false;
    std::string server_err;
    std::thread server([&] {
        rpc::Socket accepted = rpc::kInvalidSocket;
        if (rpc::accept(listener, accepted, server_err)) server_ok = service.serve(accepted, server_err);
        else server_err = "accept failed: " + server_err;
        rpc::close(listener);
    });

    LayerStageClient client;
    bool ok = client.connect("127.0.0.1:" + std::to_string(port), "test-token", spec, err);
    const int64_t prompt = 10;
    const float residual[16] = {0.25f};
    const int32_t tokens[1] = {20};
    std::vector<float> handoff((size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f);
    handoff[0] = -0.5f;
    int32_t picks[1]{};
    LayerStageMtpReply reply;
    if (ok) ok = client.reset(42, 1, 0, err) && client.prefill(42, 1, 0, 1, &prompt, residual, err) &&
                 client.verify(42, 1, 1, 1, tokens, handoff.data(), picks, err) &&
                 client.commit(42, 1, 1, true, 0.1f, &reply, err);
    const int32_t next_tokens[4] = {21, 30, 31, 32};
    std::vector<float> next_handoff(4 * (size_t)(spec.hc * spec.n_embd + spec.n_embd + spec.hc), 0.0f);
    next_handoff[0] = -0.5f;
    int32_t next_picks[4]{};
    LayerStageMtpReply empty_reply;
    if (ok) ok = client.verify(42, 1, 2, 4, next_tokens, next_handoff.data(), next_picks, err) &&
                 client.commit(42, 1, 4, false, 0.0f, &empty_reply, err) && client.abort(42, 1, err);
    client.close();
    server.join();
    if (!ok || !server_ok || !server_err.empty() || reply.count != 3 || reply.tokens[0] != 30 ||
        reply.tokens[1] != 31 || reply.tokens[2] != 32 || std::fabs(reply.probabilities[0] - 0.75f) > 1e-6f ||
        std::fabs(reply.probabilities[1] - 0.65f) > 1e-6f || std::fabs(reply.probabilities[2] - 0.55f) > 1e-6f ||
        empty_reply.count != 0 || executor.verifies != 2 || executor.commits != 2) {
        if (err.empty()) err = server_err.empty() ? "remote MTP commit response failed" : server_err;
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
    if (!test_activation_codecs(err) || !test_reduced_precision_rpc(spec, LayerStageWireFormat::bf16, err) ||
        !test_reduced_precision_rpc(spec, LayerStageWireFormat::fp16, err) ||
        !test_mtp_commit_response(spec, err) || !test_checkpoint_and_sampling(spec, err) ||
        !test_replay_identity(spec, err) || !test_resume_identity(spec, err) ||
        !test_disconnect_clears_replay(spec, err) ||
        !test_partial_header_timeout_and_reaccept(spec, err) ||
        !test_client_cancellation_and_fresh_request(spec, err) ||
        !test_coordinator_stop_keeps_connection(spec, err) ||
        !test_malformed_success_closes_client(spec, err)) {
        std::fprintf(stderr, "layer-stage RPC safety regression failed: %s\n", err.c_str());
        return 1;
    }
    std::puts("layer-stage handshake, reset guards, replay identity, checkpoint rewind/branch, sampling updates, output bounds, reconnect, boundary STOP/resume, immediate cancellation, partial-header timeout, malformed replies, prefill, verify, commit and abort passed");
    return 0;
}
