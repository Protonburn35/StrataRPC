#pragma once

#include "strata/core/layer_stage_codec.hpp"
#include "strata/core/rpc_transport.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

enum LayerStageCapability : uint32_t {
    kLayerStageCapabilityBase = 1u,
    kLayerStageCapabilityReuse = 2u,
    kLayerStageCapabilityMtp = 4u,
    kLayerStageCapabilitySampling = 8u,
    kLayerStageCapabilityVision = 16u,
};

constexpr uint16_t kLayerStageProtocolVersion = 5;
constexpr uint32_t kLayerStageMaxMtpDrafts = 7;
constexpr uint32_t kLayerStageMaxCheckpoints = 8;

/// Per-request settings, sampled on the terminal worker. Draws are keyed by seed and absolute token position.
/// No device pointers, vocabulary logits, or mutable RNG state cross the connection.
struct LayerStageSampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int32_t top_k = 0;
    uint64_t seed = 0;
};
bool layer_stage_valid_sampling(const LayerStageSampling& sampling);
/// Commit only inputs whose outputs will be emitted; discarded speculative tokens must not enter the live prefix.
uint32_t layer_stage_output_commit_count(const int32_t* picks, uint32_t accepted, int64_t remaining,
                                        const std::vector<int64_t>& eos_ids);

/// Fixed-size reply keeps draft output lossless and makes commit acknowledgement replay deterministic.
struct LayerStageMtpReply {
    uint32_t count = 0;
    std::array<int32_t, kLayerStageMaxMtpDrafts> tokens{};
    std::array<float, kLayerStageMaxMtpDrafts> probabilities{};
};

/// Wire-visible identity and limits for one contiguous model stage. Fingerprints are computed by the coordinator from
/// Strata's existing model/config identity; paths are deliberately absent because the two PCs may store files apart.
struct LayerStageSpec {
    uint64_t model_fingerprint = 0;
    uint64_t config_fingerprint = 0;
    uint64_t max_context = 0;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_expert = 0;
    uint32_t hc = 0;
    uint32_t layer_begin = 0;
    uint32_t layer_end = 0;
    uint32_t max_verify_rows = 0;
    uint32_t max_prefill_rows = 0;
    uint32_t capabilities = 0;
    LayerStageWireFormat wire_format = LayerStageWireFormat::fp32;
};

struct LayerStagePrefixIdentity {
    uint64_t hash_a = 0, hash_b = 0;
};

/// Stable identity for the exact committed token prefix, independent of vector capacity or allocator address.
LayerStagePrefixIdentity layer_stage_prefix_identity(const int32_t* tokens, size_t count);

/// A worker owns its CUDA context, weights, per-layer session state and scratch for the lifetime of this object.
/// Activation pointers are valid only for the duration of the callback and are never sent as addresses on the wire.
class LayerStageExecutor {
public:
    virtual ~LayerStageExecutor() = default;
    virtual bool reset(uint64_t session, uint64_t generation, int64_t position, std::string& err) = 0;
    virtual bool resume(uint64_t session, uint64_t old_generation, uint64_t new_generation, int64_t position,
                        std::string& err) = 0;
    virtual bool set_sampling(const LayerStageSampling& sampling, std::string& err) = 0;
    /// Full absolute (t,h,w) table, including generation positions. Image embeddings stay on the coordinator.
    virtual bool set_positions(const std::vector<int32_t>& positions, std::string& err) {
        (void)positions; err = "this worker has no vision position table"; return false;
    }
    /// Save only running state; the existing KV prefix remains resident. Retention is chosen by the coordinator.
    virtual bool checkpoint(int64_t position, const std::vector<int32_t>& tokens,
                            const std::vector<int64_t>& retained, std::string& err) = 0;
    virtual bool prefill(uint64_t session, uint64_t generation, uint64_t operation, int64_t position, uint32_t rows,
                         const int64_t* tokens, const float* residual, std::string& err) = 0;
    virtual bool verify(uint64_t session, uint64_t generation, uint64_t operation, int64_t position, uint32_t rows,
                        const int32_t* tokens, const float* handoff, int32_t* picks, std::string& err) = 0;
    virtual bool commit(uint64_t session, uint64_t generation, uint64_t operation, uint32_t keep,
                        bool draft_next, float min_p, LayerStageMtpReply* reply, std::string& err) = 0;
    virtual bool abort(uint64_t session, uint64_t generation, uint64_t operation, std::string& err) = 0;
    virtual bool health(std::string& err) = 0;
};

enum class LayerStageCancelMode {
    interrupt_io,
    finish_window, // coordinator checks STOP between prompt chunks / complete verify+commit windows
};

/// Persistent request coordinator endpoint for one remote layer stage. Use the same object for all requests so its
/// TCP connection and per-operation replay state survive between prompt chunks and decode windows.
class LayerStageClient {
public:
    LayerStageClient() = default;
    ~LayerStageClient();
    LayerStageClient(const LayerStageClient&) = delete;
    LayerStageClient& operator=(const LayerStageClient&) = delete;

    bool connect(const std::string& endpoint, const std::string& token, const LayerStageSpec& expected,
                 std::string& err);
    void close();
    /// finish_window leaves network deadlines active but defers STOP to the coordinator. In particular a
    /// verify and its commit must both finish even when STOP arrives between their RPCs, so the next request
    /// can reuse the connection and known state. interrupt_io closes an interrupted connection instead.
    void set_cancel_flag(const std::atomic<bool>* flag, LayerStageCancelMode mode = LayerStageCancelMode::interrupt_io) {
        cancelled_ = mode == LayerStageCancelMode::interrupt_io ? flag : nullptr;
    }
    bool connected() const { return socket_ != rpc::kInvalidSocket && !broken_; }
    bool reset(uint64_t session, uint64_t generation, int64_t position, std::string& err);
    bool resume(uint64_t session, uint64_t generation, int64_t position, LayerStagePrefixIdentity prefix,
                std::string& err);
    bool set_sampling(uint64_t session, uint64_t generation, const LayerStageSampling& sampling, std::string& err);
    bool set_positions(uint64_t session, uint64_t generation, const std::vector<int32_t>& positions, std::string& err);
    bool checkpoint(uint64_t session, uint64_t generation, int64_t position, LayerStagePrefixIdentity prefix,
                    const std::vector<int64_t>& retained, std::string& err);
    bool prefill(uint64_t session, uint64_t generation, int64_t position, uint32_t rows, const int64_t* tokens,
                 const float* residual, std::string& err);
    bool verify(uint64_t session, uint64_t generation, int64_t position, uint32_t rows, const int32_t* tokens,
                const float* handoff, int32_t* picks, std::string& err);
    bool commit(uint64_t session, uint64_t generation, uint32_t keep, std::string& err) {
        return commit(session, generation, keep, false, 0.0f, nullptr, err);
    }
    bool commit(uint64_t session, uint64_t generation, uint32_t keep, bool draft_next, float min_p,
                LayerStageMtpReply* reply, std::string& err);
    bool abort(uint64_t session, uint64_t generation, std::string& err);
    bool health(std::string& err);
    uint64_t bytes_sent() const { return bytes_sent_; }
    uint64_t bytes_received() const { return bytes_received_; }
    LayerStageWireFormat wire_format() const { return spec_.wire_format; }

private:
    bool request(uint16_t type, const uint8_t* metadata, size_t metadata_bytes, const void* body_a,
                 size_t body_a_bytes, const void* body_b, size_t body_b_bytes, void* response,
                 size_t response_bytes, std::string& err);
    rpc::Socket socket_ = rpc::kInvalidSocket;
    LayerStageSpec spec_{};
    std::string endpoint_, token_;
    const std::atomic<bool>* cancelled_ = nullptr;
    uint64_t next_operation_ = 1;
    uint64_t bytes_sent_ = 0, bytes_received_ = 0;
    bool broken_ = false;
    std::vector<uint8_t> activation_wire_;
};

/// One accepted client connection. Reuse the service object after a disconnect to retain the loaded model, but reset
/// the session before accepting new work: disconnects invalidate in-flight state. A new service object loads weights
/// only once; its executor decides how an interrupted session is discarded or replayed.
class LayerStageServer {
public:
    LayerStageServer(std::string token, LayerStageSpec expected, LayerStageExecutor& executor,
                     std::chrono::milliseconds operation_timeout = std::chrono::minutes(5));
    bool serve(rpc::Socket client, std::string& err);

private:
    bool dispatch(uint16_t type, uint64_t operation, const uint8_t* payload, size_t payload_bytes,
                  std::vector<uint8_t>& response, std::string& err);
    std::string token_;
    LayerStageSpec spec_{};
    LayerStageExecutor& executor_;
    std::chrono::milliseconds operation_timeout_;
    std::vector<uint8_t> payload_buffer_;
    std::vector<int64_t> prefill_tokens_;
    std::vector<int32_t> verify_tokens_, verify_picks_;
    std::vector<float> activation_buffer_;
    uint64_t session_ = 0, generation_ = 0, last_operation_ = 0;
    uint64_t last_request_bytes_ = 0, last_request_hash_a_ = 0, last_request_hash_b_ = 0;
    std::vector<int32_t> committed_tokens_;
    std::vector<int64_t> checkpoint_positions_;
    int64_t position_ = 0;
    uint32_t pending_verify_rows_ = 0;
    uint16_t last_type_ = 0;
    std::vector<uint8_t> last_response_;
    bool session_valid_ = false;
    bool positions_ready_ = false;
};

} // namespace strata::core
