#pragma once

#include "strata/core/rpc_transport.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

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
};

/// A worker owns its CUDA context, weights, per-layer session state and scratch for the lifetime of this object.
/// Activation pointers are valid only for the duration of the callback and are never sent as addresses on the wire.
class LayerStageExecutor {
public:
    virtual ~LayerStageExecutor() = default;
    virtual bool reset(uint64_t session, uint64_t generation, int64_t position, std::string& err) = 0;
    virtual bool prefill(uint64_t session, uint64_t generation, uint64_t operation, int64_t position, uint32_t rows,
                         const int64_t* tokens, const float* residual, std::string& err) = 0;
    virtual bool verify(uint64_t session, uint64_t generation, uint64_t operation, int64_t position, uint32_t rows,
                        const int32_t* tokens, const float* handoff, int32_t* picks, std::string& err) = 0;
    virtual bool commit(uint64_t session, uint64_t generation, uint64_t operation, uint32_t keep,
                        std::string& err) = 0;
    virtual bool abort(uint64_t session, uint64_t generation, uint64_t operation, std::string& err) = 0;
    virtual bool health(std::string& err) = 0;
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
    bool connected() const { return socket_ != rpc::kInvalidSocket && !broken_; }
    bool reset(uint64_t session, uint64_t generation, int64_t position, std::string& err);
    bool prefill(uint64_t session, uint64_t generation, int64_t position, uint32_t rows, const int64_t* tokens,
                 const float* residual, std::string& err);
    bool verify(uint64_t session, uint64_t generation, int64_t position, uint32_t rows, const int32_t* tokens,
                const float* handoff, int32_t* picks, std::string& err);
    bool commit(uint64_t session, uint64_t generation, uint32_t keep, std::string& err);
    bool abort(uint64_t session, uint64_t generation, std::string& err);
    bool health(std::string& err);
    uint64_t bytes_sent() const { return bytes_sent_; }
    uint64_t bytes_received() const { return bytes_received_; }

private:
    bool request(uint16_t type, const uint8_t* metadata, size_t metadata_bytes, const void* body_a,
                 size_t body_a_bytes, const void* body_b, size_t body_b_bytes, void* response,
                 size_t response_bytes, std::string& err);
    rpc::Socket socket_ = rpc::kInvalidSocket;
    LayerStageSpec spec_{};
    uint64_t next_operation_ = 1;
    uint64_t bytes_sent_ = 0, bytes_received_ = 0;
    bool broken_ = false;
};

/// One accepted client connection. Reuse the service object after a disconnect to retain the loaded model, but reset
/// the session before accepting new work: disconnects invalidate in-flight state. A new service object loads weights
/// only once; its executor decides how an interrupted session is discarded or replayed.
class LayerStageServer {
public:
    LayerStageServer(std::string token, LayerStageSpec expected, LayerStageExecutor& executor);
    bool serve(rpc::Socket client, std::string& err);

private:
    bool dispatch(uint16_t type, uint64_t operation, const uint8_t* payload, size_t payload_bytes,
                  std::vector<uint8_t>& response, std::string& err);
    std::string token_;
    LayerStageSpec spec_{};
    LayerStageExecutor& executor_;
    std::vector<uint8_t> payload_buffer_;
    std::vector<int64_t> prefill_tokens_;
    std::vector<int32_t> verify_tokens_, verify_picks_;
    uint64_t session_ = 0, generation_ = 0, last_operation_ = 0;
    int64_t position_ = 0;
    uint32_t pending_verify_rows_ = 0;
    uint16_t last_type_ = 0;
    std::vector<uint8_t> last_response_;
    bool session_valid_ = false;
};

} // namespace strata::core
