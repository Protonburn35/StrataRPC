// src/dsv41/ds4_stage.hpp - StrataRPC: DeepSeek-V4.1-Flash split over two PCs by layers (a layer stage).
//
// The coordinator (the PC the API runs on) holds the embedding table, the Engram tables and layers 0 .. split - 1; the
// worker holds layers split .. the last and the head.  Each token's row crosses the link once, as the hand-off between
// two GPUs of one PC carries it: h, the hc x dim residual streams (in the negotiated wire format: FP32 exact, BF16 /
// FP16 half the bytes and lossy), and pre, 8 floats (always FP32).  The split must be a layer whose later layers read
// nothing from before it (V4.1: 20 - the compressed KV, index keys and candidates of layers 20-39 are layer 20's own),
// so nothing else crosses: no KV cache, no logits (the worker samples).
//
// The protocol: authenticated, ordered, one request at a time, each answered; a failure closes the connection (the
// coordinator connects again and both ends start over from position 0).  A prompt streams: PROMPT names its tokens,
// then ROWS frames carry its rows as the coordinator's last layer produces them (each acknowledged once the worker took
// it - the worker's GPU reads them as its first layer needs them), then the worker answers the PROMPT when its layers
// are through.  A step or a short window sends its rows with its tokens in one WINDOW.  Plain C++ (no CUDA), so the
// protocol is tested without a GPU (tests/dsv41/ds4_stage_test.cpp).
#pragma once

#include "strata/core/layer_stage_codec.hpp"
#include "strata/core/rpc_transport.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::ds4::stage {

using core::LayerStageWireFormat;

constexpr uint16_t kProtocol = 1;
constexpr int kPreFloats = 8;        // the hand-off's `pre` floats a row
constexpr int kPieceRows = 256;      // rows a ROWS frame carries at most (20 MiB of FP32 rows at V4.1's 4 x 5120)
constexpr int kMaxTokens = 1 << 22;  // tokens a PROMPT names at most

/// What both ends must agree on; the worker refuses a connection that differs.
struct Spec {
    uint64_t identity_a = 0, identity_b = 0;   // the model's identity (its geometry and tensors): the same model on both
    uint32_t n_layers = 0;                     // the whole model's layers
    uint32_t split = 0;                        // the worker's first layer
    uint32_t dim = 0, hc = 0, vocab = 0;
    uint64_t max_context = 0;
    LayerStageWireFormat wire = LayerStageWireFormat::fp32;
    size_t h_floats() const { return (size_t) hc * dim; }   // h's floats a row
};
bool same_spec(const Spec& a, const Spec& b, std::string& why);
bool parse_wire(const std::string& s, LayerStageWireFormat& f);
const char* wire_name(LayerStageWireFormat f);
/// A stable 128-bit hash of a text (the model identity the coordinator and the worker each build from their pack).
void hash_text(const std::string& s, uint64_t& a, uint64_t& b);

/// The model's cross-layer reads, as its geometry lists them (the model's layer numbers).
struct Layout {
    int n_layers = 0;
    std::vector<int> kv_src;      // layers that write compressed KV and index keys (later layers read the latest one)
    std::vector<int> ix_src;      // layers with an indexer of their own (later layers reuse its picks)
    std::vector<int> eg_layers;   // Engram layers (read the n-gram tables: the coordinator's)
    int cand_src = -1;            // the layer whose candidates later index layers use (-1: none)
};
/// Whether the model splits cleanly before layer `split` (the worker's first): nothing but the hand-off (h | pre)
/// crosses - the worker's first layer writes its own KV and index keys, no worker layer reads candidates from before the
/// split, and every Engram layer is the coordinator's.  V4.1 (sources 2, 8, 14, 20; Engram 1, 14; candidates 20): 20.
bool check_split(const Layout& m, int split, std::string& why);
/// The layer numbers of v in [b, e), shifted to start at 0 (a stage's own numbering).
std::vector<int> cut_layers(const std::vector<int>& v, int b, int e);
/// "2, 8, 14" (for messages and the identity).
std::string join(const std::vector<int>& v);

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int32_t top_k = 0;
    uint64_t seed = 0, counter = 0;
};

/// The rows for positions [p0, p0 + T): h [T][hc * dim], then pre [T][8].  false with err: they did not come (the
/// coordinator stopped the prompt, or the link failed).
using RowFn = std::function<bool(int64_t p0, int T, float* h, float* pre, std::string& err)>;

/// The worker's model as the server drives it (strata-dsv41 --layer-stage-server: a Model of layers split .. last).
class Backend {
public:
    virtual ~Backend() = default;
    virtual int64_t pos() const = 0;
    virtual void reset() = 0;
    virtual void checkpoint(int slot) = 0;
    virtual bool restore(int64_t limit) = 0;
    virtual void pin() = 0;
    /// n tokens whose input rows come from `rows` (snap_slot >= 0: Model::feed_mark into that checkpoint)
    virtual bool feed(const int32_t* tokens, int n, int logits_rows, int snap_slot, const RowFn& rows,
                      std::string& err) = 0;
    virtual int sample(const Sampling& s) = 0;
    virtual bool logits(std::vector<float>& out) = 0;
};

/// The coordinator's end: one connection to the worker, kept for the life of the engine (connected again after a
/// failure by the next reset()).
class Client {
public:
    Client() = default;
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    bool connect(const std::string& endpoint, const std::string& token, const Spec& spec, std::string& err);
    void close();
    bool connected() const { return socket_ != core::rpc::kInvalidSocket; }
    const Spec& spec() const { return spec_; }

    bool reset(std::string& err);
    bool checkpoint(int slot, int64_t pos, std::string& err);
    bool restore(int64_t limit, bool& restored, int64_t& pos, std::string& err);
    bool pin(std::string& err);
    /// a window whose rows are all known (a step, a short feed): the worker feeds them and answers when done
    bool window(int64_t p0, const int32_t* tokens, int n, int logits_rows, const float* h, const float* pre,
                int64_t& pos_after, std::string& err);
    /// a prompt as a stream of rows: begin, rows() as the coordinator's last layer gives them (in order, from p0),
    /// end() once they are all sent (or halt: the coordinator stopped early - the worker's feed ends too)
    bool prompt_begin(int64_t p0, const int32_t* tokens, int n, int logits_rows, int snap_slot, std::string& err);
    bool prompt_rows(int64_t p0, int T, const float* h, const float* pre, std::string& err);
    bool prompt_end(bool halt, int64_t& pos_after, std::string& err);
    bool sample(const Sampling& s, int& token, std::string& err);
    bool logits(std::vector<float>& out, std::string& err);

    uint64_t bytes_sent() const { return sent_; }
    uint64_t bytes_received() const { return received_; }
    double wait_ms() const { return wait_ms_; }   // time spent in requests (the network and the worker's work)

private:
    bool send(uint16_t type, const std::vector<uint8_t>& meta, const void* body, size_t body_bytes,
              const core::rpc::IoDeadline& deadline, std::string& err);
    bool reply(uint16_t type, std::vector<uint8_t>& body, const core::rpc::IoDeadline& deadline, std::string& err);
    bool call(uint16_t type, const std::vector<uint8_t>& meta, std::vector<uint8_t>& body, std::string& err,
              std::chrono::milliseconds timeout);
    bool rows_frame(int64_t p0, int T, const float* h, const float* pre, uint32_t flags, std::vector<uint8_t>& meta,
                    std::string& err);
    void fail();

    core::rpc::Socket socket_ = core::rpc::kInvalidSocket;
    Spec spec_{};
    uint64_t op_ = 0, prompt_op_ = 0;
    // the prompt being streamed: the end of its rows, the rows sent so far
    int64_t prompt_end_ = 0, prompt_sent_ = 0;
    bool prompt_open_ = false;
    std::vector<uint8_t> wire_;
    uint64_t sent_ = 0, received_ = 0;
    double wait_ms_ = 0;
};

/// The worker's end: serves one coordinator connection at a time.
class Server {
public:
    Server(std::string token, Spec spec, Backend& backend);
    /// one connection until it closes or fails; the backend is reset when it ends
    bool serve(core::rpc::Socket client, std::string& err);
    /// listen on `endpoint` and serve coordinators one after another (returns only on a listen failure)
    bool serve_forever(const std::string& endpoint, std::string& err);

private:
    std::string token_;
    Spec spec_;
    Backend& backend_;
};

}  // namespace strata::ds4::stage
