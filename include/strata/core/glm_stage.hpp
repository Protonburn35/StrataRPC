// include/strata/core/glm_stage.hpp - StrataRPCG: GLM-5.3-Flash split over two PCs by layers (a layer stage).
//
// The coordinator (the PC the API runs on) holds the embedding, the vision rows and layers 0 .. split - 1; the worker
// holds layers split .. the last and, by default, the head (--layer-stage-head coordinator moves the head and the
// sampling to the coordinator).  Each position's row crosses the link once, as the hop between two GPUs of one PC
// carries it: the residual's hc x n_embd streams, in the negotiated wire format (FP32 exact, BF16 / FP16 half the
// bytes and lossy).  Every GLM layer keeps its own state (a KDA layer its recurrent state and conv history, a DSA
// layer its latent cache and indexer keys), so any layer boundary is a clean split: nothing else crosses - no cache,
// no logits unless asked for.  With the head on the coordinator the worker sends back the last layer's streams
// instead of a token (always FP32: what the head reads is not rounded).
//
// The protocol: authenticated, ordered, one request at a time, each answered; a failure closes the connection (the
// coordinator connects again with its next reset and both ends start over from position 0).  A prompt streams: BEGIN
// names its length and the chunk the coordinator reads it in (the worker answers with the chunk it can take), then
// each chunk's rows in ROWS pieces (each acknowledged; a chunk's last piece once the worker's layers have read the
// chunk), then END.  Plain C++ (no CUDA): tests/core/glm_stage_test.cpp runs it without a GPU.
#pragma once

#include "strata/core/layer_stage_codec.hpp"
#include "strata/core/rpc_transport.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::glm::stage {

using core::LayerStageWireFormat;

constexpr uint16_t kProtocol = 1;              // update both PCs together
constexpr int kPieceRows = 512;                // rows a ROWS frame carries at most (32 MiB of FP32 at 4 x 4096)
constexpr int64_t kMaxPromptTokens = int64_t(1) << 22;
constexpr int kMaxChunk = 1 << 16;             // a prompt chunk's rows at most
constexpr int kMaxHistory = 1 << 16;           // a sample's penalty history at most
constexpr uint32_t kMaxBias = 1u << 20;        // a logit-bias table's entries at most

/// What both ends must agree on; the worker refuses a connection that differs.
struct Spec {
    uint64_t identity_a = 0, identity_b = 0;   // the model's identity (its pack and files): the same model on both
    uint32_t n_layers = 0;                     // the whole model's trunk layers
    uint32_t split = 0;                        // the worker's first layer
    uint32_t dim = 0, hc = 0, vocab = 0;
    uint64_t max_context = 0;
    bool head_on_coordinator = false;          // --layer-stage-head coordinator
    bool sessions = false;                     // paired disk sessions, explicitly enabled on both PCs
    LayerStageWireFormat wire = LayerStageWireFormat::bf16;
    size_t h_floats() const { return (size_t) hc * dim; }
};
bool same_spec(const Spec& a, const Spec& b, std::string& why);
bool parse_wire(const std::string& s, LayerStageWireFormat& f);
const char* wire_name(LayerStageWireFormat f);
/// A stable 128-bit hash of a text (the model identity both ends build from their pack).
void hash_text(const std::string& s, uint64_t& a, uint64_t& b);
/// Whether `split` leaves layers on both PCs (any boundary does for GLM-5.3-Flash).
bool check_split(int n_layers, int split, std::string& why);

/// A request's sampling for the worker's sampler (the head on the worker), as Strata's SamplerParams carries it.
struct Sampling {
    int32_t top_k = 20;
    float top_p = 0.95f, min_p = 0.0f, temperature = 1.0f;
    int32_t min_keep = 1, penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    uint64_t seed = 0, counter = 0;
    bool greedy = false;
    int32_t bias_slot = -1;                    // the worker's logit-bias table (Client::bias), -1: none
    std::vector<int32_t> history;              // the tokens fed so far (the penalties' window; empty without them)
};

/// A paired disk session's reference: the worker's file is <session dir>/<id>.gslot.
struct SnapshotRef {
    std::string id;                            // generated 32-character hex ID, never a path from the peer
    int64_t position = 0;
    uint64_t prefix = 0;                       // a hash of the tokens it holds (both files carry it)
};
bool valid_snapshot(const SnapshotRef& ref, uint64_t context);
std::string new_snapshot_id();
uint64_t prefix_hash(const std::vector<int32_t>& tokens);

/// The worker's model as the server drives it (maya --layer-stage-server: a Glm5Model of layers split .. last).
class Backend {
public:
    virtual ~Backend() = default;
    virtual int64_t pos() const = 0;
    virtual void reset() = 0;
    /// the snapshot (Glm5Model::snapshot_save / restore): false = none could be taken / none to restore
    virtual bool snap_save(std::string& err) = 0;
    virtual bool snap_restore(int64_t& pos, std::string& err) = 0;
    virtual int64_t snap_pos() const = 0;
    /// conversation checkpoints (Glm5Model::ckpt_save / ckpt_save_snapshot / ckpt_restore)
    virtual bool ckpt_save(int slot, bool from_snapshot, std::string& err) = 0;
    virtual bool ckpt_restore(int slot, int64_t& pos, std::string& err) = 0;
    virtual int64_t ckpt_pos(int slot) = 0;   // the position checkpoint `slot` holds, -1: none
    /// one position p: its input row (h_floats FP32) through the worker's layers, then the head's argmax into token
    /// (the head here) or the last layer's streams into r_out (h_floats FP32; the head on the coordinator)
    virtual bool step(int64_t p, const float* row, int& token, float* r_out, std::string& err) = 0;
    /// a prompt of n positions from p0, read in chunks of at most T rows: t_bound = the chunk this end takes (<= T)
    virtual bool prompt_begin(int64_t p0, int64_t n, int T, int& t_bound, std::string& err) = 0;
    /// the host buffer the next chunk's rows land in (t_bound x h_floats FP32), valid from prompt_begin to prompt_end
    virtual float* prompt_rows() = 0;
    virtual bool prompt_chunk(int64_t p0, int T, std::string& err) = 0;
    /// the prompt's end: completed (every chunk read) or stopped early; the worker's prompt memory goes back
    virtual bool prompt_end(bool completed, std::string& err) = 0;
    virtual int sample(const Sampling& s, std::string& err) = 0;
    virtual bool bias(int slot, const std::vector<std::pair<int32_t, float>>& b, std::string& err) = 0;
    virtual bool logits(std::vector<float>& out, std::string& err) = 0;
    /// paired disk sessions: the worker's half of the snapshot to / from <session dir>/<ref.id>.gslot
    virtual bool session(bool save, const SnapshotRef& ref, uint64_t& bytes, std::string& err) {
        (void) save;
        (void) ref;
        bytes = 0;
        err = "disk sessions are disabled";
        return false;
    }
};

/// The coordinator's end: one connection to the worker, kept for the life of the engine.
class Client {
public:
    Client() = default;
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    bool connect(const std::string& endpoint, const std::string& token, const Spec& spec, std::string& err);
    /// connect again with the last endpoint, token and spec (after a failure closed the link)
    bool reconnect(std::string& err);
    /// the last (re)connect reached the worker and was turned away (the token, a setting that differs) or had no token:
    /// trying again cannot help, unlike a worker whose port does not answer yet
    bool refused() const { return refused_; }
    void close();
    bool connected() const { return socket_ != core::rpc::kInvalidSocket; }
    const Spec& spec() const { return spec_; }
    /// how many times the link was (re)established - a change means the worker started over from position 0
    uint64_t epoch() const { return epoch_; }

    bool reset(std::string& err);
    bool snap_save(int64_t pos, bool& saved, std::string& err);
    bool snap_restore(int64_t want_pos, bool& restored, std::string& err);
    bool ckpt_save(int slot, bool from_snapshot, int64_t pos, bool& saved, std::string& err);
    bool ckpt_restore(int slot, int64_t want_pos, bool& restored, std::string& err);
    /// one position: row (h_floats FP32) out; token back (the head on the worker) or r_out (h_floats FP32)
    bool step(int64_t p, const float* row, int& token, float* r_out, std::string& err);
    bool prompt_begin(int64_t p0, int64_t n, int T, int& t_bound, std::string& err);
    /// a chunk's rows (T x h_floats FP32), in pieces; returns once the worker's layers have read them
    bool prompt_chunk(int64_t p0, int T, const float* rows, std::string& err);
    bool prompt_end(bool completed, int64_t& pos_after, std::string& err);
    bool sample(const Sampling& s, int& token, std::string& err);
    bool bias(int slot, const std::vector<std::pair<int32_t, float>>& b, std::string& err);
    bool logits(std::vector<float>& out, std::string& err);
    bool session(bool save, const SnapshotRef& ref, uint64_t& bytes, std::string& err);

    uint64_t bytes_sent() const { return sent_; }
    uint64_t bytes_received() const { return received_; }
    double wait_ms() const { return wait_ms_; }   // time spent in requests (the network and the worker's work)

private:
    bool send(uint16_t type, const std::vector<uint8_t>& meta, const void* body, size_t body_bytes,
              const core::rpc::IoDeadline& deadline, std::string& err);
    bool reply(uint16_t type, std::vector<uint8_t>& body, const core::rpc::IoDeadline& deadline, std::string& err,
               bool keep_on_refusal);
    bool call(uint16_t type, const std::vector<uint8_t>& meta, const void* body, size_t body_bytes,
              std::vector<uint8_t>& answer, std::string& err, std::chrono::milliseconds timeout,
              bool keep_on_refusal = false);
    void fail();

    core::rpc::Socket socket_ = core::rpc::kInvalidSocket;
    Spec spec_{};
    std::string endpoint_, token_;
    uint64_t op_ = 0, epoch_ = 0;
    bool prompt_open_ = false, refused_ = false;
    int64_t prompt_at_ = 0, prompt_end_ = 0;
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

}  // namespace strata::glm::stage
