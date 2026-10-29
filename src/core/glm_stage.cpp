// src/core/glm_stage.cpp - StrataRPCG: the GLM-5.3-Flash layer-stage protocol (see glm_stage.hpp).
#include "strata/core/glm_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace strata::glm::stage {
namespace rpc = core::rpc;

namespace {

constexpr size_t kHeaderBytes = 24;
constexpr uint64_t kMaxFrameBytes = 1ull << 31;
constexpr uint32_t kMaxMessageBytes = 4096;
constexpr std::chrono::milliseconds kHelloTimeout{30 * 1000};
constexpr std::chrono::milliseconds kCallTimeout{10 * 60 * 1000};
// a prompt chunk's answer waits for the worker's layers to read the chunk - long, so a long chunk on a slow disk is not
// cut off
constexpr std::chrono::milliseconds kPromptTimeout{120 * 60 * 1000};

enum Type : uint16_t {
    kHello = 1, kReset = 2, kSnapSave = 3, kSnapRestore = 4, kCkptSave = 5, kCkptRestore = 6, kStep = 7,
    kPromptBegin = 8, kRows = 9, kPromptEnd = 10, kSample = 11, kBias = 12, kLogits = 13, kSave = 14, kLoad = 15
};
constexpr uint16_t kReply = 0x8000;

void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); }
void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t) (v >> (8 * i)); }
void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (v >> (8 * i)); }
uint16_t get16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
uint32_t get32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (8 * i);
    return v;
}
uint64_t get64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}

struct Writer {
    std::vector<uint8_t> b;
    void u32(uint32_t v) { const size_t n = b.size(); b.resize(n + 4); put32(b.data() + n, v); }
    void u64(uint64_t v) { const size_t n = b.size(); b.resize(n + 8); put64(b.data() + n, v); }
    void i32(int32_t v) { u32((uint32_t) v); }
    void i64(int64_t v) { u64((uint64_t) v); }
    void f32(float v) { uint32_t x; std::memcpy(&x, &v, 4); u32(x); }
    void raw(const void* p, size_t n) {
        if (n == 0) return;
        const size_t at = b.size();
        b.resize(at + n);
        std::memcpy(b.data() + at, p, n);
    }
};

struct Reader {
    const uint8_t* p = nullptr;
    size_t n = 0, at = 0;
    bool u32(uint32_t& v) { if (n - at < 4) return false; v = get32(p + at); at += 4; return true; }
    bool u64(uint64_t& v) { if (n - at < 8) return false; v = get64(p + at); at += 8; return true; }
    bool i32(int32_t& v) { uint32_t x; if (!u32(x)) return false; v = (int32_t) x; return true; }
    bool i64(int64_t& v) { uint64_t x; if (!u64(x)) return false; v = (int64_t) x; return true; }
    bool f32(float& v) { uint32_t x; if (!u32(x)) return false; std::memcpy(&v, &x, 4); return true; }
    bool raw(void* d, size_t k) { if (n - at < k) return false; if (k) std::memcpy(d, p + at, k); at += k; return true; }
    size_t left() const { return n - at; }
    const uint8_t* here() const { return p + at; }
};

void write_spec(Writer& w, const Spec& s) {
    w.u64(s.identity_a); w.u64(s.identity_b); w.u32(s.n_layers); w.u32(s.split); w.u32(s.dim); w.u32(s.hc);
    w.u32(s.vocab); w.u64(s.max_context); w.u32((uint32_t) s.wire);
    w.u32(s.head_on_coordinator ? 1u : 0u); w.u32(s.sessions ? 1u : 0u);
}
bool read_spec(Reader& r, Spec& s) {
    uint32_t wire = 0, head = 0, sessions = 0;
    if (!r.u64(s.identity_a) || !r.u64(s.identity_b) || !r.u32(s.n_layers) || !r.u32(s.split) || !r.u32(s.dim) ||
        !r.u32(s.hc) || !r.u32(s.vocab) || !r.u64(s.max_context) || !r.u32(wire) ||
        wire > (uint32_t) LayerStageWireFormat::fp16 || !r.u32(head) || head > 1 || !r.u32(sessions) || sessions > 1)
        return false;
    s.wire = (LayerStageWireFormat) wire;
    s.head_on_coordinator = head != 0;
    s.sessions = sessions != 0;
    return true;
}

bool send_frame(rpc::Socket s, uint16_t type, uint64_t op, const void* a, size_t an, const void* b, size_t bn,
                const rpc::IoDeadline& dl, std::string& err) {
    if ((uint64_t) an + (uint64_t) bn > kMaxFrameBytes) {
        err = "layer-stage frame over the protocol's limit";
        return false;
    }
    uint8_t h[kHeaderBytes];
    std::memcpy(h, "GSTG", 4);
    put16(h + 4, kProtocol);
    put16(h + 6, type);
    put64(h + 8, op);
    put64(h + 16, (uint64_t) an + (uint64_t) bn);
    return rpc::send_all(s, h, kHeaderBytes, dl, err) && (an == 0 || rpc::send_all(s, a, an, dl, err)) &&
           (bn == 0 || rpc::send_all(s, b, bn, dl, err));
}

bool parse_header(const uint8_t* h, uint16_t& type, uint64_t& op, uint64_t& bytes, std::string& err) {
    if (std::memcmp(h, "GSTG", 4) != 0) {
        err = "not a StrataRPCG GLM layer-stage peer (wrong magic)";
        return false;
    }
    if (get16(h + 4) != kProtocol) {
        err = "the other PC runs another layer-stage protocol version (" + std::to_string(get16(h + 4)) + ", this " +
              std::to_string(kProtocol) + "): update both PCs to the same release";
        return false;
    }
    type = get16(h + 6);
    op = get64(h + 8);
    bytes = get64(h + 16);
    if (bytes > kMaxFrameBytes) {
        err = "layer-stage frame over the protocol's limit";
        return false;
    }
    return true;
}

// a whole frame by an absolute deadline (the coordinator's answers)
bool recv_frame(rpc::Socket s, const rpc::IoDeadline& dl, uint16_t& type, uint64_t& op, std::vector<uint8_t>& payload,
                std::string& err) {
    uint8_t h[kHeaderBytes];
    uint64_t n = 0;
    if (!rpc::recv_all(s, h, kHeaderBytes, dl, err) || !parse_header(h, type, op, n, err)) return false;
    payload.resize((size_t) n);
    return n == 0 || rpc::recv_all(s, payload.data(), payload.size(), dl, err);
}

// a whole frame that may be long in coming (a request on an idle connection, the next chunk of a prompt): its first
// byte waited for as long as it takes (idle_ms: how long), then `timeout` for the rest
bool recv_frame_idle(rpc::Socket s, std::chrono::milliseconds timeout, uint16_t& type, uint64_t& op,
                     std::vector<uint8_t>& payload, std::string& err, double* idle_ms = nullptr) {
    uint8_t h[kHeaderBytes];
    uint64_t n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (!rpc::recv_all(s, h, 1, rpc::wait_forever(), err)) return false;
    if (idle_ms != nullptr)
        *idle_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const rpc::IoDeadline dl = rpc::deadline_after(timeout);
    if (!rpc::recv_all(s, h + 1, kHeaderBytes - 1, dl, err) || !parse_header(h, type, op, n, err)) return false;
    payload.resize((size_t) n);
    return n == 0 || rpc::recv_all(s, payload.data(), payload.size(), dl, err);
}

std::vector<uint8_t> answer(bool ok, const std::string& msg, const std::vector<uint8_t>& body = {}) {
    Writer w;
    const size_t n = std::min<size_t>(msg.size(), kMaxMessageBytes);
    w.u32(ok ? 0u : 1u);
    w.u32((uint32_t) n);
    w.raw(msg.data(), n);
    w.raw(body.data(), body.size());
    return std::move(w.b);
}

// an answer's status: ok (body: what follows), or the worker's error in err
bool parse_answer(const std::vector<uint8_t>& payload, std::vector<uint8_t>& body, std::string& err) {
    Reader r{payload.data(), payload.size(), 0};
    uint32_t status = 0, n = 0;
    if (!r.u32(status) || !r.u32(n) || n > kMaxMessageBytes || n > r.left() || status > 1) {
        err = "a malformed layer-stage answer";
        return false;
    }
    std::string msg((const char*) r.here(), n);
    r.at += n;
    if (status != 0) {
        err = msg.empty() ? "the layer-stage worker refused the request" : "the layer-stage worker: " + msg;
        return false;
    }
    body.assign(r.here(), r.here() + r.left());
    return true;
}

bool env_on(const char* k) {
    const char* v = std::getenv(k);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool finite_all(const float* v, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

}  // namespace

bool same_spec(const Spec& a, const Spec& b, std::string& why) {
    why.clear();
    if (a.identity_a != b.identity_a || a.identity_b != b.identity_b)
        why = "the two PCs load different models (their packs or model files differ)";
    else if (a.n_layers != b.n_layers || a.dim != b.dim || a.hc != b.hc || a.vocab != b.vocab)
        why = "the model geometry differs";
    else if (a.split != b.split)
        why = "--layer-stage-split differs (coordinator " + std::to_string(a.split) + ", worker " +
              std::to_string(b.split) + ")";
    else if (a.max_context != b.max_context)
        why = "the context differs (coordinator " + std::to_string(a.max_context) + ", worker " +
              std::to_string(b.max_context) + ")";
    else if (a.wire != b.wire)
        why = std::string("--layer-stage-wire differs (coordinator ") + wire_name(a.wire) + ", worker " +
              wire_name(b.wire) + ")";
    else if (a.head_on_coordinator != b.head_on_coordinator)
        why = "--layer-stage-head differs (set it the same on both PCs)";
    else if (a.sessions != b.sessions)
        why = "--layer-stage-sessions must match on both PCs";
    return why.empty();
}

bool parse_wire(const std::string& s, LayerStageWireFormat& f) {
    if (s == "fp32") f = LayerStageWireFormat::fp32;
    else if (s == "bf16") f = LayerStageWireFormat::bf16;
    else if (s == "fp16") f = LayerStageWireFormat::fp16;
    else return false;
    return true;
}

const char* wire_name(LayerStageWireFormat f) {
    return f == LayerStageWireFormat::bf16 ? "bf16" : f == LayerStageWireFormat::fp16 ? "fp16" : "fp32";
}

void hash_text(const std::string& s, uint64_t& a, uint64_t& b) {
    a = 1469598103934665603ull;
    b = 0x9e3779b97f4a7c15ull;
    for (unsigned char c : s) {
        a = (a ^ c) * 1099511628211ull;
        b ^= c + 0x517cc1b727220a95ull + (b << 6) + (b >> 2);
        b = (b ^ (b >> 29)) * 0xbf58476d1ce4e5b9ull;
    }
    b ^= (uint64_t) s.size() + 0x94d049bb133111ebull;
    b = (b ^ (b >> 30)) * 0xbf58476d1ce4e5b9ull;
    b = (b ^ (b >> 27)) * 0x94d049bb133111ebull;
    b ^= b >> 31;
}

bool check_split(int n_layers, int split, std::string& why) {
    if (n_layers < 2 || split < 1 || split >= n_layers) {
        why = "the split must leave layers on both PCs (1 to " + std::to_string(std::max(1, n_layers - 1)) + ")";
        return false;
    }
    why.clear();
    return true;
}

bool valid_snapshot(const SnapshotRef& ref, uint64_t context) {
    return ref.id.size() == 32 && ref.id.find_first_not_of("0123456789abcdef") == std::string::npos &&
           ref.position > 0 && (uint64_t) ref.position <= context;
}

std::string new_snapshot_id() {
    std::random_device rd;
    static const char* hex = "0123456789abcdef";
    std::string id(32, '0');
    for (int i = 0; i < 32; i += 8) {
        uint32_t v = rd();
        for (int k = 0; k < 8; ++k) id[(size_t) (i + k)] = hex[(v >> (4 * k)) & 15];
    }
    return id;
}

uint64_t prefix_hash(const std::vector<int32_t>& tokens) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : tokens)
        for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) ((uint32_t) t >> (8 * b))) * 1099511628211ull;
    return h;
}

// ------------------------------------------------------------------ the coordinator's end

Client::~Client() { close(); }

void Client::close() {
    if (socket_ != rpc::kInvalidSocket) rpc::close(socket_);
    socket_ = rpc::kInvalidSocket;
    prompt_open_ = false;
}

void Client::fail() { close(); }

bool Client::connect(const std::string& endpoint, const std::string& token, const Spec& spec, std::string& err) {
    endpoint_ = endpoint;
    token_ = token;
    spec_ = spec;
    return reconnect(err);
}

bool Client::reconnect(std::string& err) {
    close();
    refused_ = token_.empty();
    if (refused_) {
        err = "set STRATA_RPC_TOKEN (the same secret on both PCs) before starting a layer-stage split";
        return false;
    }
    rpc::Socket s = rpc::kInvalidSocket;
    if (!rpc::connect(endpoint_, s, rpc::deadline_after(std::chrono::seconds(10)), err)) return false;
    std::string e2;
    rpc::set_nodelay(s, true, e2);
    socket_ = s;
    Writer w;
    w.u32((uint32_t) token_.size());
    w.raw(token_.data(), token_.size());
    write_spec(w, spec_);
    std::vector<uint8_t> body;
    if (!call(kHello, w.b, nullptr, 0, body, err, kHelloTimeout)) {
        err = "layer-stage handshake with " + endpoint_ + " failed: " + err;
        refused_ = true;
        close();
        return false;
    }
    ++epoch_;
    return true;
}

bool Client::send(uint16_t type, const std::vector<uint8_t>& meta, const void* body, size_t body_bytes,
                  const rpc::IoDeadline& dl, std::string& err) {
    if (!connected()) {
        err = "the layer-stage worker is not connected";
        return false;
    }
    if (!send_frame(socket_, type, ++op_, meta.data(), meta.size(), body, body_bytes, dl, err)) {
        err = "layer-stage send failed: " + err;
        fail();
        return false;
    }
    sent_ += kHeaderBytes + meta.size() + body_bytes;
    return true;
}

// the answer to request `type` (the last one sent: op_), its payload in body
bool Client::reply(uint16_t type, std::vector<uint8_t>& body, const rpc::IoDeadline& dl, std::string& err,
                   bool keep_on_refusal) {
    uint16_t t = 0;
    uint64_t op = 0;
    std::vector<uint8_t> payload;
    if (!recv_frame(socket_, dl, t, op, payload, err)) {
        err = "layer-stage receive failed: " + err;
        fail();
        return false;
    }
    received_ += kHeaderBytes + payload.size();
    if (t != (uint16_t) (type | kReply) || op != op_) {
        err = "layer-stage answer out of order";
        fail();
        return false;
    }
    if (!parse_answer(payload, body, err)) {
        // (the worker closes after a refused request: both ends start over - but a refused disk session leaves the
        // connection and both ends' state as they were)
        if (!keep_on_refusal || err == "a malformed layer-stage answer") fail();
        return false;
    }
    return true;
}

bool Client::call(uint16_t type, const std::vector<uint8_t>& meta, const void* body, size_t body_bytes,
                  std::vector<uint8_t>& answer_body, std::string& err, std::chrono::milliseconds timeout,
                  bool keep_on_refusal) {
    if (prompt_open_ && type != kRows && type != kPromptEnd) {
        err = "layer-stage request while a prompt is streaming";
        fail();
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const rpc::IoDeadline dl = rpc::deadline_after(timeout);
    const bool ok = send(type, meta, body, body_bytes, dl, err) && reply(type, answer_body, dl, err, keep_on_refusal);
    wait_ms_ += ms_since(t0);
    return ok;
}

bool Client::reset(std::string& err) {
    std::vector<uint8_t> body;
    return call(kReset, {}, nullptr, 0, body, err, kCallTimeout);
}

bool Client::snap_save(int64_t pos, bool& saved, std::string& err) {
    Writer w;
    w.i64(pos);
    std::vector<uint8_t> body;
    if (!call(kSnapSave, w.b, nullptr, 0, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    uint32_t ok = 0;
    if (!r.u32(ok) || r.left() != 0) {
        err = "a malformed layer-stage snapshot answer";
        fail();
        return false;
    }
    saved = ok != 0;
    return true;
}

bool Client::snap_restore(int64_t want_pos, bool& restored, std::string& err) {
    Writer w;
    w.i64(want_pos);
    std::vector<uint8_t> body;
    if (!call(kSnapRestore, w.b, nullptr, 0, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    uint32_t ok = 0;
    int64_t pos = 0;
    if (!r.u32(ok) || !r.i64(pos) || r.left() != 0 || (ok != 0 && pos != want_pos)) {
        err = "a malformed layer-stage snapshot restore answer";
        fail();
        return false;
    }
    restored = ok != 0;
    return true;
}

bool Client::ckpt_save(int slot, bool from_snapshot, int64_t pos, bool& saved, std::string& err) {
    Writer w;
    w.i32(slot);
    w.u32(from_snapshot ? 1u : 0u);
    w.i64(pos);
    std::vector<uint8_t> body;
    if (!call(kCkptSave, w.b, nullptr, 0, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    uint32_t ok = 0;
    if (!r.u32(ok) || r.left() != 0) {
        err = "a malformed layer-stage checkpoint answer";
        fail();
        return false;
    }
    saved = ok != 0;
    return true;
}

bool Client::ckpt_restore(int slot, int64_t want_pos, bool& restored, std::string& err) {
    Writer w;
    w.i32(slot);
    w.i64(want_pos);
    std::vector<uint8_t> body;
    if (!call(kCkptRestore, w.b, nullptr, 0, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    uint32_t ok = 0;
    int64_t pos = 0;
    if (!r.u32(ok) || !r.i64(pos) || r.left() != 0 || (ok != 0 && pos != want_pos)) {
        err = "a malformed layer-stage checkpoint restore answer";
        fail();
        return false;
    }
    restored = ok != 0;
    return true;
}

bool Client::step(int64_t p, const float* row, int& token, float* r_out, std::string& err) {
    const size_t hf = spec_.h_floats();
    if (!core::layer_stage_encode_activations(spec_.wire, row, hf, wire_, err)) {
        err = "a row to the worker: " + err;   // (a non-finite value: the coordinator's layers went wrong)
        return false;
    }
    Writer w;
    w.i64(p);
    std::vector<uint8_t> body;
    if (!call(kStep, w.b, wire_.data(), wire_.size(), body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    if (spec_.head_on_coordinator) {
        if (r_out == nullptr || r.left() != hf * 4) {
            err = "a malformed layer-stage step answer";
            fail();
            return false;
        }
        std::memcpy(r_out, r.here(), hf * 4);
        if (!finite_all(r_out, hf)) {
            err = "the worker's last layer gave a non-finite value";
            return false;
        }
        token = -1;
        return true;
    }
    int32_t t = 0;
    if (!r.i32(t) || r.left() != 0 || t < 0 || (uint32_t) t >= spec_.vocab) {
        err = "a malformed layer-stage step answer";
        fail();
        return false;
    }
    token = t;
    return true;
}

bool Client::prompt_begin(int64_t p0, int64_t n, int T, int& t_bound, std::string& err) {
    if (prompt_open_) {
        err = "a layer-stage prompt while another streams";
        fail();
        return false;
    }
    if (n < 1 || n > kMaxPromptTokens || T < 1 || T > kMaxChunk || p0 < 0 || (uint64_t) (p0 + n) > spec_.max_context) {
        err = "a layer-stage prompt of " + std::to_string(n) + " tokens in chunks of " + std::to_string(T);
        return false;
    }
    Writer w;
    w.i64(p0);
    w.i64(n);
    w.i32(T);
    std::vector<uint8_t> body;
    if (!call(kPromptBegin, w.b, nullptr, 0, body, err, kPromptTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    int32_t tb = 0;
    if (!r.i32(tb) || r.left() != 0 || tb < 1 || tb > T) {
        err = "a malformed layer-stage prompt answer";
        fail();
        return false;
    }
    t_bound = tb;
    prompt_open_ = true;
    prompt_at_ = p0;
    prompt_end_ = p0 + n;
    return true;
}

bool Client::prompt_chunk(int64_t p0, int T, const float* rows, std::string& err) {
    if (!prompt_open_) {
        err = "layer-stage rows outside a prompt";
        return false;
    }
    if (p0 != prompt_at_ || T < 1 || p0 + T > prompt_end_) {
        err = "layer-stage rows out of order";
        fail();
        return false;
    }
    const size_t hf = spec_.h_floats();
    for (int r0 = 0; r0 < T; r0 += kPieceRows) {
        const int nr = std::min(kPieceRows, T - r0);
        if (!core::layer_stage_encode_activations(spec_.wire, rows + (size_t) r0 * hf, (size_t) nr * hf, wire_, err)) {
            err = "a row to the worker: " + err;
            fail();
            return false;
        }
        Writer w;   // the chunk (its start and rows), then this piece's place in it
        w.i64(p0);
        w.u32((uint32_t) T);
        w.u32((uint32_t) r0);
        w.u32((uint32_t) nr);
        std::vector<uint8_t> body;
        if (!call(kRows, w.b, wire_.data(), wire_.size(), body, err, kPromptTimeout)) return false;
        if (!body.empty()) {
            err = "a malformed layer-stage rows answer";
            fail();
            return false;
        }
    }
    prompt_at_ = p0 + T;
    return true;
}

bool Client::prompt_end(bool completed, int64_t& pos_after, std::string& err) {
    if (!prompt_open_) {
        err = "no layer-stage prompt to end (the link failed while it streamed)";
        return false;
    }
    if (completed && prompt_at_ != prompt_end_) {
        err = "the layer-stage prompt ended with " + std::to_string(prompt_end_ - prompt_at_) + " rows unsent";
        fail();
        return false;
    }
    Writer w;
    w.u32(completed ? 1u : 0u);
    std::vector<uint8_t> body;
    const bool ok = call(kPromptEnd, w.b, nullptr, 0, body, err, kPromptTimeout);
    prompt_open_ = false;
    if (!ok) return false;
    Reader r{body.data(), body.size(), 0};
    if (!r.i64(pos_after) || r.left() != 0 || (completed && pos_after != prompt_end_)) {
        err = "the layer-stage worker's position after a prompt disagrees";
        fail();
        return false;
    }
    return true;
}

bool Client::sample(const Sampling& s, int& token, std::string& err) {
    if (s.history.size() > (size_t) kMaxHistory) {
        err = "a penalty history longer than the protocol takes";
        return false;
    }
    Writer w;
    w.i32(s.top_k);
    w.f32(s.top_p);
    w.f32(s.min_p);
    w.f32(s.temperature);
    w.i32(s.min_keep);
    w.i32(s.penalty_last_n);
    w.f32(s.penalty_repeat);
    w.f32(s.penalty_freq);
    w.f32(s.penalty_present);
    w.u64(s.seed);
    w.u64(s.counter);
    w.u32(s.greedy ? 1u : 0u);
    w.i32(s.bias_slot);
    w.u32((uint32_t) s.history.size());
    w.raw(s.history.data(), s.history.size() * 4);
    std::vector<uint8_t> body;
    if (!call(kSample, w.b, nullptr, 0, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    int32_t t = 0;
    if (!r.i32(t) || r.left() != 0 || t < 0 || (uint32_t) t >= spec_.vocab) {
        err = "a malformed layer-stage sample answer";
        fail();
        return false;
    }
    token = t;
    return true;
}

bool Client::bias(int slot, const std::vector<std::pair<int32_t, float>>& b, std::string& err) {
    if (b.size() > kMaxBias || slot < 0) {
        err = "a logit-bias table the protocol does not take";
        return false;
    }
    Writer w;
    w.i32(slot);
    w.u32((uint32_t) b.size());
    for (const auto& [id, v] : b) {
        w.i32(id);
        w.f32(v);
    }
    std::vector<uint8_t> body;
    return call(kBias, w.b, nullptr, 0, body, err, kCallTimeout);
}

bool Client::logits(std::vector<float>& out, std::string& err) {
    std::vector<uint8_t> body;
    if (!call(kLogits, {}, nullptr, 0, body, err, kCallTimeout)) return false;
    if (body.size() != (size_t) spec_.vocab * 4) {
        err = "a malformed layer-stage logits answer";
        fail();
        return false;
    }
    out.resize(spec_.vocab);
    std::memcpy(out.data(), body.data(), body.size());
    return true;
}

bool Client::session(bool save, const SnapshotRef& ref, uint64_t& bytes, std::string& err) {
    if (!spec_.sessions || !valid_snapshot(ref, spec_.max_context)) {
        err = "invalid or disabled paired disk session";
        return false;
    }
    Writer w;
    w.raw(ref.id.data(), 32);
    w.i64(ref.position);
    w.u64(ref.prefix);
    std::vector<uint8_t> body;
    if (!call(save ? kSave : kLoad, w.b, nullptr, 0, body, err, kCallTimeout, true)) return false;
    Reader r{body.data(), body.size(), 0};
    if (!r.u64(bytes) || r.left() != 0) {
        err = "a malformed layer-stage session answer";
        fail();
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ the worker's end

Server::Server(std::string token, Spec spec, Backend& backend)
    : token_(std::move(token)), spec_(spec), backend_(backend) {}

bool Server::serve(rpc::Socket s, std::string& err) {
    // the prompt the coordinator streams: its span, the chunk the backend takes, the rows of the chunk being received
    struct Prompt {
        bool open = false;
        int64_t p0 = 0, n = 0, next = 0;   // next: the first position of the chunk not yet read
        int t_bound = 0;
        int64_t chunk_p0 = -1;
        int chunk_T = 0, chunk_have = 0;
        double rx_ms = 0, idle_ms = 0, work_ms = 0;
        uint64_t rx_bytes = 0;
        std::chrono::steady_clock::time_point t0;
    } pr;
    struct Close {   // (every way out: the connection closed, an open prompt's memory given back, position 0)
        rpc::Socket s;
        Backend& b;
        Prompt& pr;
        ~Close() {
            rpc::close(s);
            if (pr.open) {
                std::string e;
                b.prompt_end(false, e);
            }
            b.reset();
        }
    } close_{s, backend_, pr};
    std::string e2;
    rpc::set_nodelay(s, true, e2);
    const bool stats = env_on("STRATA_LAYER_STAGE_STATS");
    const size_t hf = spec_.h_floats();
    const size_t elem = core::layer_stage_wire_element_bytes(spec_.wire);
    const size_t row_bytes = hf * elem;   // a row on the wire
    bool hello = false;
    std::vector<uint8_t> payload;
    std::vector<float> row(hf), r_out(hf);
    // decode-step statistics (-Stats): the steps' wall time on this end, and the time between them (the coordinator)
    uint64_t steps = 0;
    double step_ms = 0, gap_ms = 0;
    for (;;) {
        uint16_t type = 0;
        uint64_t op = 0;
        double idle = 0;
        // the hello comes within kHelloTimeout of the connect; between requests a connection may stay idle
        if (!hello) {
            uint8_t h[kHeaderBytes];
            uint64_t n = 0;
            const rpc::IoDeadline dl = rpc::deadline_after(kHelloTimeout);
            if (!rpc::recv_all(s, h, kHeaderBytes, dl, err) || !parse_header(h, type, op, n, err)) return false;
            payload.resize((size_t) n);
            if (n > 0 && !rpc::recv_all(s, payload.data(), payload.size(), dl, err)) return false;
        } else {
            const auto r0 = std::chrono::steady_clock::now();
            if (!recv_frame_idle(s, pr.open ? kPromptTimeout : kCallTimeout, type, op, payload, err, &idle)) {
                if (err == "RPC connection closed") err = "the coordinator closed the connection";
                return false;
            }
            if (pr.open && type == kRows) {
                pr.idle_ms += idle;
                pr.rx_ms += ms_since(r0) - idle;
                pr.rx_bytes += kHeaderBytes + payload.size();
            }
            if (type == kStep) gap_ms += idle;
        }
        Reader r{payload.data(), payload.size(), 0};
        const auto respond = [&](bool ok, const std::string& msg, const std::vector<uint8_t>& body = {}) {
            const std::vector<uint8_t> a = answer(ok, msg, body);
            std::string se;
            return send_frame(s, (uint16_t) (type | kReply), op, a.data(), a.size(), nullptr, 0,
                              rpc::deadline_after(kCallTimeout), se);
        };
        // a refused request: its answer, then the connection closes (the coordinator connects again and resets)
        const auto refuse = [&](const std::string& msg) {
            respond(false, msg);
            err = msg;
            return false;
        };
        if (!hello) {
            uint32_t tn = 0;
            if (type != kHello || !r.u32(tn) || tn > 4096 || tn > r.left()) {
                err = "a connection that did not start with a layer-stage hello";
                return false;
            }
            const std::string tok((const char*) r.here(), tn);
            r.at += tn;
            Spec peer;
            if (!read_spec(r, peer) || r.left() != 0) return refuse("a malformed hello");
            // (the secret compared in constant time: a wrong one takes as long to refuse whatever it shares)
            unsigned diff = token_.empty() || tok.size() != token_.size() ? 1u : 0u;
            for (size_t i = 0; i < tok.size() && i < token_.size(); ++i) diff |= (unsigned) (uint8_t) (tok[i] ^ token_[i]);
            if (diff != 0) return refuse("wrong STRATA_RPC_TOKEN (set the same secret on both PCs)");
            std::string why;
            if (!same_spec(peer, spec_, why)) return refuse(why);
            hello = true;
            backend_.reset();
            if (!respond(true, "")) {
                err = "the hello's answer could not be sent";
                return false;
            }
            std::fprintf(stderr, "glm stage: coordinator connected (split %u, wire %s, head on the %s)\n", spec_.split,
                         wire_name(spec_.wire), spec_.head_on_coordinator ? "coordinator" : "worker");
            continue;
        }
        if (pr.open && type != kRows && type != kPromptEnd)
            return refuse("a request while a prompt streams (expected its rows or its end)");
        switch (type) {
            case kReset: {
                backend_.reset();
                if (!respond(true, "")) return false;
                break;
            }
            case kSnapSave: {
                int64_t pos = 0;
                if (!r.i64(pos) || r.left() != 0) return refuse("a malformed snapshot");
                if (pos != backend_.pos())
                    return refuse("a snapshot at " + std::to_string(pos) + ", the worker is at " +
                                  std::to_string(backend_.pos()));
                std::string se;
                const bool ok = backend_.snap_save(se);
                if (!ok && !se.empty()) std::fprintf(stderr, "glm stage: snapshot not kept: %s\n", se.c_str());
                Writer w;
                w.u32(ok ? 1u : 0u);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kSnapRestore: {
                int64_t want = 0;
                if (!r.i64(want) || r.left() != 0) return refuse("a malformed snapshot restore");
                bool ok = false;
                int64_t pos = backend_.pos();
                if (backend_.snap_pos() >= 0 && backend_.snap_pos() == want) {
                    std::string se;
                    ok = backend_.snap_restore(pos, se);
                    if (!ok) std::fprintf(stderr, "glm stage: snapshot not restored: %s\n", se.c_str());
                }
                Writer w;
                w.u32(ok ? 1u : 0u);
                w.i64(ok ? pos : backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kCkptSave: {
                int32_t slot = 0;
                uint32_t from_snap = 0;
                int64_t pos = 0;
                if (!r.i32(slot) || !r.u32(from_snap) || !r.i64(pos) || r.left() != 0 || slot < 0 || slot > 4096 ||
                    from_snap > 1)
                    return refuse("a malformed checkpoint");
                const int64_t here = from_snap ? backend_.snap_pos() : backend_.pos();
                if (pos != here)
                    return refuse("a checkpoint at " + std::to_string(pos) + ", the worker's is at " + std::to_string(here));
                std::string se;
                const bool ok = backend_.ckpt_save(slot, from_snap != 0, se);
                if (!ok && !se.empty()) std::fprintf(stderr, "glm stage: checkpoint not kept: %s\n", se.c_str());
                Writer w;
                w.u32(ok ? 1u : 0u);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kCkptRestore: {
                int32_t slot = 0;
                int64_t want = 0;
                if (!r.i32(slot) || !r.i64(want) || r.left() != 0 || slot < 0 || slot > 4096)
                    return refuse("a malformed checkpoint restore");
                bool ok = false;
                int64_t pos = backend_.pos();
                if (backend_.ckpt_pos(slot) == want && want >= 0) {
                    std::string se;
                    ok = backend_.ckpt_restore(slot, pos, se);
                    if (!ok) std::fprintf(stderr, "glm stage: checkpoint not restored: %s\n", se.c_str());
                }
                Writer w;
                w.u32(ok ? 1u : 0u);
                w.i64(ok ? pos : backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kStep: {   // p, then its row
                int64_t p = 0;
                if (!r.i64(p) || r.left() != row_bytes) return refuse("a malformed step");
                if (p != backend_.pos())
                    return refuse("a step at " + std::to_string(p) + ", the worker is at " + std::to_string(backend_.pos()));
                std::string de;
                if (!core::layer_stage_decode_activations(spec_.wire, r.here(), row_bytes, row.data(), hf, de))
                    return refuse("a step's row: " + de);
                const auto t0 = std::chrono::steady_clock::now();
                int token = -1;
                std::string se;
                if (!backend_.step(p, row.data(), token, spec_.head_on_coordinator ? r_out.data() : nullptr, se))
                    return refuse("a step failed: " + se);
                Writer w;
                if (spec_.head_on_coordinator) {
                    w.raw(r_out.data(), hf * 4);
                } else {
                    if (token < 0 || (uint32_t) token >= spec_.vocab) return refuse("a step picked no token");
                    w.i32(token);
                }
                step_ms += ms_since(t0);
                if (!respond(true, "", w.b)) return false;
                if (stats && ++steps % 256 == 0) {
                    std::fprintf(stderr, "glm stage: last 256 steps %.1f ms a token on this PC, %.1f between them\n",
                                 step_ms / 256.0, gap_ms / 256.0);
                    step_ms = gap_ms = 0;
                }
                break;
            }
            case kPromptBegin: {
                int64_t p0 = 0, n = 0;
                int32_t T = 0;
                if (!r.i64(p0) || !r.i64(n) || !r.i32(T) || r.left() != 0 || n < 1 || n > kMaxPromptTokens || T < 1 ||
                    T > kMaxChunk || p0 < 0 || (uint64_t) (p0 + n) > spec_.max_context)
                    return refuse("a malformed prompt");
                if (p0 != backend_.pos())
                    return refuse("a prompt at " + std::to_string(p0) + ", the worker is at " +
                                  std::to_string(backend_.pos()));
                int tb = 0;
                std::string se;
                if (!backend_.prompt_begin(p0, n, T, tb, se)) return refuse("the prompt could not start: " + se);
                if (tb < 1 || tb > T || backend_.prompt_rows() == nullptr) {
                    std::string e3;
                    backend_.prompt_end(false, e3);
                    return refuse("the worker's prompt path gave no chunk");
                }
                pr = Prompt{};
                pr.open = true;
                pr.p0 = p0;
                pr.n = n;
                pr.next = p0;
                pr.t_bound = tb;
                pr.t0 = std::chrono::steady_clock::now();
                Writer w;
                w.i32(tb);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kRows: {   // the chunk (p0, T), this piece (r0, nr), its rows
                if (!pr.open) return refuse("rows outside a prompt");
                int64_t cp0 = 0;
                uint32_t cT = 0, r0 = 0, nr = 0;
                if (!r.i64(cp0) || !r.u32(cT) || !r.u32(r0) || !r.u32(nr)) return refuse("malformed rows");
                const bool new_chunk = pr.chunk_p0 < 0;
                if (cT < 1 || (int) cT > pr.t_bound || cp0 != (new_chunk ? pr.next : pr.chunk_p0) ||
                    (!new_chunk && (int) cT != pr.chunk_T) || cp0 + (int64_t) cT > pr.p0 + pr.n ||
                    (int) r0 != (new_chunk ? 0 : pr.chunk_have) || nr < 1 || nr > (uint32_t) kPieceRows ||
                    r0 + nr > cT || r.left() != (size_t) nr * row_bytes)
                    return refuse("rows out of order or of the wrong size");
                if (new_chunk) {
                    pr.chunk_p0 = cp0;
                    pr.chunk_T = (int) cT;
                    pr.chunk_have = 0;
                }
                std::string de;
                if (!core::layer_stage_decode_activations(spec_.wire, r.here(), (size_t) nr * row_bytes,
                                                          backend_.prompt_rows() + (size_t) r0 * hf, (size_t) nr * hf, de))
                    return refuse("the prompt's rows: " + de);
                pr.chunk_have += (int) nr;
                if (pr.chunk_have == pr.chunk_T) {   // the chunk is here: the worker's layers read it, then the answer
                    const auto t0 = std::chrono::steady_clock::now();
                    std::string se;
                    if (!backend_.prompt_chunk(pr.chunk_p0, pr.chunk_T, se)) return refuse("a prompt chunk failed: " + se);
                    pr.work_ms += ms_since(t0);
                    pr.next = pr.chunk_p0 + pr.chunk_T;
                    pr.chunk_p0 = -1;
                }
                if (!respond(true, "")) return false;
                break;
            }
            case kPromptEnd: {
                if (!pr.open) return refuse("no prompt to end");
                uint32_t completed = 0;
                if (!r.u32(completed) || completed > 1 || r.left() != 0) return refuse("a malformed prompt end");
                if (completed && (pr.chunk_p0 >= 0 || pr.next != pr.p0 + pr.n))
                    return refuse("the prompt ended before its last chunk");
                pr.open = false;
                std::string se;
                const bool ok = backend_.prompt_end(completed != 0, se);
                if (stats && completed) {
                    const double all = ms_since(pr.t0);
                    std::fprintf(stderr,
                                 "glm stage: prompt of %lld tokens at %lld in %.0f ms: %.0f MB of rows in %.0f ms (%.0f "
                                 "MB/s), %.0f ms waiting for the coordinator, %.0f ms the worker's own\n",
                                 (long long) pr.n, (long long) pr.p0, all, (double) pr.rx_bytes / 1e6, pr.rx_ms,
                                 pr.rx_ms > 0 ? (double) pr.rx_bytes / 1e3 / pr.rx_ms : 0.0, pr.idle_ms, pr.work_ms);
                }
                if (!ok) return refuse("the prompt's end failed: " + se);
                if (!completed) {   // stopped early: both ends go back (the coordinator resets or restores next)
                    Writer w;
                    w.i64(backend_.pos());
                    if (!respond(true, "", w.b)) return false;
                    break;
                }
                if (backend_.pos() != pr.p0 + pr.n) return refuse("the worker's feed did not take every row");
                Writer w;
                w.i64(backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kSample: {
                Sampling sp;
                uint32_t greedy = 0, nh = 0;
                if (!r.i32(sp.top_k) || !r.f32(sp.top_p) || !r.f32(sp.min_p) || !r.f32(sp.temperature) ||
                    !r.i32(sp.min_keep) || !r.i32(sp.penalty_last_n) || !r.f32(sp.penalty_repeat) ||
                    !r.f32(sp.penalty_freq) || !r.f32(sp.penalty_present) || !r.u64(sp.seed) || !r.u64(sp.counter) ||
                    !r.u32(greedy) || greedy > 1 || !r.i32(sp.bias_slot) || !r.u32(nh) || nh > (uint32_t) kMaxHistory ||
                    r.left() != (size_t) nh * 4 || !std::isfinite(sp.temperature) || !std::isfinite(sp.top_p) ||
                    !std::isfinite(sp.min_p) || !std::isfinite(sp.penalty_repeat) || !std::isfinite(sp.penalty_freq) ||
                    !std::isfinite(sp.penalty_present))
                    return refuse("a malformed sample");
                sp.greedy = greedy != 0;
                sp.history.resize(nh);
                r.raw(sp.history.data(), (size_t) nh * 4);
                std::string se;
                const int tok = backend_.sample(sp, se);
                if (tok < 0 || (uint32_t) tok >= spec_.vocab) return refuse("sampling failed: " + se);
                Writer w;
                w.i32(tok);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kBias: {
                int32_t slot = 0;
                uint32_t n = 0;
                if (!r.i32(slot) || slot < 0 || slot > 64 || !r.u32(n) || n > kMaxBias || r.left() != (size_t) n * 8)
                    return refuse("a malformed logit bias");
                std::vector<std::pair<int32_t, float>> b(n);
                for (auto& [id, v] : b) {
                    r.i32(id);
                    r.f32(v);
                    if (id < 0 || (uint32_t) id >= spec_.vocab || std::isnan(v)) return refuse("a logit bias out of range");
                }
                std::string se;
                if (!backend_.bias(slot, b, se)) return refuse("the logit bias: " + se);
                if (!respond(true, "")) return false;
                break;
            }
            case kLogits: {
                std::vector<float> lg;
                std::string se;
                if (!backend_.logits(lg, se) || lg.size() != spec_.vocab) return refuse("no logits: " + se);
                Writer w;
                w.raw(lg.data(), lg.size() * 4);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kSave:
            case kLoad: {
                SnapshotRef ref;
                ref.id.resize(32);
                if (!spec_.sessions || !r.raw(&ref.id[0], 32) || !r.i64(ref.position) || !r.u64(ref.prefix) ||
                    r.left() != 0 || !valid_snapshot(ref, spec_.max_context))
                    return refuse("invalid or disabled paired disk session");
                uint64_t bytes = 0;
                std::string se;
                // (a session that fails is the request's error: the connection stays, both ends keep their state -
                // the coordinator resets both after a failed restore)
                const bool ok = backend_.session(type == kSave, ref, bytes, se);
                if (!ok) {
                    if (!respond(false, "disk session: " + se)) return false;
                    break;
                }
                Writer w;
                w.u64(bytes);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            default:
                return refuse("an unknown request " + std::to_string(type));
        }
    }
}

bool Server::serve_forever(const std::string& endpoint, std::string& err) {
    rpc::Socket ls = rpc::kInvalidSocket;
    if (!rpc::listen(endpoint, ls, err)) return false;
    std::fprintf(stderr, "glm stage: worker listening on %s (layers %u-%u%s)\n", endpoint.c_str(), spec_.split,
                 spec_.n_layers - 1, spec_.head_on_coordinator ? "; the head on the coordinator" : " and the head");
    for (;;) {
        rpc::Socket c = rpc::kInvalidSocket;
        std::string e;
        if (!rpc::accept(ls, c, e)) {
            std::fprintf(stderr, "glm stage: %s\n", e.c_str());
            continue;
        }
        if (!serve(c, e)) std::fprintf(stderr, "glm stage: connection ended: %s\n", e.c_str());
    }
}

}  // namespace strata::glm::stage
