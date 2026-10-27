// src/dsv41/ds4_stage.cpp - StrataRPC: the DeepSeek layer-stage protocol (see ds4_stage.hpp).
#include "ds4_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::ds4::stage {
namespace rpc = core::rpc;

namespace {

constexpr size_t kHeaderBytes = 24;
constexpr uint64_t kMaxFrameBytes = 1ull << 31;
constexpr uint32_t kMaxMessageBytes = 4096;
constexpr uint32_t kRowsHalt = 1;
constexpr std::chrono::milliseconds kHelloTimeout{30 * 1000};
constexpr std::chrono::milliseconds kCallTimeout{10 * 60 * 1000};
// a prompt's waits: for the worker to take the next rows (its GPU may still be on the chunk before) and for its answer
// (its layers through the whole prompt) - long, so a long prompt on a slow disk is not cut off
constexpr std::chrono::milliseconds kPromptTimeout{120 * 60 * 1000};

enum Type : uint16_t {
    kHello = 1, kReset = 2, kCheckpoint = 3, kRestore = 4, kPin = 5, kWindow = 6, kPrompt = 7, kRows = 8, kSample = 9,
    kLogits = 10
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
}
bool read_spec(Reader& r, Spec& s) {
    uint32_t wire = 0;
    if (!r.u64(s.identity_a) || !r.u64(s.identity_b) || !r.u32(s.n_layers) || !r.u32(s.split) || !r.u32(s.dim) ||
        !r.u32(s.hc) || !r.u32(s.vocab) || !r.u64(s.max_context) || !r.u32(wire) ||
        wire > (uint32_t) LayerStageWireFormat::fp16)
        return false;
    s.wire = (LayerStageWireFormat) wire;
    return true;
}

bool send_frame(rpc::Socket s, uint16_t type, uint64_t op, const void* a, size_t an, const void* b, size_t bn,
                const rpc::IoDeadline& dl, std::string& err) {
    if ((uint64_t) an + (uint64_t) bn > kMaxFrameBytes) {
        err = "layer-stage frame over the protocol's limit";
        return false;
    }
    uint8_t h[kHeaderBytes];
    std::memcpy(h, "DSTG", 4);
    put16(h + 4, kProtocol);
    put16(h + 6, type);
    put64(h + 8, op);
    put64(h + 16, (uint64_t) an + (uint64_t) bn);
    return rpc::send_all(s, h, kHeaderBytes, dl, err) && (an == 0 || rpc::send_all(s, a, an, dl, err)) &&
           (bn == 0 || rpc::send_all(s, b, bn, dl, err));
}

bool parse_header(const uint8_t* h, uint16_t& type, uint64_t& op, uint64_t& bytes, std::string& err) {
    if (std::memcmp(h, "DSTG", 4) != 0) {
        err = "not a StrataRPC DeepSeek layer-stage peer (wrong magic)";
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

// a whole frame that may be long in coming (a request on an idle connection, the next rows of a prompt): its first byte
// waited for as long as it takes, then `timeout` for the rest
bool recv_frame_idle(rpc::Socket s, std::chrono::milliseconds timeout, uint16_t& type, uint64_t& op,
                     std::vector<uint8_t>& payload, std::string& err) {
    uint8_t h[kHeaderBytes];
    uint64_t n = 0;
    if (!rpc::recv_all(s, h, 1, rpc::wait_forever(), err)) return false;
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

bool tokens_valid(const int32_t* t, size_t n, uint32_t vocab) {
    for (size_t i = 0; i < n; ++i)
        if (t[i] < 0 || (uint32_t) t[i] >= vocab) return false;
    return true;
}

bool env_on(const char* k) {
    const char* v = std::getenv(k);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool same_spec(const Spec& a, const Spec& b, std::string& why) {
    why.clear();
    if (a.identity_a != b.identity_a || a.identity_b != b.identity_b)
        why = "the two PCs load different models (their packs' geometry or tensors differ) or --experts differs";
    else if (a.n_layers != b.n_layers || a.dim != b.dim || a.hc != b.hc || a.vocab != b.vocab)
        why = "the model geometry differs";
    else if (a.split != b.split)
        why = "--layer-stage-split differs (coordinator " + std::to_string(a.split) + ", worker " +
              std::to_string(b.split) + ")";
    else if (a.max_context != b.max_context)
        why = "--ctx differs (coordinator " + std::to_string(a.max_context) + ", worker " +
              std::to_string(b.max_context) + ")";
    else if (a.wire != b.wire)
        why = std::string("--layer-stage-wire differs (coordinator ") + wire_name(a.wire) + ", worker " +
              wire_name(b.wire) + ")";
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

// ------------------------------------------------------------------ the coordinator's end

Client::~Client() { close(); }

void Client::close() {
    if (socket_ != rpc::kInvalidSocket) rpc::close(socket_);
    socket_ = rpc::kInvalidSocket;
    prompt_open_ = false;
}

void Client::fail() { close(); }

bool Client::connect(const std::string& endpoint, const std::string& token, const Spec& spec, std::string& err) {
    close();
    spec_ = spec;
    if (token.empty()) {
        err = "set STRATA_RPC_TOKEN (the same secret on both PCs) before starting a layer-stage split";
        return false;
    }
    rpc::Socket s = rpc::kInvalidSocket;
    if (!rpc::connect(endpoint, s, rpc::deadline_after(std::chrono::seconds(10)), err)) return false;
    std::string e2;
    rpc::set_nodelay(s, true, e2);
    socket_ = s;
    Writer w;
    w.u32((uint32_t) token.size());
    w.raw(token.data(), token.size());
    write_spec(w, spec);
    std::vector<uint8_t> body;
    if (!call(kHello, w.b, body, err, kHelloTimeout)) {
        err = "layer-stage handshake with " + endpoint + " failed: " + err;
        close();
        return false;
    }
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
bool Client::reply(uint16_t type, std::vector<uint8_t>& body, const rpc::IoDeadline& dl, std::string& err) {
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
        fail();   // (the worker closes after a refused request: both ends start over)
        return false;
    }
    return true;
}

bool Client::call(uint16_t type, const std::vector<uint8_t>& meta, std::vector<uint8_t>& body, std::string& err,
                  std::chrono::milliseconds timeout) {
    if (prompt_open_) {
        err = "layer-stage request while a prompt is streaming";
        fail();
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const rpc::IoDeadline dl = rpc::deadline_after(timeout);
    const bool ok = send(type, meta, nullptr, 0, dl, err) && reply(type, body, dl, err);
    wait_ms_ += ms_since(t0);
    return ok;
}

bool Client::reset(std::string& err) {
    std::vector<uint8_t> body;
    return call(kReset, {}, body, err, kCallTimeout);
}

bool Client::checkpoint(int slot, int64_t pos, std::string& err) {
    Writer w;
    w.i32(slot);
    w.i64(pos);
    std::vector<uint8_t> body;
    return call(kCheckpoint, w.b, body, err, kCallTimeout);
}

bool Client::restore(int64_t limit, bool& restored, int64_t& pos, std::string& err) {
    Writer w;
    w.i64(limit);
    std::vector<uint8_t> body;
    if (!call(kRestore, w.b, body, err, kCallTimeout)) return false;
    Reader r{body.data(), body.size(), 0};
    uint32_t ok = 0;
    if (!r.u32(ok) || !r.i64(pos) || r.left() != 0) {
        err = "a malformed layer-stage restore answer";
        fail();
        return false;
    }
    restored = ok != 0;
    return true;
}

bool Client::pin(std::string& err) {
    std::vector<uint8_t> body;
    return call(kPin, {}, body, err, kCallTimeout);
}

// a ROWS frame's meta (p0, T, flags) and the rows' bytes into wire_: h in the wire format, then pre as FP32
bool Client::rows_frame(int64_t p0, int T, const float* h, const float* pre, uint32_t flags, std::vector<uint8_t>& meta,
                        std::string& err) {
    Writer w;
    w.i64(p0);
    w.u32((uint32_t) T);
    w.u32(flags);
    meta = std::move(w.b);
    wire_.clear();
    if (T <= 0) return true;
    if (!core::layer_stage_encode_activations(spec_.wire, h, (size_t) T * spec_.h_floats(), wire_, err)) {
        err = "a row to the worker: " + err;   // (a non-finite value: the coordinator's layers went wrong)
        return false;
    }
    for (size_t i = 0; i < (size_t) T * kPreFloats; ++i)
        if (!std::isfinite(pre[i])) {
            err = "a row to the worker has a non-finite pre value";
            return false;
        }
    const size_t at = wire_.size();
    wire_.resize(at + (size_t) T * kPreFloats * 4);
    std::memcpy(wire_.data() + at, pre, (size_t) T * kPreFloats * 4);
    return true;
}

bool Client::window(int64_t p0, const int32_t* tokens, int n, int logits_rows, const float* h, const float* pre,
                    int64_t& pos_after, std::string& err) {
    if (prompt_open_) {
        err = "layer-stage window while a prompt is streaming";
        fail();
        return false;
    }
    if (n < 1 || n > kPieceRows) {
        err = "a layer-stage window of " + std::to_string(n) + " tokens";
        return false;
    }
    std::vector<uint8_t> rows_meta;
    if (!rows_frame(p0, n, h, pre, 0, rows_meta, err)) return false;
    Writer w;   // p0, n, logits_rows, the tokens - then the rows (wire_)
    w.i64(p0);
    w.u32((uint32_t) n);
    w.i32(logits_rows);
    w.raw(tokens, (size_t) n * 4);
    const auto t0 = std::chrono::steady_clock::now();
    const rpc::IoDeadline dl = rpc::deadline_after(kCallTimeout);
    std::vector<uint8_t> body;
    const bool ok = send(kWindow, w.b, wire_.data(), wire_.size(), dl, err) && reply(kWindow, body, dl, err);
    wait_ms_ += ms_since(t0);
    if (!ok) return false;
    Reader r{body.data(), body.size(), 0};
    if (!r.i64(pos_after) || r.left() != 0 || pos_after != p0 + n) {
        err = "the layer-stage worker's position after a window disagrees";
        fail();
        return false;
    }
    return true;
}

bool Client::prompt_begin(int64_t p0, const int32_t* tokens, int n, int logits_rows, int snap_slot, std::string& err) {
    if (prompt_open_) {
        err = "a layer-stage prompt while another streams";
        fail();
        return false;
    }
    if (n < 1 || n > kMaxTokens) {
        err = "a layer-stage prompt of " + std::to_string(n) + " tokens";
        return false;
    }
    Writer w;
    w.i64(p0);
    w.u32((uint32_t) n);
    w.i32(logits_rows);
    w.i32(snap_slot);
    w.raw(tokens, (size_t) n * 4);
    if (!send(kPrompt, w.b, nullptr, 0, rpc::deadline_after(kCallTimeout), err)) return false;
    prompt_op_ = op_;
    prompt_sent_ = p0;
    prompt_end_ = p0 + n;
    prompt_open_ = true;
    return true;
}

bool Client::prompt_rows(int64_t p0, int T, const float* h, const float* pre, std::string& err) {
    if (!prompt_open_) {
        err = "layer-stage rows outside a prompt";
        return false;
    }
    if (p0 != prompt_sent_ || T < 0 || p0 + T > prompt_end_) {
        err = "layer-stage rows out of order";
        fail();
        return false;
    }
    const size_t hf = spec_.h_floats();
    for (int r0 = 0; r0 < T; r0 += kPieceRows) {
        const int nr = std::min(kPieceRows, T - r0);
        std::vector<uint8_t> meta;
        if (!rows_frame(p0 + r0, nr, h + (size_t) r0 * hf, pre + (size_t) r0 * kPreFloats, 0, meta, err)) {
            fail();
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const rpc::IoDeadline dl = rpc::deadline_after(kPromptTimeout);
        if (!send(kRows, meta, wire_.data(), wire_.size(), dl, err)) return false;
        // its acknowledgement - or the prompt's answer, when the worker ended the prompt before taking these rows
        uint16_t t = 0;
        uint64_t op = 0;
        std::vector<uint8_t> payload, body;
        if (!recv_frame(socket_, dl, t, op, payload, err)) {
            err = "layer-stage receive failed: " + err;
            fail();
            return false;
        }
        received_ += kHeaderBytes + payload.size();
        wait_ms_ += ms_since(t0);
        if (t == (uint16_t) (kPrompt | kReply) && op == prompt_op_) {
            err = parse_answer(payload, body, err) ? "the layer-stage worker ended the prompt before taking its rows" : err;
            fail();
            return false;
        }
        if (t != (uint16_t) (kRows | kReply) || op != op_) {
            err = "layer-stage answer out of order";
            fail();
            return false;
        }
        if (!parse_answer(payload, body, err)) {
            fail();
            return false;
        }
        prompt_sent_ += nr;
    }
    return true;
}

bool Client::prompt_end(bool halt, int64_t& pos_after, std::string& err) {
    if (!prompt_open_) {
        err = "no layer-stage prompt to end (the link failed while it streamed)";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const rpc::IoDeadline dl = rpc::deadline_after(kPromptTimeout);
    if (prompt_sent_ < prompt_end_) {
        if (!halt) {
            err = "the layer-stage prompt ended with " + std::to_string(prompt_end_ - prompt_sent_) + " rows unsent";
            fail();
            return false;
        }
        std::vector<uint8_t> meta;
        rows_frame(prompt_sent_, 0, nullptr, nullptr, kRowsHalt, meta, err);
        if (!send(kRows, meta, nullptr, 0, dl, err)) return false;   // (not acknowledged: the answer follows)
    }
    prompt_open_ = false;
    uint16_t t = 0;
    uint64_t op = 0;
    std::vector<uint8_t> payload, body;
    if (!recv_frame(socket_, dl, t, op, payload, err)) {
        err = "layer-stage receive failed: " + err;
        fail();
        return false;
    }
    received_ += kHeaderBytes + payload.size();
    wait_ms_ += ms_since(t0);
    if (t != (uint16_t) (kPrompt | kReply) || op != prompt_op_) {
        err = "layer-stage answer out of order";
        fail();
        return false;
    }
    if (!parse_answer(payload, body, err)) {
        if (!halt) fail();   // (a stopped prompt's refusal is the expected answer: the connection stays)
        return false;
    }
    Reader r{body.data(), body.size(), 0};
    if (!r.i64(pos_after) || r.left() != 0 || pos_after != prompt_end_) {
        err = "the layer-stage worker's position after a prompt disagrees";
        fail();
        return false;
    }
    return true;
}

bool Client::sample(const Sampling& s, int& token, std::string& err) {
    Writer w;
    w.f32(s.temperature);
    w.f32(s.top_p);
    w.i32(s.top_k);
    w.f32(s.min_p);
    w.u64(s.seed);
    w.u64(s.counter);
    std::vector<uint8_t> body;
    if (!call(kSample, w.b, body, err, kCallTimeout)) return false;
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

bool Client::logits(std::vector<float>& out, std::string& err) {
    std::vector<uint8_t> body;
    if (!call(kLogits, {}, body, err, kCallTimeout)) return false;
    if (body.size() % 4 != 0) {
        err = "a malformed layer-stage logits answer";
        fail();
        return false;
    }
    out.resize(body.size() / 4);
    if (!body.empty()) std::memcpy(out.data(), body.data(), body.size());
    return true;
}

// ------------------------------------------------------------------ the worker's end

Server::Server(std::string token, Spec spec, Backend& backend)
    : token_(std::move(token)), spec_(spec), backend_(backend) {}

bool Server::serve(rpc::Socket s, std::string& err) {
    struct Close {   // (every way out: the connection closed, the model back at position 0)
        rpc::Socket s;
        Backend& b;
        ~Close() {
            rpc::close(s);
            b.reset();
        }
    } close_{s, backend_};
    std::string e2;
    rpc::set_nodelay(s, true, e2);
    const bool stats = env_on("STRATA_LAYER_STAGE_STATS");
    const size_t hf = spec_.h_floats();
    const size_t elem = core::layer_stage_wire_element_bytes(spec_.wire);
    const size_t row_bytes = hf * elem + (size_t) kPreFloats * 4;   // a row on the wire
    bool hello = false;
    std::vector<uint8_t> payload;
    for (;;) {
        uint16_t type = 0;
        uint64_t op = 0;
        // the hello comes within kHelloTimeout of the connect; between requests a connection may stay idle
        if (!hello) {
            uint8_t h[kHeaderBytes];
            uint64_t n = 0;
            const rpc::IoDeadline dl = rpc::deadline_after(kHelloTimeout);
            if (!rpc::recv_all(s, h, kHeaderBytes, dl, err) || !parse_header(h, type, op, n, err)) return false;
            payload.resize((size_t) n);
            if (n > 0 && !rpc::recv_all(s, payload.data(), payload.size(), dl, err)) return false;
        } else if (!recv_frame_idle(s, kCallTimeout, type, op, payload, err)) {
            if (err == "RPC connection closed") err = "the coordinator closed the connection";
            return false;
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
            std::fprintf(stderr, "ds4 stage: coordinator connected (split %u, wire %s)\n", spec_.split,
                         wire_name(spec_.wire));
            continue;
        }
        switch (type) {
            case kReset: {
                backend_.reset();
                if (!respond(true, "")) return false;
                break;
            }
            case kCheckpoint: {
                int32_t slot = 0;
                int64_t pos = 0;
                if (!r.i32(slot) || !r.i64(pos) || r.left() != 0) return refuse("a malformed checkpoint");
                if (pos != backend_.pos())
                    return refuse("a checkpoint at " + std::to_string(pos) + ", the worker is at " +
                                  std::to_string(backend_.pos()));
                backend_.checkpoint(slot);
                if (!respond(true, "")) return false;
                break;
            }
            case kRestore: {
                int64_t limit = 0;
                if (!r.i64(limit) || r.left() != 0) return refuse("a malformed restore");
                const bool ok = backend_.restore(limit);
                Writer w;
                w.u32(ok ? 1u : 0u);
                w.i64(backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kPin: {
                backend_.pin();
                if (!respond(true, "")) return false;
                break;
            }
            case kWindow: {   // p0, n, logits_rows, n tokens, then n rows
                int64_t p0 = 0;
                uint32_t nt = 0;
                int32_t lr = 0;
                if (!r.i64(p0) || !r.u32(nt) || !r.i32(lr) || nt < 1 || nt > (uint32_t) kPieceRows ||
                    r.left() != (size_t) nt * 4 + (size_t) nt * row_bytes)
                    return refuse("a malformed window");
                std::vector<int32_t> toks(nt);
                r.raw(toks.data(), (size_t) nt * 4);
                if (!tokens_valid(toks.data(), nt, spec_.vocab)) return refuse("a window token out of range");
                if (p0 != backend_.pos())
                    return refuse("a window at " + std::to_string(p0) + ", the worker is at " +
                                  std::to_string(backend_.pos()));
                std::vector<float> h((size_t) nt * hf), pre((size_t) nt * kPreFloats);
                std::string de;
                if (!core::layer_stage_decode_activations(spec_.wire, r.here(), (size_t) nt * hf * elem, h.data(),
                                                          h.size(), de))
                    return refuse("a window's rows: " + de);
                std::memcpy(pre.data(), r.here() + (size_t) nt * hf * elem, pre.size() * 4);
                for (float v : pre)
                    if (!std::isfinite(v)) return refuse("a window's rows: a non-finite pre value");
                bool given = false;
                const RowFn rows = [&](int64_t q0, int T, float* hd, float* pd, std::string& e) {
                    if (given || q0 != p0 || T != (int) nt) {
                        e = "the worker asked for rows the window does not hold";
                        return false;
                    }
                    given = true;
                    std::memcpy(hd, h.data(), h.size() * 4);
                    std::memcpy(pd, pre.data(), pre.size() * 4);
                    return true;
                };
                std::string fe;
                if (!backend_.feed(toks.data(), (int) nt, lr, -1, rows, fe)) return refuse("a window failed: " + fe);
                if (!given) return refuse("the worker's feed did not take the window's rows");
                Writer w;
                w.i64(backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kPrompt: {   // p0, n, logits_rows, snap_slot, n tokens; the rows follow in ROWS frames
                int64_t p0 = 0;
                uint32_t nt = 0;
                int32_t lr = 0, snap = -1;
                if (!r.i64(p0) || !r.u32(nt) || !r.i32(lr) || !r.i32(snap) || nt < 1 || nt > (uint32_t) kMaxTokens ||
                    r.left() != (size_t) nt * 4)
                    return refuse("a malformed prompt");
                std::vector<int32_t> toks(nt);
                r.raw(toks.data(), (size_t) nt * 4);
                if (!tokens_valid(toks.data(), nt, spec_.vocab)) return refuse("a prompt token out of range");
                if (p0 != backend_.pos())
                    return refuse("a prompt at " + std::to_string(p0) + ", the worker is at " +
                                  std::to_string(backend_.pos()));
                // the rows as the worker's first layer asks for them: frames read as needed (each acknowledged), a
                // frame's rows past the request kept for the next one
                std::vector<float> fh, fpre;
                size_t f_at = 0, f_n = 0;   // the queue's rows: [f_at, f_at + f_n) of fh / fpre
                int64_t f_p0 = p0;          // the position of the queue's first row
                bool halted = false, broken = false;
                std::vector<uint8_t> fp;
                const RowFn rows = [&](int64_t q0, int T, float* hd, float* pd, std::string& e) {
                    if (halted || broken) {
                        e = halted ? "the coordinator stopped the prompt" : "the link to the coordinator failed";
                        return false;
                    }
                    if (q0 != f_p0 || T < 0 || q0 + T > p0 + (int64_t) nt) {
                        e = "the worker asked for rows out of order";
                        broken = true;
                        return false;
                    }
                    while ((int64_t) f_n < T) {
                        uint16_t t2 = 0;
                        uint64_t op2 = 0;
                        std::string ie;
                        if (!recv_frame_idle(s, kPromptTimeout, t2, op2, fp, ie) || t2 != kRows) {
                            broken = true;
                            e = ie.empty() ? "expected the prompt's rows" : "the prompt's rows: " + ie;
                            return false;
                        }
                        Reader q{fp.data(), fp.size(), 0};
                        int64_t rp0 = 0;
                        uint32_t rT = 0, flags = 0;
                        if (!q.i64(rp0) || !q.u32(rT) || !q.u32(flags)) {
                            broken = true;
                            e = "malformed rows";
                            return false;
                        }
                        if (flags & kRowsHalt) {   // (the coordinator stopped: not acknowledged, the answer follows)
                            halted = true;
                            e = "the coordinator stopped the prompt";
                            return false;
                        }
                        if (rT < 1 || rT > (uint32_t) kPieceRows || rp0 != f_p0 + (int64_t) f_n ||
                            rp0 + (int64_t) rT > p0 + (int64_t) nt || q.left() != (size_t) rT * row_bytes) {
                            broken = true;
                            e = "rows out of order or of the wrong size";
                            return false;
                        }
                        if (f_at > 0) {   // (the queue's taken front dropped before it grows)
                            fh.erase(fh.begin(), fh.begin() + (std::ptrdiff_t) (f_at * hf));
                            fpre.erase(fpre.begin(), fpre.begin() + (std::ptrdiff_t) (f_at * kPreFloats));
                            f_at = 0;
                        }
                        fh.resize((f_n + rT) * hf);
                        fpre.resize((f_n + rT) * kPreFloats);
                        std::string de;
                        if (!core::layer_stage_decode_activations(spec_.wire, q.here(), (size_t) rT * hf * elem,
                                                                  fh.data() + f_n * hf, (size_t) rT * hf, de)) {
                            broken = true;
                            e = "the prompt's rows: " + de;
                            return false;
                        }
                        std::memcpy(fpre.data() + f_n * kPreFloats, q.here() + (size_t) rT * hf * elem,
                                    (size_t) rT * kPreFloats * 4);
                        for (size_t i = 0; i < (size_t) rT * kPreFloats; ++i)
                            if (!std::isfinite(fpre[f_n * kPreFloats + i])) {
                                broken = true;
                                e = "the prompt's rows: a non-finite pre value";
                                return false;
                            }
                        f_n += rT;
                        const std::vector<uint8_t> a = answer(true, "");
                        if (!send_frame(s, (uint16_t) (kRows | kReply), op2, a.data(), a.size(), nullptr, 0,
                                        rpc::deadline_after(kCallTimeout), ie)) {
                            broken = true;
                            e = "the rows' acknowledgement: " + ie;
                            return false;
                        }
                    }
                    std::memcpy(hd, fh.data() + f_at * hf, (size_t) T * hf * 4);
                    std::memcpy(pd, fpre.data() + f_at * kPreFloats, (size_t) T * kPreFloats * 4);
                    f_at += (size_t) T;
                    f_n -= (size_t) T;
                    f_p0 += T;
                    return true;
                };
                const auto t0 = std::chrono::steady_clock::now();
                std::string fe;
                const bool ok = backend_.feed(toks.data(), (int) nt, lr, snap, rows, fe);
                if (broken) {   // (the link is gone or out of step: nothing more to say on it)
                    err = "a prompt's rows failed: " + fe;
                    return false;
                }
                if (halted) {   // the coordinator stopped: the connection stays, both ends go back to a checkpoint
                    if (!respond(false, "stopped")) return false;
                    break;
                }
                if (!ok) return refuse("a prompt failed: " + fe);
                if (f_p0 != p0 + (int64_t) nt) return refuse("the worker's feed did not take every row");
                if (stats)
                    std::fprintf(stderr, "ds4 stage: prompt of %u tokens at %lld in %.0f ms\n", nt, (long long) p0,
                                 ms_since(t0));
                Writer w;
                w.i64(backend_.pos());
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kSample: {
                Sampling sp;
                if (!r.f32(sp.temperature) || !r.f32(sp.top_p) || !r.i32(sp.top_k) || !r.f32(sp.min_p) ||
                    !r.u64(sp.seed) || !r.u64(sp.counter) || r.left() != 0 || !std::isfinite(sp.temperature) ||
                    !std::isfinite(sp.top_p) || !std::isfinite(sp.min_p))
                    return refuse("a malformed sample");
                const int tok = backend_.sample(sp);
                if (tok < 0 || (uint32_t) tok >= spec_.vocab) return refuse("sampling failed");
                Writer w;
                w.i32(tok);
                if (!respond(true, "", w.b)) return false;
                break;
            }
            case kLogits: {
                std::vector<float> lg;
                if (!backend_.logits(lg)) return refuse("no logits");
                Writer w;
                w.raw(lg.data(), lg.size() * 4);
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
    std::fprintf(stderr, "ds4 stage: worker listening on %s (layers %u-%u and the head)\n", endpoint.c_str(), spec_.split,
                 spec_.n_layers - 1);
    for (;;) {
        rpc::Socket c = rpc::kInvalidSocket;
        std::string e;
        if (!rpc::accept(ls, c, e)) {
            std::fprintf(stderr, "ds4 stage: %s\n", e.c_str());
            continue;
        }
        if (!serve(c, e)) std::fprintf(stderr, "ds4 stage: connection ended: %s\n", e.c_str());
    }
}

}  // namespace strata::ds4::stage
