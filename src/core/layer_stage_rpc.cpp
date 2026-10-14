#include "strata/core/layer_stage_rpc.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

namespace strata::core {
namespace {
constexpr uint16_t kVersion = 1;
constexpr size_t kHeaderBytes = 24;
constexpr uint64_t kMaxFrameBytes = 1ull << 31;
constexpr uint32_t kMaxErrorBytes = 4096;
constexpr auto kHandshakeTimeout = std::chrono::minutes(10);
constexpr auto kOperationTimeout = std::chrono::minutes(5);

enum Message : uint16_t { kHello = 1, kReset = 2, kPrefill = 3, kVerify = 4, kCommit = 5, kAbort = 6, kHealth = 7 };

void put16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}
void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (i * 8));
}
void put64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (i * 8));
}
uint16_t get16(const uint8_t* p) { return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8); }
uint32_t get32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)p[i] << (i * 8);
    return v;
}
uint64_t get64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (i * 8);
    return v;
}

uint32_t native_endian_marker() {
    const uint32_t one = 1;
    return *(const uint8_t*)&one == 1 ? 0x01020304u : 0x04030201u;
}

uint32_t native_float_marker() {
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559) return 0;
    const float one = 1.0f;
    uint32_t bits = 0;
    std::memcpy(&bits, &one, sizeof(bits));
    return bits;
}

struct Writer {
    std::vector<uint8_t> bytes;
    void u32(uint32_t v) { const size_t n = bytes.size(); bytes.resize(n + 4); put32(bytes.data() + n, v); }
    void u64(uint64_t v) { const size_t n = bytes.size(); bytes.resize(n + 8); put64(bytes.data() + n, v); }
    void i64(int64_t v) { u64((uint64_t)v); }
    void raw(const void* p, size_t n) {
        if (n == 0) return;
        const size_t at = bytes.size();
        bytes.resize(at + n);
        std::memcpy(bytes.data() + at, p, n);
    }
};

struct Reader {
    const uint8_t* p = nullptr;
    size_t n = 0, at = 0;
    bool u32(uint32_t& v) {
        if (n - at < 4) return false;
        v = get32(p + at); at += 4; return true;
    }
    bool u64(uint64_t& v) {
        if (n - at < 8) return false;
        v = get64(p + at); at += 8; return true;
    }
    bool i64(int64_t& v) { uint64_t x = 0; if (!u64(x)) return false; v = (int64_t)x; return true; }
    bool raw(void* dst, size_t bytes) {
        if (bytes > n - at) return false;
        if (bytes != 0) std::memcpy(dst, p + at, bytes);
        at += bytes; return true;
    }
    const uint8_t* remaining() const { return p + at; }
    size_t left() const { return n - at; }
};

bool checked_mul(uint64_t a, uint64_t b, uint64_t& result) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool activation_float_count(const LayerStageSpec& s, uint32_t rows, bool prefill, uint64_t& floats) {
    uint64_t per_row = 0;
    if (prefill) {
        if (!checked_mul(s.n_embd, s.hc, per_row)) return false;
    } else {
        uint64_t residual = 0;
        if (!checked_mul(s.n_embd, s.hc, residual) ||
            residual > std::numeric_limits<uint64_t>::max() - s.n_embd ||
            residual + s.n_embd > std::numeric_limits<uint64_t>::max() - s.hc) return false;
        per_row = residual + s.n_embd + s.hc;
    }
    return checked_mul(rows, per_row, floats);
}

bool activation_frame_fits(const LayerStageSpec& s, uint32_t rows, bool prefill) {
    uint64_t floats = 0, float_bytes = 0, token_bytes = 0;
    const uint64_t token_size = prefill ? sizeof(int64_t) : sizeof(int32_t);
    constexpr uint64_t metadata_bytes = 36;
    return activation_float_count(s, rows, prefill, floats) && checked_mul(floats, sizeof(float), float_bytes) &&
           checked_mul(rows, token_size, token_bytes) && metadata_bytes + token_bytes <= kMaxFrameBytes &&
           float_bytes <= kMaxFrameBytes - metadata_bytes - token_bytes;
}

bool valid_spec(const LayerStageSpec& s) {
    return s.model_fingerprint != 0 && s.config_fingerprint != 0 && s.max_context > 0 &&
           s.max_context <= (uint64_t)std::numeric_limits<int64_t>::max() && s.n_layers > 0 &&
           s.n_embd > 0 && s.n_expert > 0 && s.hc > 0 && s.layer_begin < s.layer_end &&
           s.layer_end <= s.n_layers && s.max_verify_rows > 0 && s.max_prefill_rows > 0 &&
           activation_frame_fits(s, s.max_verify_rows, false) &&
           activation_frame_fits(s, s.max_prefill_rows, true);
}

uint64_t max_operation_payload(const LayerStageSpec& s) {
    uint64_t prefill_floats = 0, verify_floats = 0;
    if (!activation_float_count(s, s.max_prefill_rows, true, prefill_floats) ||
        !activation_float_count(s, s.max_verify_rows, false, verify_floats)) return 0;
    const uint64_t prefill_tokens = (uint64_t)s.max_prefill_rows * sizeof(int64_t);
    const uint64_t verify_tokens = (uint64_t)s.max_verify_rows * sizeof(int32_t);
    const uint64_t prefill = 36 + prefill_tokens + prefill_floats * sizeof(float);
    const uint64_t verify = 36 + verify_tokens + verify_floats * sizeof(float);
    return std::max<uint64_t>({prefill, verify, 24, 20, 16});
}

void write_spec(Writer& w, const LayerStageSpec& s) {
    w.u64(s.model_fingerprint); w.u64(s.config_fingerprint); w.u64(s.max_context);
    w.u32(s.n_layers); w.u32(s.n_embd); w.u32(s.n_expert); w.u32(s.hc);
    w.u32(s.layer_begin); w.u32(s.layer_end); w.u32(s.max_verify_rows); w.u32(s.max_prefill_rows);
    w.u32(s.capabilities);
}

bool read_spec(Reader& r, LayerStageSpec& s) {
    return r.u64(s.model_fingerprint) && r.u64(s.config_fingerprint) && r.u64(s.max_context) &&
           r.u32(s.n_layers) && r.u32(s.n_embd) && r.u32(s.n_expert) && r.u32(s.hc) &&
           r.u32(s.layer_begin) && r.u32(s.layer_end) && r.u32(s.max_verify_rows) &&
           r.u32(s.max_prefill_rows) && r.u32(s.capabilities);
}

bool same_spec(const LayerStageSpec& a, const LayerStageSpec& b) {
    return a.model_fingerprint == b.model_fingerprint && a.config_fingerprint == b.config_fingerprint &&
           a.max_context == b.max_context && a.n_layers == b.n_layers && a.n_embd == b.n_embd &&
           a.n_expert == b.n_expert && a.hc == b.hc && a.layer_begin == b.layer_begin &&
           a.layer_end == b.layer_end && a.max_verify_rows == b.max_verify_rows &&
           a.max_prefill_rows == b.max_prefill_rows && a.capabilities == b.capabilities;
}

bool make_header(uint16_t type, uint64_t operation, uint64_t payload_bytes, std::array<uint8_t, kHeaderBytes>& h,
                 std::string& err) {
    if (payload_bytes > kMaxFrameBytes) { err = "layer-stage frame exceeds the protocol limit"; return false; }
    std::memcpy(h.data(), "STGE", 4);
    put16(h.data() + 4, kVersion); put16(h.data() + 6, type);
    put64(h.data() + 8, operation); put64(h.data() + 16, payload_bytes);
    return true;
}

bool send_frame(rpc::Socket socket, uint16_t type, uint64_t operation, const void* a, size_t an,
                const void* b, size_t bn, const rpc::IoDeadline& deadline, std::string& err) {
    if (an > kMaxFrameBytes || bn > kMaxFrameBytes - an) {
        err = "layer-stage frame exceeds the protocol limit";
        return false;
    }
    std::array<uint8_t, kHeaderBytes> header{};
    if (!make_header(type, operation, (uint64_t)an + (uint64_t)bn, header, err) ||
        !rpc::send_all(socket, header.data(), header.size(), deadline, err) ||
        (an != 0 && !rpc::send_all(socket, a, an, deadline, err)) ||
        (bn != 0 && !rpc::send_all(socket, b, bn, deadline, err))) return false;
    return true;
}

bool receive_header(rpc::Socket socket, uint16_t& type, uint64_t& operation, uint64_t& payload_bytes,
                    const rpc::IoDeadline& deadline, std::string& err) {
    std::array<uint8_t, kHeaderBytes> h{};
    if (!rpc::recv_all(socket, h.data(), h.size(), deadline, err)) return false;
    if (std::memcmp(h.data(), "STGE", 4) != 0 || get16(h.data() + 4) != kVersion) {
        err = "layer-stage frame has invalid magic or version";
        return false;
    }
    type = get16(h.data() + 6); operation = get64(h.data() + 8); payload_bytes = get64(h.data() + 16);
    if (payload_bytes > kMaxFrameBytes) { err = "layer-stage frame exceeds the protocol limit"; return false; }
    return true;
}

bool receive_operation_header(rpc::Socket socket, uint16_t& type, uint64_t& operation, uint64_t& payload_bytes,
                              std::chrono::milliseconds timeout, rpc::IoDeadline& deadline, std::string& err) {
    std::array<uint8_t, kHeaderBytes> h{};
    // A connected authenticated worker may sit idle between requests. Only after its first byte arrives do we start
    // the bounded total deadline for this operation's remaining header, payload, execution, and reply.
    if (!rpc::recv_all(socket, h.data(), 1, rpc::wait_forever(), err)) return false;
    deadline = rpc::deadline_after(timeout);
    if (!rpc::recv_all(socket, h.data() + 1, h.size() - 1, deadline, err)) return false;
    if (std::memcmp(h.data(), "STGE", 4) != 0 || get16(h.data() + 4) != kVersion) {
        err = "layer-stage frame has invalid magic or version";
        return false;
    }
    type = get16(h.data() + 6); operation = get64(h.data() + 8); payload_bytes = get64(h.data() + 16);
    if (payload_bytes > kMaxFrameBytes) { err = "layer-stage frame exceeds the protocol limit"; return false; }
    return true;
}

struct RequestFingerprint {
    uint64_t bytes = 0;
    uint64_t a = 1469598103934665603ull;
    uint64_t b = 0x9e3779b97f4a7c15ull;
};

RequestFingerprint fingerprint_request(const uint8_t* data, size_t bytes) {
    RequestFingerprint f;
    f.bytes = (uint64_t)bytes;
    for (size_t i = 0; i < bytes; ++i) {
        const uint64_t v = data[i];
        f.a = (f.a ^ v) * 1099511628211ull;
        f.b ^= v + 0x517cc1b727220a95ull + (f.b << 6) + (f.b >> 2);
        f.b = (f.b ^ (f.b >> 29)) * 0xbf58476d1ce4e5b9ull;
    }
    f.b ^= f.bytes + 0x94d049bb133111ebull;
    f.b = (f.b ^ (f.b >> 30)) * 0xbf58476d1ce4e5b9ull;
    f.b = (f.b ^ (f.b >> 27)) * 0x94d049bb133111ebull;
    f.b ^= f.b >> 31;
    return f;
}

std::vector<uint8_t> response_payload(bool ok, const std::string& message, const void* body = nullptr,
                                      size_t body_bytes = 0) {
    const size_t n = std::min<size_t>(message.size(), kMaxErrorBytes);
    Writer w;
    w.u32(ok ? 0u : 1u);
    w.u32((uint32_t)n);
    w.raw(message.data(), n);
    w.raw(body, body_bytes);
    return std::move(w.bytes);
}

bool send_response(rpc::Socket socket, uint16_t type, uint64_t op, const std::vector<uint8_t>& payload,
                   const rpc::IoDeadline& deadline, std::string& err) {
    return send_frame(socket, (uint16_t)(0x8000u | type), op, payload.data(), payload.size(), nullptr, 0,
                      deadline, err);
}

enum class ParsedResponse { success, rejected, malformed };

ParsedResponse parse_response(Reader& r, void* response, size_t response_bytes, std::string& err) {
    uint32_t status = 0, message_bytes = 0;
    if (!r.u32(status) || !r.u32(message_bytes) || message_bytes > kMaxErrorBytes || message_bytes > r.left()) {
        err = "layer-stage response has an invalid status frame";
        return ParsedResponse::malformed;
    }
    if (status > 1) {
        err = "layer-stage response uses an undefined status code";
        return ParsedResponse::malformed;
    }
    std::string message((const char*)r.remaining(), message_bytes);
    r.at += message_bytes;
    if (status != 0) {
        if (r.left() != 0) {
            err = "layer-stage rejection response contains an unexpected body";
            return ParsedResponse::malformed;
        }
        err = message.empty() ? "layer-stage worker rejected the operation" : message;
        return ParsedResponse::rejected;
    }
    if (message_bytes != 0) {
        err = "layer-stage success response contains an unexpected message";
        return ParsedResponse::malformed;
    }
    if (r.left() != response_bytes || (response_bytes != 0 && response == nullptr)) {
        err = "layer-stage response has an unexpected payload size";
        return ParsedResponse::malformed;
    }
    if (!r.raw(response, response_bytes)) {
        err = "layer-stage success response body is truncated";
        return ParsedResponse::malformed;
    }
    return ParsedResponse::success;
}

bool checked_bytes(uint64_t n, size_t element, size_t& bytes) {
    if (element != 0 && n > (uint64_t)std::numeric_limits<size_t>::max() / element) return false;
    bytes = (size_t)n * element;
    return true;
}

} // namespace

LayerStageClient::~LayerStageClient() { close(); }

bool LayerStageClient::connect(const std::string& endpoint, const std::string& token, const LayerStageSpec& expected,
                               std::string& err) {
    close();
    if (!valid_spec(expected) || token.empty() || token.size() > kMaxErrorBytes || native_float_marker() == 0) {
        err = "layer-stage RPC needs a valid stage identity, IEEE-754 FP32 support, and a nonempty token";
        return false;
    }
    endpoint_ = endpoint;
    token_ = token;
    spec_ = expected;
    if (!rpc::connect(endpoint_, socket_, rpc::deadline_after(std::chrono::seconds(5), cancelled_), err)) return false;
    if (!rpc::set_nodelay(socket_, true, err)) { close(); return false; }
    Writer hello;
    write_spec(hello, spec_);
    hello.u32(native_endian_marker());
    hello.u32(native_float_marker());
    hello.u32((uint32_t)token.size());
    hello.raw(token.data(), token.size());
    const auto deadline = rpc::deadline_after(kHandshakeTimeout, cancelled_);
    if (!send_frame(socket_, kHello, 0, hello.bytes.data(), hello.bytes.size(), nullptr, 0, deadline, err)) {
        close(); return false;
    }
    std::array<uint8_t, 8> ack{};
    uint16_t type = 0;
    uint64_t op = 0, bytes = 0;
    if (!receive_header(socket_, type, op, bytes, deadline, err) || type != (0x8000u | kHello) || op != 0 || bytes != ack.size() ||
        !rpc::recv_all(socket_, ack.data(), ack.size(), deadline, err)) {
        if (err.empty()) err = "layer-stage worker returned an invalid handshake";
        close(); return false;
    }
    Reader reader{ack.data(), ack.size(), 0};
    if (parse_response(reader, nullptr, 0, err) != ParsedResponse::success) { close(); return false; }
    broken_ = false;
    next_operation_ = 1;
    bytes_sent_ = bytes_received_ = 0;
    return true;
}

void LayerStageClient::close() {
    if (socket_ != rpc::kInvalidSocket) rpc::close(socket_);
    socket_ = rpc::kInvalidSocket;
    broken_ = false;
    next_operation_ = 1;
}

bool LayerStageClient::request(uint16_t type, const uint8_t* metadata, size_t metadata_bytes, const void* body_a,
                               size_t body_a_bytes, const void* body_b, size_t body_b_bytes, void* response,
                               size_t response_bytes, std::string& err) {
    if (!connected()) { err = "layer-stage RPC is disconnected or was invalidated"; return false; }
    const bool health_req = type == kHealth;
    const uint64_t operation = health_req ? 0 : next_operation_;
    const auto deadline = rpc::deadline_after(health_req ? std::chrono::seconds(5) : kOperationTimeout, cancelled_);
    if (metadata_bytes > kMaxFrameBytes || body_a_bytes > kMaxFrameBytes - metadata_bytes ||
        body_b_bytes > kMaxFrameBytes - metadata_bytes - body_a_bytes) {
        err = "layer-stage frame exceeds the protocol limit";
        return false;
    }
    const uint64_t payload_bytes = (uint64_t)metadata_bytes + (uint64_t)body_a_bytes + (uint64_t)body_b_bytes;
    std::array<uint8_t, kHeaderBytes> header{};
    if (!make_header(type, operation, payload_bytes, header, err) ||
        !rpc::send_all(socket_, header.data(), header.size(), deadline, err) ||
        (metadata_bytes != 0 && !rpc::send_all(socket_, metadata, metadata_bytes, deadline, err)) ||
        (body_a_bytes != 0 && !rpc::send_all(socket_, body_a, body_a_bytes, deadline, err)) ||
        (body_b_bytes != 0 && !rpc::send_all(socket_, body_b, body_b_bytes, deadline, err))) {
        broken_ = true;
        rpc::close(socket_); socket_ = rpc::kInvalidSocket;
        return false;
    }
    bytes_sent_ += kHeaderBytes + payload_bytes;
    uint16_t response_type = 0;
    uint64_t response_op = 0, response_payload_bytes = 0;
    if (!receive_header(socket_, response_type, response_op, response_payload_bytes, deadline, err) ||
        response_type != (0x8000u | type) || response_op != operation || response_payload_bytes > kMaxFrameBytes ||
        response_payload_bytes > 8ull + kMaxErrorBytes + response_bytes) {
        if (err.empty()) err = "layer-stage worker returned a mismatched response frame";
        broken_ = true;
        rpc::close(socket_); socket_ = rpc::kInvalidSocket;
        return false;
    }
    std::vector<uint8_t> frame((size_t)response_payload_bytes);
    if (!frame.empty() && !rpc::recv_all(socket_, frame.data(), frame.size(), deadline, err)) {
        broken_ = true;
        rpc::close(socket_); socket_ = rpc::kInvalidSocket;
        return false;
    }
    bytes_received_ += kHeaderBytes + response_payload_bytes;
    Reader reader{frame.data(), frame.size(), 0};
    const ParsedResponse parsed = parse_response(reader, response, response_bytes, err);
    if (parsed == ParsedResponse::malformed) {
        broken_ = true;
        rpc::close(socket_); socket_ = rpc::kInvalidSocket;
        return false;
    }
    if (parsed == ParsedResponse::rejected) return false;   // a valid rejection is recoverable on this connection
    if (!health_req) ++next_operation_;
    return true;
}

bool LayerStageClient::reset(uint64_t session, uint64_t generation, int64_t position, std::string& err) {
    if (!connected()) {
        if (endpoint_.empty() || token_.empty()) {
            err = "layer-stage RPC is disconnected and has no saved worker endpoint";
            return false;
        }
        const std::string endpoint = endpoint_, token = token_;
        const LayerStageSpec expected = spec_;
        if (!connect(endpoint, token, expected, err)) return false;
    }
    next_operation_ = 1;
    Writer m; m.u64(session); m.u64(generation); m.i64(position);
    return request(kReset, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::prefill(uint64_t session, uint64_t generation, int64_t position, uint32_t rows,
                               const int64_t* tokens, const float* residual, std::string& err) {
    if (rows == 0 || rows > spec_.max_prefill_rows || tokens == nullptr || residual == nullptr) {
        err = "layer-stage prefill has invalid rows or null input";
        return false;
    }
    uint64_t floats = 0;
    if (!activation_float_count(spec_, rows, true, floats)) {
        err = "layer-stage prefill shape overflows the protocol range";
        return false;
    }
    size_t token_bytes = 0, residual_bytes = 0;
    if (!checked_bytes(rows, sizeof(int64_t), token_bytes) || !checked_bytes(floats, sizeof(float), residual_bytes)) {
        err = "layer-stage prefill size overflows this host";
        return false;
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u32(rows); m.u64(floats);
    return request(kPrefill, m.bytes.data(), m.bytes.size(), tokens, token_bytes, residual, residual_bytes,
                   nullptr, 0, err);
}

bool LayerStageClient::verify(uint64_t session, uint64_t generation, int64_t position, uint32_t rows,
                              const int32_t* tokens, const float* handoff, int32_t* picks, std::string& err) {
    if (rows == 0 || rows > spec_.max_verify_rows || tokens == nullptr || handoff == nullptr || picks == nullptr) {
        err = "layer-stage verify has invalid rows or null input";
        return false;
    }
    uint64_t floats = 0;
    if (!activation_float_count(spec_, rows, false, floats)) {
        err = "layer-stage verify shape overflows the protocol range";
        return false;
    }
    size_t token_bytes = 0, handoff_bytes = 0, pick_bytes = 0;
    if (!checked_bytes(rows, sizeof(int32_t), token_bytes) || !checked_bytes(floats, sizeof(float), handoff_bytes) ||
        !checked_bytes(rows, sizeof(int32_t), pick_bytes)) {
        err = "layer-stage verify size overflows this host";
        return false;
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u32(rows); m.u64(floats);
    return request(kVerify, m.bytes.data(), m.bytes.size(), tokens, token_bytes, handoff, handoff_bytes,
                   picks, pick_bytes, err);
}

bool LayerStageClient::commit(uint64_t session, uint64_t generation, uint32_t keep, std::string& err) {
    Writer m; m.u64(session); m.u64(generation); m.u32(keep);
    return request(kCommit, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::abort(uint64_t session, uint64_t generation, std::string& err) {
    Writer m; m.u64(session); m.u64(generation);
    return request(kAbort, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::health(std::string& err) {
    return request(kHealth, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, err);
}

LayerStageServer::LayerStageServer(std::string token, LayerStageSpec expected, LayerStageExecutor& executor,
                                   std::chrono::milliseconds operation_timeout)
    : token_(std::move(token)), spec_(expected), executor_(executor),
      operation_timeout_(operation_timeout.count() > 0 ? operation_timeout : std::chrono::milliseconds(1)) {}

bool LayerStageServer::dispatch(uint16_t type, uint64_t operation, const uint8_t* payload, size_t payload_bytes,
                                std::vector<uint8_t>& response, std::string& err) {
    Reader r{payload, payload_bytes, 0};
    uint64_t session = 0, generation = 0;
    int64_t position = 0;
    uint32_t rows = 0, keep = 0;
    uint64_t float_count = 0;
    auto fail = [&](const std::string& why) { err = why; response = response_payload(false, why); return true; };
    if (type == kHealth) {
        if (operation != 0 || payload_bytes != 0) return fail("layer-stage health frame is invalid");
        const bool ok = executor_.health(err);
        response = response_payload(ok, ok ? std::string() : err);
        return true;
    }
    if (operation == 0) return fail("layer-stage state-changing operation has ID zero");
    if (payload_bytes < 16 || !r.u64(session) || !r.u64(generation)) return fail("layer-stage operation metadata is truncated");
    const RequestFingerprint request = fingerprint_request(payload, payload_bytes);
    if (operation == last_operation_ && type == last_type_ && session == session_ && generation == generation_) {
        if (request.bytes != last_request_bytes_ || request.a != last_request_hash_a_ || request.b != last_request_hash_b_)
            return fail("layer-stage operation ID was reused with a different request payload");
        response = last_response_;
        return true; // replay the prior acknowledgement/result only for an identical request
    }
    if (type == kReset) {
        if (operation != 1 || !r.i64(position) || r.left() != 0 || session == 0 || generation == 0 || position != 0 ||
            (uint64_t)position > spec_.max_context)
            return fail("layer-stage reset must start a fresh session at position zero");
        if (session == session_ && generation <= generation_)
            return fail("layer-stage reset generation is not newer than the current session");
        const bool ok = executor_.reset(session, generation, position, err);
        response = response_payload(ok, ok ? std::string() : err);
        if (!ok) { session_valid_ = false; return true; }
        session_ = session; generation_ = generation; last_operation_ = operation; last_type_ = type;
        last_request_bytes_ = request.bytes; last_request_hash_a_ = request.a; last_request_hash_b_ = request.b;
        position_ = position; pending_verify_rows_ = 0;
        last_response_ = response; session_valid_ = true;
        return true;
    }
    if (!session_valid_ || session != session_ || generation != generation_ || operation != last_operation_ + 1)
        return fail("layer-stage session is absent, stale, or out of operation order; reset and replay the prompt");
    bool ok = false;
    if (type == kPrefill || type == kVerify) {
        if (!r.i64(position) || !r.u32(rows) || !r.u64(float_count)) return fail("layer-stage activation metadata is truncated");
        const bool prefill = type == kPrefill;
        const uint32_t max_rows = prefill ? spec_.max_prefill_rows : spec_.max_verify_rows;
        uint64_t expected_floats = 0;
        if (!activation_float_count(spec_, rows, prefill, expected_floats))
            return fail("layer-stage activation geometry overflows the protocol range");
        if (rows == 0 || rows > max_rows || float_count != expected_floats)
            return fail("layer-stage activation shape exceeds the advertised stage capability");
        if (position < 0 || (uint64_t)position > spec_.max_context || rows > spec_.max_context - (uint64_t)position)
            return fail("layer-stage activation positions exceed the session context");
        if (pending_verify_rows_ != 0 || position != position_)
            return fail("layer-stage prefill or verifier window is out of recurrent order");
        const uint64_t token_count = rows;
        const size_t token_size = prefill ? sizeof(int64_t) : sizeof(int32_t);
        size_t token_bytes = 0, float_bytes = 0;
        if (!checked_bytes(token_count, token_size, token_bytes) || !checked_bytes(float_count, sizeof(float), float_bytes) ||
            token_bytes > r.left() || float_bytes != r.left() - token_bytes)
            return fail("layer-stage activation frame has an invalid byte count");
        if (prefill) {
            prefill_tokens_.resize(rows);
            if (!r.raw(prefill_tokens_.data(), token_bytes) || r.left() != float_bytes ||
                ((uintptr_t)r.remaining() % alignof(float)) != 0)
                return fail("layer-stage prefill frame is truncated");
            ok = executor_.prefill(session, generation, operation, position, rows, prefill_tokens_.data(),
                                   (const float*)r.remaining(), err);
            if (ok) position_ += rows;
        } else {
            verify_tokens_.resize(rows); verify_picks_.resize(rows);
            if (!r.raw(verify_tokens_.data(), token_bytes) || r.left() != float_bytes ||
                ((uintptr_t)r.remaining() % alignof(float)) != 0)
                return fail("layer-stage verify frame is truncated");
            ok = executor_.verify(session, generation, operation, position, rows, verify_tokens_.data(),
                                  (const float*)r.remaining(), verify_picks_.data(), err);
            if (ok) {
                pending_verify_rows_ = rows;
                response = response_payload(true, std::string(), verify_picks_.data(),
                                            verify_picks_.size() * sizeof(int32_t));
            }
        }
    } else if (type == kCommit) {
        if (!r.u32(keep) || r.left() != 0 || keep == 0 || keep > pending_verify_rows_ ||
            keep > spec_.max_verify_rows || (uint64_t)position_ + keep > spec_.max_context)
            return fail("layer-stage accepted-prefix commit is invalid");
        ok = executor_.commit(session, generation, operation, keep, err);
        if (ok) { position_ += keep; pending_verify_rows_ = 0; }
    } else if (type == kAbort) {
        if (r.left() != 0) return fail("layer-stage abort frame has unexpected data");
        ok = executor_.abort(session, generation, operation, err);
        session_valid_ = false; pending_verify_rows_ = 0;
    } else {
        return fail("layer-stage worker received an unknown operation");
    }
    if (!ok) {
        session_valid_ = false;
        pending_verify_rows_ = 0;
        response = response_payload(false, err.empty() ? "layer-stage operation failed" : err);
        return true;
    }
    if (response.empty()) response = response_payload(true, std::string());
    last_operation_ = operation; last_type_ = type; last_response_ = response;
    last_request_bytes_ = request.bytes; last_request_hash_a_ = request.a; last_request_hash_b_ = request.b;
    return true;
}

bool LayerStageServer::serve(rpc::Socket client, std::string& err) {
    // Replay acknowledgements belong to one TCP connection. Keep the last session identity for generation checks, but
    // never let a new connection receive an acknowledgement for state that was aborted at disconnect.
    last_operation_ = 0;
    last_type_ = 0;
    last_response_.clear();
    last_request_bytes_ = last_request_hash_a_ = last_request_hash_b_ = 0;
    if (client == rpc::kInvalidSocket || token_.empty() || token_.size() > kMaxErrorBytes || !valid_spec(spec_) ||
        native_float_marker() == 0) {
        err = "layer-stage server was configured with invalid identity, token, or connection";
        if (client != rpc::kInvalidSocket) rpc::close(client);
        return false;
    }
    std::string close_error;
    if (!rpc::set_nodelay(client, true, err)) { rpc::close(client); return false; }
    const rpc::IoDeadline init_deadline = rpc::deadline_after(kHandshakeTimeout);
    uint16_t type = 0;
    uint64_t operation = 0, bytes = 0;
    if (!receive_header(client, type, operation, bytes, init_deadline, err) || type != kHello || operation != 0 ||
        bytes > 8192) {
        if (err.empty()) err = "layer-stage client sent an invalid handshake frame";
        rpc::close(client);
        return false;
    }
    std::vector<uint8_t> payload((size_t)bytes);
    if (!payload.empty() && !rpc::recv_all(client, payload.data(), payload.size(), init_deadline, err)) {
        rpc::close(client); return false;
    }
    Reader hello{payload.data(), payload.size(), 0};
    LayerStageSpec actual{};
    uint32_t endian = 0, float_format = 0, token_bytes = 0;
    if (!read_spec(hello, actual) || !hello.u32(endian) || !hello.u32(float_format) ||
        endian != native_endian_marker() || float_format != native_float_marker() ||
        !hello.u32(token_bytes) || token_bytes == 0 || token_bytes > kMaxErrorBytes ||
        token_bytes != hello.left()) {
        err = "layer-stage client handshake is malformed or uses an incompatible integer/FP32 format";
        rpc::close(client);
        return false;
    }
    const std::string token((const char*)hello.remaining(), token_bytes);
    if (token != token_) { err = "layer-stage authentication failed"; rpc::close(client); return false; }
    if (!valid_spec(actual) || !same_spec(actual, spec_)) {
        err = "layer-stage model, config, geometry, layer range, or capability differs from the worker";
        rpc::close(client); return false;
    }
    const auto ack = response_payload(true, std::string());
    if (!send_response(client, kHello, 0, ack, init_deadline, err)) { rpc::close(client); return false; }

    for (;;) {
        uint64_t op_bytes = 0;
        rpc::IoDeadline deadline;
        err.clear();
        if (!receive_operation_header(client, type, operation, op_bytes, operation_timeout_, deadline, err)) {
            const bool disconnected = err == "RPC connection closed";
            if (!disconnected) close_error = err;
            break;
        }
        if (type < kReset || type > kHealth || op_bytes > max_operation_payload(spec_)) {
            err = "layer-stage worker received an invalid operation frame";
            break;
        }
        try {
            payload_buffer_.resize((size_t)op_bytes);
        } catch (const std::exception& e) {
            err = std::string("layer-stage request buffer allocation failed: ") + e.what();
            break;
        }
        if (!payload_buffer_.empty() && !rpc::recv_all(client, payload_buffer_.data(), payload_buffer_.size(), deadline, err)) break;
        std::vector<uint8_t> response;
        try {
            if (!dispatch(type, operation, payload_buffer_.data(), payload_buffer_.size(), response, err)) break;
        } catch (const std::exception& e) {
            err = std::string("layer-stage executor or protocol handler threw: ") + e.what();
            break;
        } catch (...) {
            err = "layer-stage executor or protocol handler threw an unknown exception";
            break;
        }
        if (!send_response(client, type, operation, response, deadline, err)) break;
    }
    rpc::close(client);
    if (session_valid_) {
        std::string ignored;
        (void)executor_.abort(session_, generation_, last_operation_ + 1, ignored);
        session_valid_ = false; pending_verify_rows_ = 0;
    }
    if (close_error.empty() && err == "RPC connection closed") { err.clear(); return true; }
    if (err.empty()) err = close_error.empty() ? "layer-stage connection ended" : close_error;
    return false;
}

} // namespace strata::core
