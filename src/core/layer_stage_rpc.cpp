#include "strata/core/layer_stage_rpc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::core {
namespace {
constexpr uint16_t kVersion = kLayerStageProtocolVersion;
constexpr size_t kHeaderBytes = 24;
constexpr uint64_t kMaxFrameBytes = 1ull << 31;
constexpr uint32_t kMaxErrorBytes = 4096;
constexpr auto kHandshakeTimeout = std::chrono::minutes(10);
constexpr auto kOperationTimeout = std::chrono::minutes(5);

enum Message : uint16_t {
    kHello = 1, kReset = 2, kPrefill = 3, kVerify = 4, kCommit = 5, kAbort = 6, kHealth = 7, kResume = 8,
    kSampling = 9, kCheckpoint = 10, kPositions = 11, kSaveSnapshot = 12, kRestoreSnapshot = 13
};
constexpr size_t kMtpReplyBytes = sizeof(uint32_t) + kLayerStageMaxMtpDrafts * sizeof(int32_t) +
                                  kLayerStageMaxMtpDrafts * sizeof(float);

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
    void f32(float v) { uint32_t bits; std::memcpy(&bits, &v, sizeof(bits)); u32(bits); }
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
    bool f32(float& v) { uint32_t bits; if (!u32(bits)) return false; std::memcpy(&v, &bits, sizeof(v)); return true; }
    bool raw(void* dst, size_t bytes) {
        if (bytes > n - at) return false;
        if (bytes != 0) std::memcpy(dst, p + at, bytes);
        at += bytes; return true;
    }
    const uint8_t* remaining() const { return p + at; }
    size_t left() const { return n - at; }
};

bool valid_snapshot(const LayerStageSnapshot& s, const LayerStageSpec& spec) {
    if ((spec.capabilities & kLayerStageCapabilitySessions) == 0 || !s.id[0] || !s.id[1] ||
        s.position <= 0 || (uint64_t)s.position > spec.max_context ||
        s.checkpoints.size() > kLayerStageMaxCheckpoints) return false;
    for (size_t i = 0; i < s.checkpoints.size(); ++i) {
        if (s.checkpoints[i] <= 0 || s.checkpoints[i] > s.position ||
            std::find(s.checkpoints.begin(), s.checkpoints.begin() + i, s.checkpoints[i]) != s.checkpoints.begin() + i)
            return false;
    }
    return true;
}

void write_snapshot(Writer& w, const LayerStageSnapshot& s) {
    w.u64(s.id[0]); w.u64(s.id[1]); w.i64(s.position);
    w.u64(s.prefix.hash_a); w.u64(s.prefix.hash_b); w.u32((uint32_t)s.checkpoints.size());
    for (int64_t at : s.checkpoints) w.i64(at);
}

bool read_snapshot(Reader& r, LayerStageSnapshot& s, const LayerStageSpec& spec) {
    uint32_t count = 0;
    if (!r.u64(s.id[0]) || !r.u64(s.id[1]) || !r.i64(s.position) ||
        !r.u64(s.prefix.hash_a) || !r.u64(s.prefix.hash_b) || !r.u32(count) ||
        count > kLayerStageMaxCheckpoints || r.left() != (size_t)count * 8) return false;
    s.checkpoints.resize(count);
    for (auto& at : s.checkpoints) if (!r.i64(at)) return false;
    return valid_snapshot(s, spec);
}

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
    uint64_t floats = 0, activation_bytes = 0, token_bytes = 0;
    const uint64_t token_size = prefill ? sizeof(int64_t) : sizeof(int32_t);
    constexpr uint64_t metadata_bytes = 44;
    const size_t wire_element_bytes = layer_stage_wire_element_bytes(s.wire_format);
    return wire_element_bytes != 0 && activation_float_count(s, rows, prefill, floats) &&
           checked_mul(floats, wire_element_bytes, activation_bytes) &&
           checked_mul(rows, token_size, token_bytes) && metadata_bytes + token_bytes <= kMaxFrameBytes &&
           activation_bytes <= kMaxFrameBytes - metadata_bytes - token_bytes;
}

bool valid_spec(const LayerStageSpec& s) {
    constexpr uint32_t known_capabilities = kLayerStageCapabilityBase | kLayerStageCapabilityReuse |
                                            kLayerStageCapabilityMtp | kLayerStageCapabilitySampling |
                                            kLayerStageCapabilityVision | kLayerStageCapabilitySessions;
    return s.model_fingerprint != 0 && s.config_fingerprint != 0 && s.max_context > 0 &&
           s.max_context <= (uint64_t)std::numeric_limits<int64_t>::max() && s.n_layers > 0 &&
           s.n_embd > 0 && s.n_expert > 0 && s.hc > 0 && s.layer_begin < s.layer_end &&
           s.layer_end <= s.n_layers && s.max_verify_rows > 0 && s.max_verify_rows <= 8 && s.max_prefill_rows > 0 &&
           (s.capabilities & kLayerStageCapabilityBase) != 0 && (s.capabilities & ~known_capabilities) == 0 &&
           layer_stage_wire_element_bytes(s.wire_format) != 0 &&
           ((s.capabilities & kLayerStageCapabilityVision) == 0 ||
            s.max_context <= (kMaxFrameBytes - 16) / (3 * sizeof(int32_t)) - 64) &&
           activation_frame_fits(s, s.max_verify_rows, false) &&
           activation_frame_fits(s, s.max_prefill_rows, true);
}

uint64_t max_operation_payload(const LayerStageSpec& s) {
    uint64_t prefill_floats = 0, verify_floats = 0;
    if (!activation_float_count(s, s.max_prefill_rows, true, prefill_floats) ||
        !activation_float_count(s, s.max_verify_rows, false, verify_floats)) return 0;
    const uint64_t prefill_tokens = (uint64_t)s.max_prefill_rows * sizeof(int64_t);
    const uint64_t verify_tokens = (uint64_t)s.max_verify_rows * sizeof(int32_t);
    const uint64_t wire_bytes = layer_stage_wire_element_bytes(s.wire_format);
    const uint64_t prefill = 44 + prefill_tokens + prefill_floats * wire_bytes;
    const uint64_t verify = 44 + verify_tokens + verify_floats * wire_bytes;
    const uint64_t positions = (s.capabilities & kLayerStageCapabilityVision) != 0
                                  ? 16 + (s.max_context + 64) * 3 * sizeof(int32_t) : 0;
    return std::max<uint64_t>({prefill, verify, positions, 64 + 8 * kLayerStageMaxCheckpoints});
}

std::vector<uint8_t> encode_mtp_reply(const LayerStageMtpReply& reply) {
    Writer w;
    w.u32(reply.count);
    w.raw(reply.tokens.data(), sizeof(reply.tokens));
    w.raw(reply.probabilities.data(), sizeof(reply.probabilities));
    return std::move(w.bytes);
}

bool decode_mtp_reply(const uint8_t* bytes, size_t count, LayerStageMtpReply& reply, std::string& err) {
    Reader r{bytes, count, 0};
    LayerStageMtpReply decoded;
    if (count != kMtpReplyBytes || !r.u32(decoded.count) || decoded.count > kLayerStageMaxMtpDrafts ||
        !r.raw(decoded.tokens.data(), sizeof(decoded.tokens)) ||
        !r.raw(decoded.probabilities.data(), sizeof(decoded.probabilities)) || r.left() != 0) {
        err = "layer-stage MTP response has an invalid size or draft count";
        return false;
    }
    for (uint32_t i = 0; i < decoded.count; ++i) {
        if (decoded.tokens[i] < 0 || !std::isfinite(decoded.probabilities[i]) ||
            decoded.probabilities[i] < 0.0f || decoded.probabilities[i] > 1.0f) {
            err = "layer-stage MTP response contains an invalid draft token or probability";
            return false;
        }
    }
    reply = decoded;
    return true;
}

bool valid_mtp_reply(const LayerStageMtpReply& reply, uint32_t max_count, std::string& err) {
    if (reply.count > max_count || reply.count > kLayerStageMaxMtpDrafts) {
        err = "layer-stage MTP executor returned too many drafts for this window";
        return false;
    }
    for (uint32_t i = 0; i < reply.count; ++i) {
        if (reply.tokens[i] < 0 || !std::isfinite(reply.probabilities[i]) ||
            reply.probabilities[i] < 0.0f || reply.probabilities[i] > 1.0f) {
            err = "layer-stage MTP executor returned an invalid draft token or probability";
            return false;
        }
    }
    return true;
}

bool layer_stage_stats_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_LAYER_STAGE_STATS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

const char* message_name(uint16_t type) {
    switch (type) {
        case kReset: return "reset";
        case kResume: return "resume";
        case kSampling: return "sampling";
        case kCheckpoint: return "checkpoint";
        case kPositions: return "positions";
        case kSaveSnapshot: return "save-snapshot";
        case kRestoreSnapshot: return "restore-snapshot";
        case kPrefill: return "prefill";
        case kVerify: return "verify";
        case kCommit: return "commit";
        case kAbort: return "abort";
        case kHealth: return "health";
        default: return "unknown";
    }
}

void write_spec(Writer& w, const LayerStageSpec& s) {
    w.u64(s.model_fingerprint); w.u64(s.config_fingerprint); w.u64(s.max_context);
    w.u32(s.n_layers); w.u32(s.n_embd); w.u32(s.n_expert); w.u32(s.hc);
    w.u32(s.layer_begin); w.u32(s.layer_end); w.u32(s.max_verify_rows); w.u32(s.max_prefill_rows);
    w.u32(s.capabilities); w.u32((uint32_t)s.wire_format);
}

bool read_spec(Reader& r, LayerStageSpec& s) {
    return r.u64(s.model_fingerprint) && r.u64(s.config_fingerprint) && r.u64(s.max_context) &&
           r.u32(s.n_layers) && r.u32(s.n_embd) && r.u32(s.n_expert) && r.u32(s.hc) &&
           r.u32(s.layer_begin) && r.u32(s.layer_end) && r.u32(s.max_verify_rows) &&
           r.u32(s.max_prefill_rows) && r.u32(s.capabilities) && [&] {
               uint32_t format = 0;
               if (!r.u32(format) || format > (uint32_t)LayerStageWireFormat::fp16) return false;
               s.wire_format = (LayerStageWireFormat)format;
               return true;
           }();
}

bool same_spec(const LayerStageSpec& a, const LayerStageSpec& b) {
    return a.model_fingerprint == b.model_fingerprint && a.config_fingerprint == b.config_fingerprint &&
           a.max_context == b.max_context && a.n_layers == b.n_layers && a.n_embd == b.n_embd &&
           a.n_expert == b.n_expert && a.hc == b.hc && a.layer_begin == b.layer_begin &&
           a.layer_end == b.layer_end && a.max_verify_rows == b.max_verify_rows &&
           a.max_prefill_rows == b.max_prefill_rows && a.capabilities == b.capabilities &&
           a.wire_format == b.wire_format;
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
    if (std::memcmp(h.data(), "STGE", 4) != 0) {
        err = "layer-stage frame has invalid magic";
        return false;
    }
    if (get16(h.data() + 4) != kVersion) {
        err = "incompatible layer-stage protocol version; update both PCs to the same release";
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

uint16_t float_to_bf16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t tie_to_even = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + tie_to_even) >> 16);
}

float bf16_to_float(uint16_t x) {
    const uint32_t bits = (uint32_t)x << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool float_to_fp16(float x, uint16_t& out) {
    if (!std::isfinite(x) || std::fabs(x) > 65504.0f) return false;
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0) { out = sign; return true; } // FP32 subnormals round below the FP16 subnormal range.
    int e = (int)exponent - 127;
    if (e > 15) return false;
    if (e >= -14) {
        uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
        int half_e = e + 15;
        if ((rounded & 0x800000u) != 0) { rounded = 0; ++half_e; }
        if (half_e >= 31) return false;
        out = (uint16_t)(sign | ((uint16_t)half_e << 10) | (uint16_t)(rounded >> 13));
        return true;
    }
    if (e < -25) { out = sign; return true; }
    const uint32_t significand = 0x800000u | mantissa;
    const unsigned shift = (unsigned)(-e - 1);
    const uint32_t truncated = significand >> shift;
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = significand & mask;
    const uint32_t halfway = 1u << (shift - 1u);
    const uint32_t rounded = truncated + (remainder > halfway || (remainder == halfway && (truncated & 1u)));
    out = (uint16_t)(sign | (uint16_t)rounded);
    return true;
}

float fp16_to_float(uint16_t x) {
    const bool negative = (x & 0x8000u) != 0;
    const uint32_t exponent = (x >> 10) & 0x1fu;
    const uint32_t mantissa = x & 0x3ffu;
    float value = exponent == 0 ? std::ldexp((float)mantissa, -24)
                 : std::ldexp(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    return negative ? -value : value;
}

uint16_t get16le(const uint8_t* p) { return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8); }
void put16le(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

} // namespace

bool layer_stage_valid_sampling(const LayerStageSampling& s) {
    return std::isfinite(s.temperature) && s.temperature >= 0.0f &&
           std::isfinite(s.top_p) && s.top_p >= 0.0f && s.top_p <= 1.0f &&
           std::isfinite(s.min_p) && s.min_p >= 0.0f && s.min_p <= 1.0f && s.top_k >= 0;
}

uint32_t layer_stage_output_commit_count(const int32_t* picks, uint32_t accepted, int64_t remaining,
                                        const std::vector<int64_t>& eos_ids) {
    const auto count = (uint32_t)std::min<int64_t>(accepted, std::max<int64_t>(remaining, 0));
    for (uint32_t i = 0; i < count; ++i)
        if (std::find(eos_ids.begin(), eos_ids.end(), (int64_t)picks[i]) != eos_ids.end()) return i + 1;
    return count;
}

LayerStagePrefixIdentity layer_stage_prefix_identity(const int32_t* tokens, size_t count) {
    LayerStagePrefixIdentity identity{1469598103934665603ull, 0x9e3779b97f4a7c15ull};
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = (uint32_t)tokens[i];
        for (unsigned shift = 0; shift < 32; shift += 8) {
            const uint64_t byte = (value >> shift) & 0xffu;
            identity.hash_a = (identity.hash_a ^ byte) * 1099511628211ull;
            identity.hash_b ^= byte + 0x517cc1b727220a95ull + (identity.hash_b << 6) + (identity.hash_b >> 2);
            identity.hash_b = (identity.hash_b ^ (identity.hash_b >> 29)) * 0xbf58476d1ce4e5b9ull;
        }
    }
    identity.hash_a ^= (uint64_t)count;
    identity.hash_b ^= (uint64_t)count + 0x94d049bb133111ebull;
    identity.hash_b = (identity.hash_b ^ (identity.hash_b >> 30)) * 0xbf58476d1ce4e5b9ull;
    identity.hash_b = (identity.hash_b ^ (identity.hash_b >> 27)) * 0x94d049bb133111ebull;
    identity.hash_b ^= identity.hash_b >> 31;
    return identity;
}

bool layer_stage_encode_activations(LayerStageWireFormat format, const float* src, size_t count,
                                    std::vector<uint8_t>& dst, std::string& err) {
    const size_t element_bytes = layer_stage_wire_element_bytes(format);
    if (element_bytes == 0 || (count != 0 && src == nullptr) || count > std::numeric_limits<size_t>::max() / element_bytes) {
        err = "layer-stage activation encoder received an invalid format, pointer, or element count";
        dst.clear();
        return false;
    }
    dst.resize(count * element_bytes);
    if (format == LayerStageWireFormat::fp32) {
        for (size_t i = 0; i < count; ++i) if (!std::isfinite(src[i])) {
            err = "layer-stage FP32 activation contains a non-finite value";
            dst.clear();
            return false;
        }
        if (!dst.empty()) std::memcpy(dst.data(), src, dst.size());
        return true;
    }
    for (size_t i = 0; i < count; ++i) {
        uint16_t encoded = 0;
        if (!std::isfinite(src[i])) {
            err = "layer-stage activation contains a non-finite value";
            dst.clear();
            return false;
        }
        if (format == LayerStageWireFormat::bf16) {
            encoded = float_to_bf16(src[i]);
            if ((encoded & 0x7f80u) == 0x7f80u) {
                err = "layer-stage BF16 activation overflows to a non-finite value";
                dst.clear();
                return false;
            }
        } else if (!float_to_fp16(src[i], encoded)) {
            err = "layer-stage FP16 activation is non-finite or exceeds the finite range 65504";
            dst.clear();
            return false;
        }
        put16le(dst.data() + i * 2, encoded);
    }
    return true;
}

bool layer_stage_decode_activations(LayerStageWireFormat format, const uint8_t* src, size_t bytes,
                                    float* dst, size_t count, std::string& err) {
    const size_t element_bytes = layer_stage_wire_element_bytes(format);
    if (element_bytes == 0 || count > std::numeric_limits<size_t>::max() / element_bytes ||
        bytes != count * element_bytes || (bytes != 0 && src == nullptr) || (count != 0 && dst == nullptr)) {
        err = "layer-stage activation payload has a mismatched format, byte count, or pointer";
        return false;
    }
    if (format == LayerStageWireFormat::fp32) {
        for (size_t i = 0; i < count; ++i) {
            float value = 0.0f;
            std::memcpy(&value, src + i * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) { err = "layer-stage FP32 activation contains a non-finite value"; return false; }
        }
        if (bytes != 0) std::memcpy(dst, src, bytes);
        return true;
    }
    // Validate first so a malformed tail cannot be partially accepted by a caller that reuses its destination buffer.
    for (size_t i = 0; i < count; ++i) {
        const uint16_t encoded = get16le(src + i * 2);
        if ((format == LayerStageWireFormat::bf16 && (encoded & 0x7f80u) == 0x7f80u) ||
            (format == LayerStageWireFormat::fp16 && (encoded & 0x7c00u) == 0x7c00u)) {
            err = "layer-stage reduced-precision activation payload contains a non-finite value";
            return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const uint16_t encoded = get16le(src + i * 2);
        dst[i] = format == LayerStageWireFormat::bf16 ? bf16_to_float(encoded) : fp16_to_float(encoded);
    }
    return true;
}

std::string layer_stage_snapshot_name(const std::array<uint64_t, 2>& id) {
    char name[48];
    std::snprintf(name, sizeof(name), "%016llx%016llx.srpc", (unsigned long long)id[0], (unsigned long long)id[1]);
    return name;
}

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
    const auto request_started = std::chrono::steady_clock::now();
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
    if (layer_stage_stats_enabled()) {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - request_started).count();
        std::fprintf(stderr,
                     "strata layer-stage RPC stats: %s op=%llu tx=%llu B rx=%llu B elapsed=%.2f ms cumulative_tx=%llu B cumulative_rx=%llu B wire=%s\n",
                     message_name(type), (unsigned long long)operation,
                     (unsigned long long)(kHeaderBytes + payload_bytes),
                     (unsigned long long)(kHeaderBytes + response_payload_bytes), elapsed_ms,
                     (unsigned long long)bytes_sent_, (unsigned long long)bytes_received_,
                     spec_.wire_format == LayerStageWireFormat::fp32 ? "fp32" :
                     spec_.wire_format == LayerStageWireFormat::bf16 ? "bf16" : "fp16");
    }
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

bool LayerStageClient::resume(uint64_t session, uint64_t generation, int64_t position,
                              LayerStagePrefixIdentity prefix, std::string& err) {
    if (!connected()) {
        err = "layer-stage resume requires the existing worker connection; replay from token zero after reconnect";
        return false;
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u64(prefix.hash_a); m.u64(prefix.hash_b);
    return request(kResume, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::set_sampling(uint64_t session, uint64_t generation, const LayerStageSampling& sampling,
                                     std::string& err) {
    if (!layer_stage_valid_sampling(sampling)) { err = "invalid layer-stage sampling settings"; return false; }
    Writer m; m.u64(session); m.u64(generation);
    m.f32(sampling.temperature); m.f32(sampling.top_p); m.f32(sampling.min_p);
    m.u32((uint32_t)sampling.top_k); m.u64(sampling.seed);
    return request(kSampling, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::set_positions(uint64_t session, uint64_t generation,
                                     const std::vector<int32_t>& positions, std::string& err) {
    if ((spec_.capabilities & kLayerStageCapabilityVision) == 0 ||
        positions.size() != (spec_.max_context + 64) * 3 ||
        std::any_of(positions.begin(), positions.end(), [](int32_t p) { return p < 0; })) {
        err = "vision positions must cover the negotiated context and generation tail"; return false;
    }
    Writer m; m.u64(session); m.u64(generation);
    return request(kPositions, m.bytes.data(), m.bytes.size(), positions.data(), positions.size() * sizeof(int32_t),
                   nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::save_snapshot(uint64_t session, uint64_t generation,
                                    const LayerStageSnapshot& snapshot, std::string& err) {
    if (!valid_snapshot(snapshot, spec_)) { err = "invalid or disabled worker disk snapshot"; return false; }
    Writer m; m.u64(session); m.u64(generation); write_snapshot(m, snapshot);
    return request(kSaveSnapshot, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::restore_snapshot(uint64_t session, uint64_t generation,
                                       const LayerStageSnapshot& snapshot, std::string& err) {
    if (!valid_snapshot(snapshot, spec_)) { err = "invalid or disabled worker disk snapshot"; return false; }
    if (!connected()) {
        const std::string endpoint = endpoint_, token = token_;
        const LayerStageSpec expected = spec_;
        if (endpoint.empty() || token.empty()) { err = "no saved worker endpoint for disk restore"; return false; }
        if (!connect(endpoint, token, expected, err)) return false;
    }
    next_operation_ = 1;
    Writer m; m.u64(session); m.u64(generation); write_snapshot(m, snapshot);
    return request(kRestoreSnapshot, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
}

bool LayerStageClient::checkpoint(uint64_t session, uint64_t generation, int64_t position,
                                   LayerStagePrefixIdentity prefix, const std::vector<int64_t>& retained,
                                   std::string& err) {
    if (retained.empty() || retained.size() > kLayerStageMaxCheckpoints) {
        err = "layer-stage checkpoint retention exceeds the supported limit"; return false;
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u64(prefix.hash_a); m.u64(prefix.hash_b);
    m.u32((uint32_t)retained.size());
    for (int64_t at : retained) m.i64(at);
    return request(kCheckpoint, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0, nullptr, 0, err);
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
    size_t token_bytes = 0, activation_bytes = 0;
    if (!checked_bytes(rows, sizeof(int64_t), token_bytes) ||
        !checked_bytes(floats, layer_stage_wire_element_bytes(spec_.wire_format), activation_bytes)) {
        err = "layer-stage prefill size overflows this host";
        return false;
    }
    const void* activation_body = residual;
    if (spec_.wire_format == LayerStageWireFormat::fp32) {
        for (uint64_t i = 0; i < floats; ++i) if (!std::isfinite(residual[i])) {
            err = "layer-stage FP32 prefill activation contains a non-finite value";
            return false;
        }
    } else {
        if (!layer_stage_encode_activations(spec_.wire_format, residual, (size_t)floats, activation_wire_, err)) return false;
        activation_body = activation_wire_.data();
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u32(rows); m.u64(floats); m.u64(activation_bytes);
    return request(kPrefill, m.bytes.data(), m.bytes.size(), tokens, token_bytes, activation_body, activation_bytes,
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
    size_t token_bytes = 0, activation_bytes = 0, pick_bytes = 0;
    if (!checked_bytes(rows, sizeof(int32_t), token_bytes) ||
        !checked_bytes(floats, layer_stage_wire_element_bytes(spec_.wire_format), activation_bytes) ||
        !checked_bytes(rows, sizeof(int32_t), pick_bytes)) {
        err = "layer-stage verify size overflows this host";
        return false;
    }
    const void* activation_body = handoff;
    if (spec_.wire_format == LayerStageWireFormat::fp32) {
        for (uint64_t i = 0; i < floats; ++i) if (!std::isfinite(handoff[i])) {
            err = "layer-stage FP32 verifier activation contains a non-finite value";
            return false;
        }
    } else {
        if (!layer_stage_encode_activations(spec_.wire_format, handoff, (size_t)floats, activation_wire_, err)) return false;
        activation_body = activation_wire_.data();
    }
    Writer m; m.u64(session); m.u64(generation); m.i64(position); m.u32(rows); m.u64(floats); m.u64(activation_bytes);
    return request(kVerify, m.bytes.data(), m.bytes.size(), tokens, token_bytes, activation_body, activation_bytes,
                   picks, pick_bytes, err);
}

bool LayerStageClient::commit(uint64_t session, uint64_t generation, uint32_t keep, bool draft_next, float min_p,
                              LayerStageMtpReply* reply, std::string& err) {
    const bool mtp_enabled = (spec_.capabilities & kLayerStageCapabilityMtp) != 0;
    if (mtp_enabled != (reply != nullptr) || (draft_next && !mtp_enabled) || !std::isfinite(min_p) ||
        min_p < 0.0f || min_p > 1.0f) {
        err = "layer-stage commit MTP response does not match the negotiated capability or minimum probability";
        return false;
    }
    uint32_t min_p_bits = 0;
    std::memcpy(&min_p_bits, &min_p, sizeof(min_p_bits));
    Writer m; m.u64(session); m.u64(generation); m.u32(keep); m.u32(draft_next ? 1u : 0u); m.u32(min_p_bits);
    std::array<uint8_t, kMtpReplyBytes> body{};
    if (!request(kCommit, m.bytes.data(), m.bytes.size(), nullptr, 0, nullptr, 0,
                 mtp_enabled ? body.data() : nullptr, mtp_enabled ? body.size() : 0, err)) return false;
    if (mtp_enabled) return decode_mtp_reply(body.data(), body.size(), *reply, err);
    return true;
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
    uint64_t float_count = 0, activation_bytes = 0;
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
    if (type == kRestoreSnapshot) {
        LayerStageSnapshot snapshot;
        if (operation != 1 || session == 0 || generation == 0 || pending_verify_rows_ != 0 ||
            (session == session_ && generation <= generation_) || !read_snapshot(r, snapshot, spec_))
            return fail("disk restore requires a fresh generation and a valid snapshot with no pending verification");
        // A failed restore may have written device state. Require reset/another restore before any inference.
        session_valid_ = false; positions_ready_ = false;
        std::vector<int32_t> tokens;
        if (!executor_.restore_snapshot(snapshot, session, generation, tokens, err))
            return fail(err.empty() ? "worker disk restore failed" : err);
        const auto prefix = layer_stage_prefix_identity(tokens.data(), tokens.size());
        if (tokens.size() != (size_t)snapshot.position || prefix.hash_a != snapshot.prefix.hash_a ||
            prefix.hash_b != snapshot.prefix.hash_b) return fail("restored worker prefix differs from host snapshot");
        committed_tokens_ = std::move(tokens); checkpoint_positions_ = snapshot.checkpoints;
        position_ = snapshot.position; pending_verify_rows_ = 0;
        session_ = session; generation_ = generation; session_valid_ = true;
        response = response_payload(true, std::string());
        last_operation_ = operation; last_type_ = type; last_response_ = response;
        last_request_bytes_ = request.bytes; last_request_hash_a_ = request.a; last_request_hash_b_ = request.b;
        return true;
    }
    if (type == kReset) {
        if (operation != 1 || !r.i64(position) || r.left() != 0 || session == 0 || generation == 0 || position != 0 ||
            (uint64_t)position > spec_.max_context)
            return fail("layer-stage reset must start a fresh session at position zero");
        if (session == session_ && generation <= generation_)
            return fail("layer-stage reset generation is not newer than the current session");
        const bool ok = executor_.reset(session, generation, position, err);
        response = response_payload(ok, ok ? std::string() : err);
        if (!ok) { session_valid_ = false; pending_verify_rows_ = 0; committed_tokens_.clear(); return true; }
        session_ = session; generation_ = generation; last_operation_ = operation; last_type_ = type;
        last_request_bytes_ = request.bytes; last_request_hash_a_ = request.a; last_request_hash_b_ = request.b;
        position_ = position; pending_verify_rows_ = 0; committed_tokens_.clear(); checkpoint_positions_.clear();
        last_response_ = response; session_valid_ = true; positions_ready_ = false;
        return true;
    }
    if (type == kResume) {
        uint64_t hash_a = 0, hash_b = 0;
        if (!session_valid_ || session != session_ || generation <= generation_ || operation != last_operation_ + 1 ||
            !r.i64(position) || !r.u64(hash_a) || !r.u64(hash_b) || r.left() != 0 || position < 0 ||
            position > position_ || pending_verify_rows_ != 0 || (uint64_t)position > spec_.max_context ||
            committed_tokens_.size() != (size_t)position_ ||
            (position != position_ && std::find(checkpoint_positions_.begin(), checkpoint_positions_.end(), position) ==
                                      checkpoint_positions_.end()) ||
            (spec_.capabilities & kLayerStageCapabilityReuse) == 0)
            return fail("layer-stage resume needs a retained live/checkpoint position with no verification window pending");
        const LayerStagePrefixIdentity actual = layer_stage_prefix_identity(committed_tokens_.data(), (size_t)position);
        if (actual.hash_a != hash_a || actual.hash_b != hash_b)
            return fail("layer-stage resume token-prefix identity differs; reset and replay the full prompt");
        if (!executor_.resume(session, generation_, generation, position, err)) {
            session_valid_ = false;
            return fail(err.empty() ? "layer-stage worker refused the resume state" : err);
        }
        generation_ = generation;
        positions_ready_ = false;
        position_ = position;
        committed_tokens_.resize((size_t)position);
        checkpoint_positions_.erase(std::remove_if(checkpoint_positions_.begin(), checkpoint_positions_.end(),
                                   [&](int64_t at) { return at > position; }), checkpoint_positions_.end());
        response = response_payload(true, std::string());
        last_operation_ = operation; last_type_ = type; last_response_ = response;
        last_request_bytes_ = request.bytes; last_request_hash_a_ = request.a; last_request_hash_b_ = request.b;
        return true;
    }
    if (!session_valid_ || session != session_ || generation != generation_ || operation != last_operation_ + 1)
        return fail("layer-stage session is absent, stale, or out of operation order; reset and replay the prompt");
    bool ok = false;
    if (type == kSaveSnapshot) {
        LayerStageSnapshot snapshot;
        if (pending_verify_rows_ != 0 || !read_snapshot(r, snapshot, spec_) || snapshot.position != position_)
            return fail("disk save requires the current committed position with no pending verification");
        const auto prefix = layer_stage_prefix_identity(committed_tokens_.data(), committed_tokens_.size());
        if (prefix.hash_a != snapshot.prefix.hash_a || prefix.hash_b != snapshot.prefix.hash_b)
            return fail("disk save host and worker prefixes differ");
        for (int64_t at : snapshot.checkpoints)
            if (std::find(checkpoint_positions_.begin(), checkpoint_positions_.end(), at) == checkpoint_positions_.end())
                return fail("disk save names a checkpoint no longer retained by the worker");
        // Disk-full / permission errors do not discard an otherwise usable live session.
        if (!executor_.save_snapshot(snapshot, committed_tokens_, err))
            return fail(err.empty() ? "worker disk save failed" : err);
        ok = true;
    } else if (type == kPositions) {
        if ((spec_.capabilities & kLayerStageCapabilityVision) == 0 || pending_verify_rows_ != 0 ||
            r.left() != (spec_.max_context + 64) * 3 * sizeof(int32_t))
            return fail("vision positions do not match the negotiated context or a verification window is pending");
        std::vector<int32_t> positions(r.left() / sizeof(int32_t));
        if (!r.raw(positions.data(), positions.size() * sizeof(int32_t)) ||
            std::any_of(positions.begin(), positions.end(), [](int32_t p) { return p < 0; }))
            return fail("vision positions contain an invalid coordinate");
        ok = executor_.set_positions(positions, err);
        positions_ready_ = ok;
    } else if (type == kSampling) {
        LayerStageSampling sampling;
        uint32_t top_k = 0;
        if ((spec_.capabilities & kLayerStageCapabilitySampling) == 0 || pending_verify_rows_ != 0 ||
            !r.f32(sampling.temperature) || !r.f32(sampling.top_p) || !r.f32(sampling.min_p) ||
            !r.u32(top_k) || !r.u64(sampling.seed) || r.left() != 0 || top_k > INT32_MAX)
            return fail("layer-stage sampling frame is invalid or a verifier window is pending");
        sampling.top_k = (int32_t)top_k;
        if (!layer_stage_valid_sampling(sampling)) return fail("layer-stage sampling settings are invalid");
        ok = executor_.set_sampling(sampling, err);
    } else if (type == kCheckpoint) {
        uint64_t hash_a = 0, hash_b = 0;
        uint32_t count = 0;
        if ((spec_.capabilities & kLayerStageCapabilityReuse) == 0 || pending_verify_rows_ != 0 ||
            !r.i64(position) || !r.u64(hash_a) || !r.u64(hash_b) || !r.u32(count) ||
            position <= 0 || position != position_ || count == 0 || count > kLayerStageMaxCheckpoints ||
            r.left() != (size_t)count * sizeof(int64_t))
            return fail("layer-stage checkpoint must save the current committed position within its retention limit");
        const auto actual = layer_stage_prefix_identity(committed_tokens_.data(), committed_tokens_.size());
        if (actual.hash_a != hash_a || actual.hash_b != hash_b)
            return fail("layer-stage checkpoint token-prefix identity differs");
        std::vector<int64_t> retained;
        for (uint32_t i = 0; i < count; ++i) {
            int64_t at = 0;
            if (!r.i64(at) || at <= 0 || at > position ||
                std::find(retained.begin(), retained.end(), at) != retained.end() ||
                (at != position && std::find(checkpoint_positions_.begin(), checkpoint_positions_.end(), at) ==
                                   checkpoint_positions_.end()))
                return fail("layer-stage checkpoint retention names a missing or duplicate position");
            retained.push_back(at);
        }
        if (std::find(retained.begin(), retained.end(), position) == retained.end())
            return fail("layer-stage checkpoint retention must include the newly saved position");
        ok = executor_.checkpoint(position, committed_tokens_, retained, err);
        if (ok) checkpoint_positions_ = std::move(retained);
    } else if (type == kPrefill || type == kVerify) {
        if ((spec_.capabilities & kLayerStageCapabilityVision) != 0 && !positions_ready_)
            return fail("configure vision positions after reset or resume before running the stage");
        if (!r.i64(position) || !r.u32(rows) || !r.u64(float_count) || !r.u64(activation_bytes))
            return fail("layer-stage activation metadata is truncated");
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
        size_t token_bytes = 0, expected_activation_bytes = 0;
        const size_t element_bytes = layer_stage_wire_element_bytes(spec_.wire_format);
        if (!checked_bytes(token_count, token_size, token_bytes) ||
            !checked_bytes(float_count, element_bytes, expected_activation_bytes) ||
            activation_bytes != expected_activation_bytes || token_bytes > r.left() ||
            expected_activation_bytes != r.left() - token_bytes)
            return fail("layer-stage activation frame has an invalid byte count");
        const uint8_t* activation_wire = nullptr;
        const float* activation = nullptr;
        if (prefill) {
            prefill_tokens_.resize(rows);
            if (!r.raw(prefill_tokens_.data(), token_bytes) || r.left() != expected_activation_bytes)
                return fail("layer-stage prefill frame is truncated");
            for (uint32_t i = 0; i < rows; ++i)
                if (prefill_tokens_[i] < 0 || prefill_tokens_[i] > std::numeric_limits<int32_t>::max())
                    return fail("layer-stage prefill contains a token ID outside the committed-prefix range");
            activation_wire = r.remaining();
            if (spec_.wire_format == LayerStageWireFormat::fp32) {
                if (((uintptr_t)activation_wire % alignof(float)) != 0)
                    return fail("layer-stage FP32 prefill activation is misaligned");
                activation = (const float*)activation_wire;
                for (uint64_t i = 0; i < float_count; ++i) if (!std::isfinite(activation[i]))
                    return fail("layer-stage FP32 prefill activation contains a non-finite value");
            } else {
                activation_buffer_.resize((size_t)float_count);
                if (!layer_stage_decode_activations(spec_.wire_format, activation_wire, expected_activation_bytes,
                                                     activation_buffer_.data(), activation_buffer_.size(), err))
                    return fail(err);
                activation = activation_buffer_.data();
            }
            ok = executor_.prefill(session, generation, operation, position, rows, prefill_tokens_.data(), activation, err);
            if (ok) {
                for (int64_t token : prefill_tokens_) committed_tokens_.push_back((int32_t)token);
                position_ += rows;
            }
        } else {
            verify_tokens_.resize(rows); verify_picks_.resize(rows);
            if (!r.raw(verify_tokens_.data(), token_bytes) || r.left() != expected_activation_bytes)
                return fail("layer-stage verify frame is truncated");
            activation_wire = r.remaining();
            if (spec_.wire_format == LayerStageWireFormat::fp32) {
                if (((uintptr_t)activation_wire % alignof(float)) != 0)
                    return fail("layer-stage FP32 verifier activation is misaligned");
                activation = (const float*)activation_wire;
                for (uint64_t i = 0; i < float_count; ++i) if (!std::isfinite(activation[i]))
                    return fail("layer-stage FP32 verifier activation contains a non-finite value");
            } else {
                activation_buffer_.resize((size_t)float_count);
                if (!layer_stage_decode_activations(spec_.wire_format, activation_wire, expected_activation_bytes,
                                                     activation_buffer_.data(), activation_buffer_.size(), err))
                    return fail(err);
                activation = activation_buffer_.data();
            }
            ok = executor_.verify(session, generation, operation, position, rows, verify_tokens_.data(), activation,
                                  verify_picks_.data(), err);
            if (ok) {
                pending_verify_rows_ = rows;
                response = response_payload(true, std::string(), verify_picks_.data(),
                                            verify_picks_.size() * sizeof(int32_t));
            }
        }
    } else if (type == kCommit) {
        uint32_t draft_next_u32 = 0, min_p_bits = 0;
        if (!r.u32(keep) || !r.u32(draft_next_u32) || !r.u32(min_p_bits) || r.left() != 0 ||
            draft_next_u32 > 1 || keep == 0 || keep > pending_verify_rows_ ||
            keep > spec_.max_verify_rows || (uint64_t)position_ + keep > spec_.max_context)
            return fail("layer-stage accepted-prefix commit is invalid");
        const bool mtp_enabled = (spec_.capabilities & kLayerStageCapabilityMtp) != 0;
        const bool draft_next = draft_next_u32 != 0;
        float min_p = 0.0f;
        std::memcpy(&min_p, &min_p_bits, sizeof(min_p));
        if ((draft_next && !mtp_enabled) || !std::isfinite(min_p) || min_p < 0.0f || min_p > 1.0f)
            return fail("layer-stage accepted-prefix commit has unsupported MTP fields");
        LayerStageMtpReply mtp_reply;
        ok = executor_.commit(session, generation, operation, keep, draft_next, min_p,
                              mtp_enabled ? &mtp_reply : nullptr, err);
        // Draft capacity is negotiated for the next verify window. The current window may be shorter
        // (the first decode verify has one row), and MTP can still draft up to max_verify_rows - 1 tokens.
        const uint32_t max_drafts = draft_next ? spec_.max_verify_rows - 1 : 0;
        if (ok && mtp_enabled && !valid_mtp_reply(mtp_reply, max_drafts, err))
            ok = false;
        if (ok) {
            for (uint32_t i = 0; i < keep; ++i) committed_tokens_.push_back(verify_tokens_[i]);
            position_ += keep; pending_verify_rows_ = 0;
            if (mtp_enabled) {
                const std::vector<uint8_t> body = encode_mtp_reply(mtp_reply);
                response = response_payload(true, std::string(), body.data(), body.size());
            }
        }
    } else if (type == kAbort) {
        if (r.left() != 0) return fail("layer-stage abort frame has unexpected data");
        ok = executor_.abort(session, generation, operation, err);
        session_valid_ = false; pending_verify_rows_ = 0; committed_tokens_.clear(); checkpoint_positions_.clear();
    } else {
        return fail("layer-stage worker received an unknown operation");
    }
    if (!ok) {
        session_valid_ = false;
        pending_verify_rows_ = 0;
        committed_tokens_.clear();
        checkpoint_positions_.clear();
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
        err = actual.wire_format != spec_.wire_format || actual.capabilities != spec_.capabilities
                  ? "layer-stage wire format or enabled feature flags differ; use matching --layer-stage-* options on both PCs"
                  : "layer-stage model, config, geometry, or layer range differs from the worker";
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
        if (type < kReset || type > kRestoreSnapshot || op_bytes > max_operation_payload(spec_)) {
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
        committed_tokens_.clear();
        checkpoint_positions_.clear();
    }
    if (close_error.empty() && err == "RPC connection closed") { err.clear(); return true; }
    if (err.empty()) err = close_error.empty() ? "layer-stage connection ended" : close_error;
    return false;
}

} // namespace strata::core
