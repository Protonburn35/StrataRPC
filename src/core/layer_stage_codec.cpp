// The layer-stage activation codec (StrataRPC): FP32 is exact; BF16 / FP16 halve the bytes on the link and are lossy.
// Split out of StrataRPC's layer_stage_rpc.cpp so the DeepSeek stage protocol (src/dsv41/ds4_stage.cpp) shares it.
#include "strata/core/layer_stage_codec.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace strata::core {
namespace {

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

} // namespace strata::core
