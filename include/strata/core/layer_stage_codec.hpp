#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Activation representation negotiated by the persistent layer-stage RPC handshake.
/// FP32 is the compatibility/reference representation; BF16 and FP16 are explicitly lossy.
enum class LayerStageWireFormat : uint32_t { fp32 = 0, bf16 = 1, fp16 = 2 };

constexpr size_t layer_stage_wire_element_bytes(LayerStageWireFormat format) {
    return format == LayerStageWireFormat::fp32 ? 4 : (format == LayerStageWireFormat::bf16 ||
                                                        format == LayerStageWireFormat::fp16) ? 2 : 0;
}

/// Pack finite IEEE-754 activations to the requested wire representation. The 16-bit formats are little-endian.
/// On range overflow or a non-finite value, returns false without accepting a partial payload.
bool layer_stage_encode_activations(LayerStageWireFormat format, const float* src, size_t count,
                                    std::vector<uint8_t>& dst, std::string& err);

/// Decode a complete activation payload. `bytes` must exactly match `count * element_bytes` and every decoded value
/// must remain finite. The destination is untouched on failure.
bool layer_stage_decode_activations(LayerStageWireFormat format, const uint8_t* src, size_t bytes,
                                    float* dst, size_t count, std::string& err);

} // namespace strata::core
