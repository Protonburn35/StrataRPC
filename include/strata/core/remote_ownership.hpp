#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

inline bool is_remote_owned(const std::vector<uint8_t>& mask, int64_t n_expert, int64_t layer, int64_t expert) {
    if (n_expert <= 0 || layer < 0 || expert < 0 || expert >= n_expert) return false;
    const uint64_t index = (uint64_t)layer * (uint64_t)n_expert + (uint64_t)expert;
    return index < mask.size() && mask[(size_t)index] != 0;
}

} // namespace strata::core
