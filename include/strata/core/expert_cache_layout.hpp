#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {

struct LayerSlotRange {
    int64_t begin = 0;
    int64_t end = 0;
};

/// Build contiguous ranges from the exact slot count assigned to every global layer. A remote-owned layer has a
/// count of zero and therefore receives an empty range without changing the numbering of later layers.
inline bool make_layer_slot_ranges(const std::vector<int64_t>& counts, std::vector<LayerSlotRange>& ranges,
                                   int64_t& total, std::string& err) {
    ranges.clear();
    total = 0;
    ranges.reserve(counts.size());
    for (const int64_t count : counts) {
        if (count < 0 || count > std::numeric_limits<int64_t>::max() - total) {
            err = "expert-cache layer slot counts are negative or overflow";
            ranges.clear();
            total = 0;
            return false;
        }
        ranges.push_back({total, total + count});
        total += count;
    }
    return true;
}

/// Determine the transfer size for a slot copy, rejecting a blob that cannot fit instead of silently truncating it.
inline bool slot_copy_size(int64_t capacity, int64_t requested, size_t& bytes, std::string& err) {
    if (capacity <= 0 || requested < 0 || requested > capacity ||
        (uint64_t)(requested == 0 ? capacity : requested) > (uint64_t)std::numeric_limits<size_t>::max()) {
        err = "expert-cache blob does not fit its slot";
        bytes = 0;
        return false;
    }
    bytes = (size_t)(requested == 0 ? capacity : requested);
    return true;
}

} // namespace strata::core
