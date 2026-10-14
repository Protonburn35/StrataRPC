#include "strata/core/expert_cache_layout.hpp"
#include "strata/core/remote_ownership.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::core::LayerSlotRange;

namespace {
bool check_layout(const std::vector<int64_t>& counts, const std::vector<int64_t>& blob_bytes) {
    std::vector<LayerSlotRange> ranges;
    int64_t total = 0;
    std::string err;
    if (!strata::core::make_layer_slot_ranges(counts, ranges, total, err)) {
        std::fprintf(stderr, "range construction failed: %s\n", err.c_str());
        return false;
    }
    int64_t cursor = 0;
    std::vector<int64_t> capacities;
    for (size_t layer = 0; layer < counts.size(); ++layer) {
        if (ranges[layer].begin != cursor || ranges[layer].end != cursor + counts[layer]) {
            std::fprintf(stderr, "layer %zu has range [%lld,%lld), expected [%lld,%lld)\n", layer,
                         (long long)ranges[layer].begin, (long long)ranges[layer].end,
                         (long long)cursor, (long long)(cursor + counts[layer]));
            return false;
        }
        cursor = ranges[layer].end;
        for (int64_t slot = 0; slot < counts[layer]; ++slot)
            capacities.push_back((blob_bytes[layer] + 255) / 256 * 256);
    }
    if (total != cursor || total != (int64_t)capacities.size()) return false;

    // Every local layer's actual blob must fit every slot in its range; remote layers own no local slot.
    for (size_t layer = 0; layer < counts.size(); ++layer) {
        for (int64_t slot = ranges[layer].begin; slot < ranges[layer].end; ++slot) {
            size_t copy = 0;
            if (!strata::core::slot_copy_size(capacities[(size_t)slot], blob_bytes[layer], copy, err) ||
                copy != (size_t)blob_bytes[layer]) {
                std::fprintf(stderr, "layer %zu blob %lld does not fit its slot %lld (%lld B): %s\n", layer,
                             (long long)blob_bytes[layer], (long long)slot,
                             (long long)capacities[(size_t)slot], err.c_str());
                return false;
            }
        }
    }
    return true;
}
} // namespace

int main() {
    // Unequal native expert sizes and a remote first layer reproduce the reviewed out-of-bounds placement.
    const std::vector<int64_t> sizes{256, 1024, 256, 512};
    if (!check_layout({0, 4, 4, 4}, sizes)) return 1;
    if (!check_layout({0, 0, 0, 0}, sizes)) return 1; // all layers may be owned remotely

    // Exercise remote first, interior and last layers, then the smaller retry allocation for every placement.
    for (size_t remote = 0; remote < sizes.size(); ++remote) {
        std::vector<int64_t> full(sizes.size(), 4);
        full[remote] = 0;
        if (!check_layout(full, sizes)) return 1;
        std::vector<int64_t> retry(sizes.size(), 2);
        retry[remote] = 0;
        if (!check_layout(retry, sizes)) return 1;
    }

    size_t copied = 0;
    std::string err;
    if (strata::core::slot_copy_size(256, 1024, copied, err) || copied != 0) {
        std::fprintf(stderr, "oversized expert copy was accepted\n");
        return 1;
    }
    err.clear();
    if (strata::core::slot_copy_size(1024, -1, copied, err)) {
        std::fprintf(stderr, "negative expert copy size was accepted\n");
        return 1;
    }
    std::vector<uint8_t> remote(4 * 4, 0);
    remote[0 * 4 + 3] = 1; // sparse accepted pair in the first layer
    remote[2 * 4 + 1] = 1; // sparse accepted pair in an interior layer
    remote[3 * 4 + 0] = 1; // sparse accepted pair in the last layer
    if (!strata::core::is_remote_owned(remote, 4, 0, 3) ||
        !strata::core::is_remote_owned(remote, 4, 2, 1) ||
        !strata::core::is_remote_owned(remote, 4, 3, 0) ||
        strata::core::is_remote_owned(remote, 4, 1, 1) ||
        strata::core::is_remote_owned(remote, 4, 4, 1)) {
        std::fprintf(stderr, "sparse remote ownership mask filtered the wrong experts\n");
        return 1;
    }
    std::puts("expert cache layer ranges and slot capacities passed");
    return 0;
}
