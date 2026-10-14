#include "strata/core/expert_source.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {
class RecordingSource final : public strata::core::ExpertSource {
public:
    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    bool warms() const override { return true; }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<int64_t> selected;
        if (n > 0) selected.assign(experts, experts + n);
        calls_.push_back({layer, std::move(selected)});
        cv_.notify_all();
    }
    bool wait_calls(size_t n) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return calls_.size() >= n; });
    }
    std::vector<std::pair<int64_t, std::vector<int64_t>>> calls() {
        std::lock_guard<std::mutex> lock(mu_);
        return calls_;
    }
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::pair<int64_t, std::vector<int64_t>>> calls_;
};

uint16_t bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}
} // namespace

int main() {
    constexpr int64_t embd = 8, experts = 4, layers = 3;
    std::vector<std::vector<uint16_t>> routers((size_t)layers,
                                                std::vector<uint16_t>((size_t)(embd * experts), 0));
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (int64_t expert = 0; expert < experts; ++expert)
            routers[(size_t)layer][(size_t)(expert * embd)] = bf16((float)(experts - expert));
    }
    std::vector<uint8_t> remote((size_t)(layers * experts), 0);
    remote[(size_t)(1 * experts + 0)] = 1; // sparse remote pair in layer 1
    for (int64_t expert = 0; expert < experts; ++expert)
        remote[(size_t)(2 * experts + expert)] = 1; // whole remote block in layer 2

    RecordingSource source;
    strata::core::RouterLookahead lookahead;
    lookahead.set_remote_ownership(std::move(remote));
    std::string err;
    if (!lookahead.start(std::move(routers), embd, experts, 1, &source, err)) {
        std::fprintf(stderr, "lookahead failed to start: %s\n", err.c_str());
        return 1;
    }
    float x[embd]{};
    x[0] = 1.0f;
    int32_t host_res[layers * experts];
    for (int32_t& slot : host_res) slot = -1;

    lookahead.submit(0, x, 1, host_res);
    if (!source.wait_calls(1)) {
        std::fprintf(stderr, "local lookahead prediction did not warm\n");
        return 1;
    }
    auto last_skipped = lookahead.skipped();
    bool submitted = false;
    const auto submit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < submit_deadline) {
        lookahead.submit(1, x, 1, host_res);
        const auto skipped = lookahead.skipped();
        if (skipped == last_skipped) { submitted = true; break; }
        last_skipped = skipped;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!submitted) {
        std::fprintf(stderr, "lookahead worker stayed busy after its local prediction\n");
        return 1;
    }
    if (!source.wait_calls(2)) {
        std::fprintf(stderr, "remote-only lookahead prediction did not complete\n");
        return 1;
    }
    const auto calls = source.calls();
    if (calls.size() != 2 || calls[0].first != 1 || calls[0].second != std::vector<int64_t>{1} ||
        calls[1].first != 2 || !calls[1].second.empty()) {
        std::fprintf(stderr, "lookahead warmed a remote-owned expert or failed to replace it with the best local candidate\n");
        return 1;
    }
    std::puts("RPC ownership-filtered lookahead preserved local warming and skipped remote-only predictions");
    return 0;
}
