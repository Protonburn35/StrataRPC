#include "strata/core/remote_experts.hpp"
#include "strata/core/remote_expert_opt.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;
constexpr std::chrono::milliseconds kRpcInitializationTimeout{10 * 60 * 1000};
constexpr std::chrono::milliseconds kRpcInferenceTimeout{60 * 1000};
constexpr uint32_t kRpcVersion = 2;
constexpr uint32_t kRpcMaxPairs = 1000000;
constexpr uint32_t kRpcMaxMessage = 4096;
constexpr char kRpcMagic[8] = {'S','T','R','R','P','C','2','\0'};
struct RpcPair { uint32_t layer, expert; };
static_assert(sizeof(RpcPair) == 8, "RPC pair wire size changed");

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}

bool send_u32(rpc::Socket s, uint32_t v, const rpc::IoDeadline& deadline, std::string& err) {
    return rpc::send_all(s, &v, sizeof(v), deadline, err);
}
bool recv_u32(rpc::Socket s, uint32_t& v, const rpc::IoDeadline& deadline, std::string& err) {
    return rpc::recv_all(s, &v, sizeof(v), deadline, err);
}
bool send_u64(rpc::Socket s, uint64_t v, const rpc::IoDeadline& deadline, std::string& err) {
    return rpc::send_all(s, &v, sizeof(v), deadline, err);
}
bool recv_u64(rpc::Socket s, uint64_t& v, const rpc::IoDeadline& deadline, std::string& err) {
    return rpc::recv_all(s, &v, sizeof(v), deadline, err);
}

std::string rpc_token() {
    const char* t = std::getenv("STRATA_RPC_TOKEN");
    return t == nullptr ? std::string() : std::string(t);
}

bool send_text(rpc::Socket s, const std::string& v, const rpc::IoDeadline& deadline, std::string& err) {
    if (v.size() > kRpcMaxMessage) { err = "RPC error message is too long"; return false; }
    return send_u32(s, (uint32_t)v.size(), deadline, err) &&
           (v.empty() || rpc::send_all(s, v.data(), v.size(), deadline, err));
}

bool recv_text(rpc::Socket s, std::string& v, const rpc::IoDeadline& deadline, std::string& err) {
    uint32_t n = 0;
    if (!recv_u32(s, n, deadline, err)) return false;
    if (n > kRpcMaxMessage) { err = "RPC error message exceeds the protocol limit"; return false; }
    v.resize(n);
    return n == 0 || rpc::recv_all(s, v.data(), n, deadline, err);
}

bool send_rpc_error(rpc::Socket s, const std::string& message, const rpc::IoDeadline& deadline, std::string& err) {
    return send_u32(s, 1, deadline, err) && send_text(s, message, deadline, err);
}

uint64_t hash_blob(const uint8_t* p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
} // namespace

RemoteExperts::~RemoteExperts() { close(); }

void RemoteExperts::invalidate_rpc() {
    if (rpc_socket_ != rpc::kInvalidSocket) rpc::close(rpc_socket_);
    rpc_socket_ = rpc::kInvalidSocket;
    rpc_pending_ = false;
    rpc_broken_ = true;
}

bool RemoteExperts::open_rpc(const std::string& endpoint, int64_t layers, int64_t experts,
                             const std::vector<std::pair<int32_t, int32_t>>& ranked,
                             const ExpertCache& primary, ExpertSource& source,
                             std::vector<uint8_t>& claimed, std::string& err, bool require_all) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t total = (size_t)layers * (size_t)experts;
    if (layers <= 0 || experts <= 0 || total / (size_t)experts != (size_t)layers ||
        claimed.size() != total || ranked.empty() || layers > UINT32_MAX || experts > UINT32_MAX) {
        err = "RPC experts: invalid model geometry or empty ranking";
        return false;
    }
    const std::string token = rpc_token();
    if (token.empty() || token.size() > kRpcMaxMessage) {
        err = "RPC experts: set STRATA_RPC_TOKEN to the same nonempty secret on both machines (maximum 4096 bytes)";
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> candidates;
    candidates.reserve(ranked.size());
    std::vector<uint8_t> seen(total, 0);
    for (const auto& p : ranked) {
        if (p.first < 0 || p.first >= layers || p.second < 0 || p.second >= experts) continue;
        const size_t ix = (size_t)p.first * (size_t)experts + (size_t)p.second;
        if (claimed[ix] || seen[ix] || primary.slot_of(p.first, p.second) >= 0) continue;
        seen[ix] = 1;
        candidates.push_back(p);
    }
    if (candidates.empty() || candidates.size() > kRpcMaxPairs) {
        err = "RPC experts: no eligible ranked pairs, or the ranking exceeds the protocol limit";
        return false;
    }
    if (!rpc::connect(endpoint, rpc_socket_, err)) return false;
    rpc_client_ = true;
    n_expert_ = experts;
    if (!rpc::set_nodelay(rpc_socket_, true, err)) { close(); return false; }
    const rpc::IoDeadline init_deadline = rpc::deadline_after(kRpcInitializationTimeout);
    std::array<char, 8> magic{};
    std::memcpy(magic.data(), kRpcMagic, sizeof(kRpcMagic));
    bool ok = rpc::send_all(rpc_socket_, magic.data(), magic.size(), init_deadline, err) &&
              send_u32(rpc_socket_, kRpcVersion, init_deadline, err) &&
              send_u32(rpc_socket_, (uint32_t)layers, init_deadline, err) &&
              send_u32(rpc_socket_, (uint32_t)experts, init_deadline, err) &&
              send_text(rpc_socket_, token, init_deadline, err) &&
              send_u32(rpc_socket_, (uint32_t)candidates.size(), init_deadline, err);
    std::vector<RpcPair> wire_pairs;
    wire_pairs.reserve(candidates.size());
    for (const auto& p : candidates) wire_pairs.push_back({(uint32_t)p.first, (uint32_t)p.second});
    if (ok) ok = rpc::send_all(rpc_socket_, wire_pairs.data(), wire_pairs.size() * sizeof(RpcPair), init_deadline, err);
    std::vector<uint64_t> layer_bytes((size_t)layers);
    for (int64_t l = 0; l < layers; ++l) layer_bytes[(size_t)l] = lay.blob_bytes(l);
    if (ok) ok = rpc::send_all(rpc_socket_, layer_bytes.data(), layer_bytes.size() * sizeof(uint64_t), init_deadline, err);
    uint32_t status = 0;
    if (ok) ok = recv_u32(rpc_socket_, status, init_deadline, err);
    if (!ok) { close(); return false; }
    if (status != 0) {
        std::string message;
        if (recv_text(rpc_socket_, message, init_deadline, err)) err = message;
        if (err.empty()) err = "RPC experts: server rejected the model or cache request";
        close(); return false;
    }
    uint32_t count = 0;
    if (!recv_u32(rpc_socket_, count, init_deadline, err) || !recv_u64(rpc_socket_, remote_bytes_, init_deadline, err) ||
        count == 0 || count > candidates.size()) {
        if (err.empty()) err = "RPC experts: server returned an invalid resident-pair count";
        close(); return false;
    }
    remote_pairs_.reserve(count);
    remote_present_.assign(total, 0);
    std::vector<RpcPair> accepted(count);
    if (!rpc::recv_all(rpc_socket_, accepted.data(), accepted.size() * sizeof(RpcPair), init_deadline, err)) { close(); return false; }
    uint64_t accepted_bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t l = accepted[i].layer, e = accepted[i].expert;
        if (l >= (uint32_t)layers || e >= (uint32_t)experts) {
            err = "RPC experts: server returned an out-of-range resident pair"; close(); return false;
        }
        const size_t ix = (size_t)l * (size_t)experts + (size_t)e;
        if (!seen[ix] || remote_present_[ix] || claimed[ix] || primary.slot_of(l, e) >= 0) {
            err = "RPC experts: server returned a duplicate or locally resident pair"; close(); return false;
        }
        const uint64_t bytes = lay.blob_bytes(l);
        if (accepted_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
            err = "RPC experts: accepted cache size overflows"; close(); return false;
        }
        accepted_bytes += bytes;
        remote_present_[ix] = 1;
        claimed[ix] = 1;
        remote_pairs_.emplace_back((int32_t)l, (int32_t)e);
    }
    if (accepted_bytes != remote_bytes_) {
        err = "RPC experts: server's resident cache byte count does not match its accepted pairs";
        close(); return false;
    }
    if (require_all && count != candidates.size()) {
        err = "RPC experts: the remote GPU could not fit every expert in the requested complete layer blocks; "
              "choose fewer layers or close other GPU workloads";
        close(); return false;
    }
    uint32_t sample_count = 0;
    if (!recv_u32(rpc_socket_, sample_count, init_deadline, err) || sample_count == 0 || sample_count > 8 || sample_count > count) {
        if (err.empty()) err = "RPC experts: server returned an invalid pack-check sample count";
        close(); return false;
    }
    for (uint32_t i = 0; i < sample_count; ++i) {
        uint32_t l = 0, e = 0;
        uint64_t remote_hash = 0;
        if (!recv_u32(rpc_socket_, l, init_deadline, err) || !recv_u32(rpc_socket_, e, init_deadline, err) ||
            !recv_u64(rpc_socket_, remote_hash, init_deadline, err)) {
            close(); return false;
        }
        if (l >= (uint32_t)layers || e >= (uint32_t)experts || !remote_present_[(size_t)l * (size_t)experts + e]) {
            err = "RPC experts: server pack-check sample is outside its accepted cache"; close(); return false;
        }
        const uint8_t* blob = source.blob(l, e);
        if (blob == nullptr || hash_blob(blob, (size_t)lay.blob_bytes(l)) != remote_hash) {
            err = "RPC experts: sampled expert bytes differ; both PCs must use the same model pack";
            close(); return false;
        }
    }
    layers_present_.assign((size_t)layers, 0);
    for (const auto& p : remote_pairs_) layers_present_[(size_t)p.first] = 1;
    device_ = -2;
    owned_.assign((size_t)CAP, 0);
    rpc_kind_.resize((size_t)CAP);
    original_row_.reserve((size_t)CAP);
    returned_bytes_ = full_row_bytes_ = 0;
    rpc_sent_bytes_ = rpc_received_bytes_ = 0;
    computed_ = launched_layers_ = 0;
    ms_begin_ = ms_wait_ = 0;
    std::fprintf(stderr, "strata: RPC helper accepted %u experts (%.2f GiB) at %s\n", count,
                 (double)remote_bytes_ / 1073741824.0, endpoint.c_str());
    return true;
}

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (rpc_socket_ != rpc::kInvalidSocket) rpc::close(rpc_socket_);
    rpc_socket_ = rpc::kInvalidSocket;
    rpc_pending_ = false;
    rpc_broken_ = false;
    rpc_cancelled_.store(nullptr, std::memory_order_release);
    rpc_n_tok_ = 0;
    rpc_reduce_pending_ = false;
    if (device_ >= 0) {
        DeviceScope scope(device_);
        if (scope.ok) {
            if (stream_) cudaStreamSynchronize(stream_);
            if (read_stream_) cudaStreamSynchronize(read_stream_);
            cache_.close();
            if (d_x_) cudaFree(d_x_);
            if (d_out_) cudaFree(d_out_);
            if (d_q8_) cudaFree(d_q8_);
            if (d_scales_) cudaFree(d_scales_);
            if (d_scratch_) cudaFree(d_scratch_);
            if (d_meta_) cudaFree(d_meta_);
            if (h_x_) cudaFreeHost(h_x_);
            if (h_out_) cudaFreeHost(h_out_);
            if (h_meta_) cudaFreeHost(h_meta_);
            if (stream_) cudaStreamDestroy(stream_);
            if (read_stream_) cudaStreamDestroy(read_stream_);
        }
    }
    device_ = -1;
    rpc_client_ = false;
    remote_bytes_ = 0;
    rpc_sent_bytes_ = rpc_received_bytes_ = 0;
    remote_present_.clear();
    remote_pairs_.clear();
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size) {
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts) {
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + (512ull << 20) > free_bytes) break;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() : 0);
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device);
    if (!allocated) { close(); return false; }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    remote_pairs_ = selected;
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    if (rpc_client_) {
        if (rpc_broken_ || rpc_socket_ == rpc::kInvalidSocket) {
            err = "RPC experts: the connection was invalidated after an incomplete or invalid response; restart the request";
            return false;
        }
        if (n_tok <= 0 || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
            (size_t)layer >= layers_present_.size() || rpc_pending_) {
            err = "RPC experts: invalid layer, routing width, window size or request ordering";
            return false;
        }
        const int64_t n = n_tok * k;
        std::fill(owned_.begin(), owned_.begin() + n, 0);
        original_row_.clear();
        if (!layers_present_[(size_t)layer]) return true;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
                (primary_res && primary_res[(size_t)layer * (size_t)n_expert_ + (size_t)e] >= 0) ||
                !remote_present_[(size_t)layer * (size_t)n_expert_ + (size_t)e]) continue;
            owned_[(size_t)i] = 1;
            original_row_.push_back((int32_t)i);
        }
        if (original_row_.empty()) return true;
        if (original_row_.size() > UINT32_MAX || n_tok > UINT32_MAX || k > UINT32_MAX) {
            err = "RPC experts: request is larger than the protocol limit";
            return false;
        }
        const bool reduce = remote_opt_ != nullptr && remote_opt_->active();
        const uint32_t header[4] = {reduce ? 2u : 1u, (uint32_t)layer, (uint32_t)n_tok, (uint32_t)k};
        const size_t x_bytes = (size_t)n_tok * H * sizeof(float);
        const size_t route_bytes = (size_t)n * sizeof(int32_t);
        const size_t weight_bytes = reduce ? (size_t)n * sizeof(float) : 0;
        rpc_deadline_ = rpc::deadline_after(kRpcInferenceTimeout,
                                            rpc_cancelled_.load(std::memory_order_acquire));
        // The server must compute only rows assigned to it by this host window. In particular, a normal adaptive
        // host cache may have admitted a remote expert since startup; forwarding its original kind would make the
        // server return an extra row and break the response count/order check.
        for (int64_t i = 0; i < n; ++i) rpc_kind_[(size_t)i] = owned_[(size_t)i] ? -1 : 0;
        const bool sent = rpc::send_all(rpc_socket_, header, sizeof(header), rpc_deadline_, err) &&
                          rpc::send_all(rpc_socket_, x, x_bytes, rpc_deadline_, err) &&
                          rpc::send_all(rpc_socket_, ids, route_bytes, rpc_deadline_, err) &&
                          rpc::send_all(rpc_socket_, rpc_kind_.data(), route_bytes, rpc_deadline_, err) &&
                          (!reduce || rpc::send_all(rpc_socket_, remote_opt_->weights_, weight_bytes, rpc_deadline_, err));
        if (!sent) { invalidate_rpc(); return false; }
        rpc_sent_bytes_ += sizeof(header) + x_bytes + route_bytes * 2 + weight_bytes;
        rpc_pending_ = true;
        rpc_n_tok_ = (uint32_t)n_tok;
        rpc_reduce_pending_ = reduce;
        computed_ += (int64_t)original_row_.size();
        full_row_bytes_ += (uint64_t)n * H * sizeof(float);
        ++launched_layers_;
        return true;
    }
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(zero_copy_ ? z_x_ : d_x_, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t) layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (rpc_client_) {
        if (!rpc_pending_) return true;
        const auto w0 = std::chrono::steady_clock::now();
        uint32_t status = 0, count = 0;
        err.clear();
        auto fail = [&](const char* fallback) {
            if (err.empty()) err = fallback;
            invalidate_rpc();
            return false;
        };
        if (!recv_u32(rpc_socket_, status, rpc_deadline_, err))
            return fail("RPC experts: failed while waiting for the remote result status");
        if (status != 0) {
            std::string message;
            if (recv_text(rpc_socket_, message, rpc_deadline_, err)) err = "RPC experts: " + message;
            return fail("RPC experts: server failed to compute the routed rows");
        }
        const uint32_t expected = rpc_reduce_pending_ ? rpc_n_tok_ : (uint32_t)original_row_.size();
        if (!recv_u32(rpc_socket_, count, rpc_deadline_, err))
            return fail("RPC experts: failed while waiting for the remote result count");
        if (count != expected) { err = "RPC experts: server returned an unexpected result count"; return fail(""); }
        rpc_out_.resize((size_t)count * H);
        if (rpc_reduce_pending_) {
            if (!rpc::recv_all(rpc_socket_, rpc_out_.data(), rpc_out_.size() * sizeof(float), rpc_deadline_, err))
                return fail("RPC experts: incomplete reduced-result frame");
            if (remote_opt_ == nullptr) { err = "RPC experts: optimized result arrived without an active reducer"; return fail(""); }
            const int token_begin = remote_opt_->token_begin_;
            if ((uint32_t)remote_opt_->tokens_ != rpc_n_tok_) {
                err = "RPC experts: reduced window size differs from the active verifier window";
                return fail("");
            }
            for (size_t i = 0; i < (size_t)count * H; ++i)
                remote_opt_->h_sum_[(size_t)token_begin * H + i] += rpc_out_[i];
            for (size_t i = 0; i < owned_.size() && i < (size_t)rpc_n_tok_ * 10; ++i)
                remote_opt_->h_mask_[(size_t)token_begin * 10 + i] |= owned_[i] != 0;
            rpc_received_bytes_ += sizeof(status) + sizeof(count) + rpc_out_.size() * sizeof(float);
        } else {
            rpc_rows_.resize(count);
            if (!rpc::recv_all(rpc_socket_, rpc_rows_.data(), rpc_rows_.size() * sizeof(uint32_t), rpc_deadline_, err) ||
                !rpc::recv_all(rpc_socket_, rpc_out_.data(), rpc_out_.size() * sizeof(float), rpc_deadline_, err))
                return fail("RPC experts: incomplete raw-result frame");
            for (uint32_t q = 0; q < count; ++q) {
                const uint32_t row = rpc_rows_[q];
                if (q >= original_row_.size() || row != (uint32_t)original_row_[q]) {
                    err = "RPC experts: server returned rows in an invalid order";
                    return fail("");
                }
            }
            for (uint32_t q = 0; q < count; ++q) {
                const uint32_t row = rpc_rows_[q];
                std::memcpy(out + (size_t)row * H, rpc_out_.data() + (size_t)q * H, (size_t)H * sizeof(float));
            }
            rpc_received_bytes_ += sizeof(status) + sizeof(count) + rpc_rows_.size() * sizeof(uint32_t) +
                                   rpc_out_.size() * sizeof(float);
        }
        returned_bytes_ += (uint64_t)count * H * sizeof(float);
        ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        rpc_pending_ = false;
        rpc_n_tok_ = 0;
        rpc_reduce_pending_ = false;
        return true;
    }
    if (group_id_.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

bool RemoteExperts::serve_rpc(const std::string& endpoint, const std::string& pack_dir, const std::string& native_gguf,
                              int64_t layers, int64_t experts, std::string& err) {
    const std::string token = rpc_token();
    if (token.empty() || token.size() > kRpcMaxMessage) {
        err = "RPC server: set STRATA_RPC_TOKEN to a nonempty secret on both machines (maximum 4096 bytes)";
        return false;
    }
    FileExpertSource source;
    if (!native_gguf.empty()) source.set_gguf(native_gguf);
    else if (!std::filesystem::exists(pack_dir + "/experts.bin")) {
        err = "RPC server: this native pack has no experts.bin; pass --native with the model's first GGUF shard";
        return false;
    }
    if (!source.open(pack_dir, layers, experts, err)) return false;
    rpc::Socket listener = rpc::kInvalidSocket;
    if (!rpc::listen(endpoint, listener, err)) return false;
    std::fprintf(stderr, "strata: RPC expert server listening on %s; waiting for one client\n", endpoint.c_str());
    rpc::Socket client = rpc::kInvalidSocket;
    if (!rpc::accept(listener, client, err)) { rpc::close(listener); return false; }
    rpc::close(listener);
    if (!rpc::set_nodelay(client, true, err)) { rpc::close(client); return false; }
    const rpc::IoDeadline init_deadline = rpc::deadline_after(kRpcInitializationTimeout);
    auto fail = [&](const std::string& why, const rpc::IoDeadline& deadline) {
        std::string ignored;
        send_rpc_error(client, why, deadline, ignored);
        rpc::close(client);
        err = why;
        return false;
    };
    std::array<char, 8> magic{};
    uint32_t version = 0, client_layers = 0, client_experts = 0, pair_count = 0;
    std::string client_token;
    if (!rpc::recv_all(client, magic.data(), magic.size(), init_deadline, err)) {
        rpc::close(client);
        return false;
    }
    if (std::memcmp(magic.data(), kRpcMagic, sizeof(kRpcMagic)) != 0)
        return fail("RPC server: invalid protocol magic", init_deadline);
    if (!recv_u32(client, version, init_deadline, err) || !recv_u32(client, client_layers, init_deadline, err) ||
        !recv_u32(client, client_experts, init_deadline, err) || !recv_text(client, client_token, init_deadline, err) ||
        !recv_u32(client, pair_count, init_deadline, err)) {
        rpc::close(client);
        return false;
    }
    if (version != kRpcVersion) return fail("RPC server: protocol version mismatch", init_deadline);
    if (client_token != token) return fail("RPC server: STRATA_RPC_TOKEN authentication failed", init_deadline);
    if (client_layers != (uint32_t)layers || client_experts != (uint32_t)experts ||
        pair_count == 0 || pair_count > kRpcMaxPairs)
        return fail("RPC server: client model geometry or ranking is invalid", init_deadline);

    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve(pair_count);
    std::vector<uint8_t> pair_seen((size_t)layers * (size_t)experts, 0);
    std::vector<RpcPair> candidates(pair_count);
    if (!rpc::recv_all(client, candidates.data(), candidates.size() * sizeof(RpcPair), init_deadline, err)) {
        rpc::close(client); return false;
    }
    for (uint32_t i = 0; i < pair_count; ++i) {
        const uint32_t l = candidates[i].layer, e = candidates[i].expert;
        if (l >= (uint32_t)layers || e >= (uint32_t)experts)
            return fail("RPC server: client ranking contains an invalid pair", init_deadline);
        const size_t ix = (size_t)l * (size_t)experts + (size_t)e;
        if (pair_seen[ix]) return fail("RPC server: client ranking contains a duplicate pair", init_deadline);
        pair_seen[ix] = 1;
        ranked.emplace_back((int32_t)l, (int32_t)e);
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint64_t> client_layer_bytes((size_t)layers);
    if (!rpc::recv_all(client, client_layer_bytes.data(), client_layer_bytes.size() * sizeof(uint64_t), init_deadline, err)) {
        rpc::close(client); return false;
    }
    for (int64_t l = 0; l < layers; ++l) {
        if (client_layer_bytes[(size_t)l] != lay.blob_bytes(l))
            return fail("RPC server: expert layout differs from the client pack", init_deadline);
    }

    ExpertCache primary;
    std::vector<uint8_t> claimed((size_t)layers * (size_t)experts, 0);
    RemoteExperts worker;
    RemoteExpertOpt server_opt;
    server_opt.attach(worker);
    if (ranked.size() > (size_t)std::numeric_limits<int>::max() ||
        !worker.open(0, (int)ranked.size(), layers, experts, ranked, primary, source, claimed, err, true))
        return fail(err.empty() ? "RPC server: GPU could not open the requested cache" : err, init_deadline);
    if (!server_opt.init(err))
        return fail(err.empty() ? "RPC server: weighted reduction buffers could not be allocated" : err, init_deadline);
    worker.remote_pairs_.clear();
    for (const auto& p : ranked)
        if (worker.cache_.slot_of(p.first, p.second) >= 0) worker.remote_pairs_.push_back(p);
    uint64_t cache_bytes = 0;
    for (const auto& p : worker.remote_pairs_) cache_bytes += lay.blob_bytes(p.first);
    std::vector<std::pair<size_t, uint64_t>> samples;
    const size_t sample_count = std::min<size_t>(8, worker.remote_pairs_.size());
    samples.reserve(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        const size_t at = i * worker.remote_pairs_.size() / sample_count;
        const auto& pair = worker.remote_pairs_[at];
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (blob == nullptr) return fail("RPC server: could not read a pack-check expert", init_deadline);
        samples.emplace_back(at, hash_blob(blob, (size_t)lay.blob_bytes(pair.first)));
    }
    if (!send_u32(client, 0, init_deadline, err) ||
        !send_u32(client, (uint32_t)worker.remote_pairs_.size(), init_deadline, err) ||
        !send_u64(client, cache_bytes, init_deadline, err)) {
        rpc::close(client); return false;
    }
    std::vector<RpcPair> accepted;
    accepted.reserve(worker.remote_pairs_.size());
    for (const auto& p : worker.remote_pairs_) accepted.push_back({(uint32_t)p.first, (uint32_t)p.second});
    if (!rpc::send_all(client, accepted.data(), accepted.size() * sizeof(RpcPair), init_deadline, err)) {
        rpc::close(client); return false;
    }
    if (!send_u32(client, (uint32_t)samples.size(), init_deadline, err)) { rpc::close(client); return false; }
    for (const auto& sample : samples) {
        const auto& p = worker.remote_pairs_[sample.first];
        if (!send_u32(client, (uint32_t)p.first, init_deadline, err) ||
            !send_u32(client, (uint32_t)p.second, init_deadline, err) ||
            !send_u64(client, sample.second, init_deadline, err)) { rpc::close(client); return false; }
    }
    std::fprintf(stderr, "strata: RPC expert client connected; cached %zu experts (%.2f GiB) on CUDA0\n",
                 worker.remote_pairs_.size(), (double)cache_bytes / 1073741824.0);

    std::vector<float> x((size_t)strata::kernels::cpu::MAXT * H);
    std::vector<int32_t> ids((size_t)strata::kernels::cpu::MAXT * 10);
    std::vector<int32_t> kind((size_t)strata::kernels::cpu::MAXT * 10);
    std::vector<float> out((size_t)strata::kernels::cpu::MAXT * 10 * H);
    std::vector<float> weights((size_t)strata::kernels::cpu::MAXT * 10);
    std::vector<uint32_t> row_ids;
    std::vector<float> compact;
    row_ids.reserve((size_t)strata::kernels::cpu::MAXT * 10);
    compact.reserve((size_t)strata::kernels::cpu::MAXT * 10 * H);
    for (;;) {
        uint32_t header[4]{};
        err.clear();
        uint8_t first_header_byte = 0;
        if (!rpc::recv_all(client, &first_header_byte, sizeof(first_header_byte), rpc::wait_forever(), err)) {
            const bool disconnected = err == "RPC connection closed";
            rpc::close(client);
            return disconnected;
        }
        const rpc::IoDeadline op_deadline = rpc::deadline_after(kRpcInferenceTimeout);
        reinterpret_cast<uint8_t*>(header)[0] = first_header_byte;
        if (!rpc::recv_all(client, reinterpret_cast<uint8_t*>(header) + 1, sizeof(header) - 1, op_deadline, err)) {
            rpc::close(client);
            return false;
        }
        const uint32_t type = header[0], layer = header[1], ntok = header[2], k = header[3];
        if (type != 1 && type != 2) return fail("RPC server: unknown request type", op_deadline);
        if (layer >= (uint32_t)layers || ntok == 0 || ntok > (uint32_t)strata::kernels::cpu::MAXT || k != 10) {
            return fail("RPC server: request has an invalid layer or window shape", op_deadline);
        }
        const size_t n = (size_t)ntok * k;
        if (!rpc::recv_all(client, x.data(), (size_t)ntok * H * sizeof(float), op_deadline, err) ||
            !rpc::recv_all(client, ids.data(), n * sizeof(int32_t), op_deadline, err) ||
            !rpc::recv_all(client, kind.data(), n * sizeof(int32_t), op_deadline, err)) {
            rpc::close(client); return false;
        }
        if (type == 2 && !rpc::recv_all(client, weights.data(), n * sizeof(float), op_deadline, err)) {
            rpc::close(client); return false;
        }
        std::fill(out.begin(), out.begin() + n * H, 0.0f);
        if (type == 2) server_opt.begin(weights.data(), 0, (int)ntok);
        if (!worker.begin(layer, x.data(), ids.data(), ntok, k, kind.data(), nullptr, err) ||
            !worker.finish(out.data(), err))
            return fail(err.empty() ? "RPC server: expert computation failed" : err, op_deadline);
        if (type == 2) {
            server_opt.end();
            const uint32_t response[2] = {0, ntok};
            if (!rpc::send_all(client, response, sizeof(response), op_deadline, err) ||
                !rpc::send_all(client, server_opt.h_sum_, (size_t)ntok * H * sizeof(float), op_deadline, err)) {
                rpc::close(client); return false;
            }
        } else {
            uint32_t rows = 0;
            for (size_t i = 0; i < n; ++i) rows += worker.owns((int64_t)i) ? 1u : 0u;
            row_ids.clear();
            compact.clear();
            for (size_t i = 0; i < n; ++i) {
                if (!worker.owns((int64_t)i)) continue;
                row_ids.push_back((uint32_t)i);
                compact.insert(compact.end(), out.data() + i * H, out.data() + (i + 1) * H);
            }
            const uint32_t response[2] = {0, rows};
            if (!rpc::send_all(client, response, sizeof(response), op_deadline, err) ||
                !rpc::send_all(client, row_ids.data(), row_ids.size() * sizeof(uint32_t), op_deadline, err) ||
                !rpc::send_all(client, compact.data(), compact.size() * sizeof(float), op_deadline, err)) {
                rpc::close(client); return false;
            }
        }
    }
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (rpc_client_) return true;
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    std::lock_guard<std::mutex> lock(read_mu_);
    bool queued = false;
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = cache_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        const size_t bytes = (size_t) lay.blob_bytes(layers[i]);
        if (!check(cudaMemcpyAsync(dst[i], cache_.device_slot(slot), bytes, cudaMemcpyDeviceToHost, read_stream_),
                   "cache read", err, device_)) {
            // Nothing may still be writing a caller's buffer when this returns.  If the drain succeeds, the
            // hits queued before the failure are complete and the caller may keep them; if it does not,
            // forget every hit and let the caller refill all of them from its fallback.
            if (cudaStreamSynchronize(read_stream_) != cudaSuccess) std::fill(hit, hit + n, 0);
            return false;
        }
        hit[i] = 1;
        queued = true;
    }
    if (queued && !check(cudaStreamSynchronize(read_stream_), "cache read wait", err, device_)) {
        std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

bool RemotePrefillSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if (helpers_.empty() || n == 0) return fallback_.copy_blobs(layers, experts, dst, n);
    thread_local std::vector<int32_t> hl, he;
    thread_local std::vector<uint8_t*> hd;
    thread_local std::vector<size_t> idx;
    thread_local std::vector<uint8_t> got;
    thread_local std::vector<uint8_t> served;
    thread_local std::vector<int32_t> ml, me;
    thread_local std::vector<uint8_t*> md;
    thread_local std::vector<int> which;
    which.assign(n, -1);
    served.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        for (size_t k = 0; k < helpers_.size(); ++k)
            if (helpers_[k]->holds(layers[i], experts[i])) { which[i] = (int) k; break; }
    }
    // One stream batch per helper, so several hits pay one wait; the misses go to the file source in one
    // call, which keeps its own batched-read merging for them.
    uint64_t blobs = 0, bytes = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t k = 0; k < helpers_.size(); ++k) {
        hl.clear(); he.clear(); hd.clear(); idx.clear();
        for (size_t i = 0; i < n; ++i)
            if (which[i] == (int) k) {
                hl.push_back(layers[i]); he.push_back(experts[i]); hd.push_back(dst[i]); idx.push_back(i);
            }
        if (hl.empty()) continue;
        got.assign(hl.size(), 0);
        std::string err;
        if (!helpers_[k]->copy_cached_blobs(hl.data(), he.data(), hd.data(), hl.size(), got.data(), err)) warn(err);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j]) {
                served[idx[j]] = 1;
                ++blobs;
                bytes += lay.blob_bytes(hl[j]);
            }
    }
    if (blobs != 0) {
        helper_blobs_.fetch_add(blobs, std::memory_order_relaxed);
        helper_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    ml.clear(); me.clear(); md.clear();
    for (size_t i = 0; i < n; ++i)
        if (!served[i]) { ml.push_back(layers[i]); me.push_back(experts[i]); md.push_back(dst[i]); }
    if (ml.empty()) return true;
    return fallback_.copy_blobs(ml.data(), me.data(), md.data(), ml.size());
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}
} // namespace strata::core
