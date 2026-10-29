// Paired disk-session envelope. Each PC writes its own native slot; only Ref crosses the link.
#pragma once
#include "ds4_stage.hpp"
#include "ds4_os.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>

namespace strata::ds4::session {
struct File {
    stage::Spec spec;
    stage::SnapshotRef ref;
    bool worker = false;
    std::vector<stage::ImageSpan> images;
    std::vector<uint8_t> state;
};
inline std::string new_id() {
    std::random_device r;
    std::ostringstream s;
    s << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i) s << std::setw(8) << (uint32_t) r();
    return s.str();
}
inline uint64_t checksum(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
inline void put(std::vector<uint8_t>& b, uint64_t x) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (x >> (8 * i)));
}
inline std::vector<uint8_t> metadata(const File& f) {
    std::vector<uint8_t> b{'D','S','4','P','A','I','R', uint8_t(f.spec.middle_end ? '3' : '2')};
    if (f.spec.middle_end) put(b, f.spec.middle_end);
    for (uint64_t x : {f.spec.identity_a, f.spec.identity_b, (uint64_t) f.spec.n_layers, (uint64_t) f.spec.split,
                      (uint64_t) f.spec.dim, (uint64_t) f.spec.hc, (uint64_t) f.spec.vocab, f.spec.max_context,
                      (uint64_t) f.spec.wire, (uint64_t) (int64_t) f.spec.image_token, (uint64_t) f.spec.sessions,
                      (uint64_t) f.worker, (uint64_t) f.ref.position, f.ref.prefix}) put(b, x);
    b.insert(b.end(), f.ref.id.begin(), f.ref.id.end());
    put(b, f.images.size());
    for (const auto& im : f.images) { put(b, (uint64_t) im.start); put(b, (uint64_t) im.length); put(b, im.hash); }
    put(b, f.state.size());
    return b;
}
inline bool write(const std::string& path, const File& f, uint64_t& bytes, std::string& err, bool immutable = false) {
    namespace fs = std::filesystem;
    if (!stage::valid_snapshot(f.ref, f.spec.max_context) || !stage::valid_images(f.images, f.spec.max_context) || f.state.empty()) {
        err = "invalid snapshot state"; return false;
    }
    const fs::path dst(path), parent = dst.has_parent_path() ? dst.parent_path() : fs::path(".");
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec || fs::is_symlink(dst, ec) || (immutable && fs::exists(dst))) { err = "snapshot path is unavailable"; return false; }
    auto meta = metadata(f);
    const uint64_t size = meta.size() + f.state.size() + 8;
    const auto space = fs::space(parent, ec);
    if (ec || space.available < size + (64ull << 20)) { err = "not enough disk space for the snapshot"; return false; }
    const std::string tmp = path + "." + new_id() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write((const char*) meta.data(), (std::streamsize) meta.size());
    for (size_t at = 0; out && at < f.state.size(); at += 1u << 20) {
        const size_t n = std::min<size_t>(1u << 20, f.state.size() - at);
        out.write((const char*) f.state.data() + at, (std::streamsize) n);
    }
    uint64_t h = checksum(f.state.data(), f.state.size(), checksum(meta.data(), meta.size()));
    std::vector<uint8_t> tail; put(tail, h);
    out.write((const char*) tail.data(), 8);
    out.flush();
    const bool ok = (bool) out;
    out.close();
    if (!ok || out.fail() || !os::replace_file(tmp, path)) {
        fs::remove(tmp, ec); err = "snapshot write or atomic publish failed"; return false;
    }
    bytes = size;
    return true;
}
inline bool read(const std::string& path, File& f, uint64_t& bytes, std::string& err, uint64_t available = ~uint64_t{0}) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_symlink(fs::path(path), ec) || !fs::is_regular_file(fs::path(path), ec)) {
        err = "snapshot is not a regular local file"; return false;
    }
    const uint64_t size = fs::file_size(fs::path(path), ec);
    // Refuse before allocation. Leave at least 256 MiB available on top of the saved state.
    if (ec || size < 177 || size > (8ull << 30) ||
        (available != ~uint64_t{0} && (available < (256ull << 20) || size > available - (256ull << 20)))) {
        err = "snapshot size is invalid or there is not enough free RAM to restore it"; return false;
    }
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> b((size_t) size);
    in.read((char*) b.data(), (std::streamsize) b.size());
    if (!in || (std::memcmp(b.data(), "DS4PAIR2", 8) != 0 && std::memcmp(b.data(), "DS4PAIR3", 8) != 0)) {
        err = "snapshot is truncated or has the wrong format"; return false;
    }
    size_t at = 8;
    bool ok = true;
    const auto get = [&]() {
        uint64_t x = 0;
        if (at + 8 > b.size()) { ok = false; return x; }
        for (int i = 0; i < 8; ++i) x |= (uint64_t) b[at++] << (8 * i);
        return x;
    };
    f.spec.middle_end = b[7] == '3' ? (uint32_t) get() : 0;
    f.spec.identity_a = get(); f.spec.identity_b = get();
    f.spec.n_layers = (uint32_t) get(); f.spec.split = (uint32_t) get();
    f.spec.dim = (uint32_t) get(); f.spec.hc = (uint32_t) get(); f.spec.vocab = (uint32_t) get();
    f.spec.max_context = get(); f.spec.wire = (stage::LayerStageWireFormat) get();
    f.spec.image_token = (int32_t) get(); f.spec.sessions = get() != 0;
    f.worker = get() != 0; f.ref.position = (int64_t) get(); f.ref.prefix = get();
    f.ref.id.assign((const char*) b.data() + at, 32); at += 32;
    const uint64_t ni = get();
    if (ni > 4096 || at + ni * 24 + 16 > size) ok = false;
    if (ok) {
        f.images.resize((size_t) ni);
        for (auto& im : f.images) { im.start = (int64_t) get(); im.length = (int64_t) get(); im.hash = get(); }
    }
    const uint64_t n = ok ? get() : 0;
    if (!ok || n == 0 || n > size || at > size - n || size - at - n != 8 ||
        !stage::valid_snapshot(f.ref, f.spec.max_context) || !stage::valid_images(f.images, f.spec.max_context)) {
        err = "snapshot metadata is damaged"; return false;
    }
    const size_t data_at = at;
    at += (size_t) n;
    if (get() != checksum(b.data(), b.size() - 8)) { err = "snapshot checksum failed"; return false; }
    // Drop the small envelope in place; avoid holding a second full state during restore.
    b.erase(b.begin() + (std::ptrdiff_t) (data_at + n), b.end());
    b.erase(b.begin(), b.begin() + (std::ptrdiff_t) data_at);
    f.state = std::move(b);
    bytes = size;
    return true;
}
}  // namespace strata::ds4::session
