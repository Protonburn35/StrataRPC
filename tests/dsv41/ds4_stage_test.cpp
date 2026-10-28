// The DeepSeek layer-stage protocol (src/dsv41/ds4_stage.cpp) over loopback TCP, without a GPU: a fake worker model that
// takes its rows in its own chunk size (unlike the coordinator's pieces), checks every row it gets, and keeps the state
// a Model keeps (position, checkpoints).
#include "ds4_stage.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <map>
#include <vector>

using namespace strata::ds4::stage;
namespace rpc = strata::core::rpc;

namespace {

int failures = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c);   \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)

constexpr uint32_t kDim = 24, kHc = 4, kVocab = 1000;

// the row values the coordinator sends for position p (h then pre), so the worker can check what it got
float h_value(int64_t p, size_t i) { return (float) std::sin(0.37 * (double) p + 0.011 * (double) i) * 3.0f + 0.5f; }
float pre_value(int64_t p, int i) { return 0.125f * (float) i - (float) (p % 7); }

void make_rows(int64_t p0, int T, std::vector<float>& h, std::vector<float>& pre) {
    h.resize((size_t) T * kHc * kDim);
    pre.resize((size_t) T * kPreFloats);
    for (int t = 0; t < T; ++t) {
        for (size_t i = 0; i < (size_t) kHc * kDim; ++i) h[(size_t) t * kHc * kDim + i] = h_value(p0 + t, i);
        for (int i = 0; i < kPreFloats; ++i) pre[(size_t) t * kPreFloats + i] = pre_value(p0 + t, i);
    }
}

struct FakeModel : Backend {
    int chunk = 300;           // rows the fake asks for at a time
    float tolerance = 0.0f;    // 0: the rows must be exact (FP32)
    int fail_after = -1;       // >= 0: the feed fails after taking this many rows
    std::atomic<int> resets{0};
    int64_t p = 0;
    std::vector<int32_t> toks;
    int64_t ck[4] = {-1, -1, -1, -1};
    int64_t rows_seen = 0, bad_rows = 0;
    int last_logits_rows = 0;
    double sum = 0;            // what the rows add up to: the "logits" the fake samples from

    std::vector<ImageSpan> image_spans;
    struct Saved { int64_t pos; double sum; std::vector<int32_t> tokens; std::vector<ImageSpan> images; };
    std::map<std::string, Saved> snapshots;
    void images(const std::vector<ImageSpan>& v) override { image_spans = v; }
    bool snapshot(bool save, const SnapshotRef& ref, uint64_t& bytes, std::string& err) override {
        if (save) {
            if (ref.position != p || ref.prefix != prefix_hash(toks)) { err = "prefix mismatch"; return false; }
            snapshots[ref.id] = {p, sum, toks, image_spans};
        } else {
            const auto it = snapshots.find(ref.id);
            if (it == snapshots.end() || ref.position != it->second.pos || ref.prefix != prefix_hash(it->second.tokens)) {
                err = "missing or mismatched snapshot"; return false;
            }
            p = it->second.pos; sum = it->second.sum; toks = it->second.tokens; image_spans = it->second.images;
        }
        bytes = toks.size() * 4; return true;
    }
    int64_t pos() const override { return p; }
    void reset() override {
        ++resets;
        p = 0;
        toks.clear();
        image_spans.clear();
        for (auto& c : ck) c = -1;
        sum = 0;
    }
    void checkpoint(int slot) override { ck[slot & 3] = p; }
    bool restore(int64_t limit) override {
        int b = -1;
        for (int i = 0; i < 4; ++i)
            if (ck[i] >= 0 && ck[i] <= limit && ck[i] <= p && (b < 0 || ck[i] > ck[b])) b = i;
        if (b < 0) return false;
        p = ck[b];
        toks.resize((size_t) p);
        for (auto& c : ck)
            if (c > p) c = -1;
        return true;
    }
    void pin() override {}
    bool feed(const int32_t* t, int n, int logits_rows, int snap_slot, const RowFn& rows, std::string& err) override {
        std::vector<float> h, pre;
        int taken = 0;
        for (int a = 0; a < n; a += chunk) {
            const int T = std::min(chunk, n - a);
            if (fail_after >= 0 && taken >= fail_after) {
                err = "the fake failed";
                return false;
            }
            h.assign((size_t) T * kHc * kDim, 0.0f);
            pre.assign((size_t) T * kPreFloats, 0.0f);
            if (!rows(p + a, T, h.data(), pre.data(), err)) return false;
            taken += T;
            for (int r = 0; r < T; ++r) {
                bool ok = true;
                for (size_t i = 0; i < (size_t) kHc * kDim; ++i) {
                    const float want = h_value(p + a + r, i), got = h[(size_t) r * kHc * kDim + i];
                    if (tolerance == 0.0f ? got != want : std::fabs(got - want) > tolerance * std::fabs(want) + 1e-6f)
                        ok = false;
                    sum += got;
                }
                for (int i = 0; i < kPreFloats; ++i)
                    if (pre[(size_t) r * kPreFloats + i] != pre_value(p + a + r, i)) ok = false;
                ++rows_seen;
                if (!ok) ++bad_rows;
            }
        }
        if (snap_slot >= 0) ck[snap_slot & 3] = p + n - 1;
        toks.insert(toks.end(), t, t + n);
        p += n;
        last_logits_rows = logits_rows;
        return true;
    }
    int sample(const Sampling& s) override {
        return (int) ((uint64_t) std::llround(std::fabs(sum)) + s.seed + s.counter + (uint64_t) toks.size()) % kVocab;
    }
    bool logits(std::vector<float>& out) override {
        out.assign(kVocab, 0.0f);
        out[(size_t) p % kVocab] = 1.0f;
        return true;
    }
};

Spec make_spec(LayerStageWireFormat wire) {
    Spec s;
    hash_text("fake model", s.identity_a, s.identity_b);
    s.n_layers = 40;
    s.split = 20;
    s.dim = kDim;
    s.hc = kHc;
    s.vocab = kVocab;
    s.max_context = 1 << 20;
    s.wire = wire;
    return s;
}

// a worker on its own thread: accepts connections one after another until stop
struct Worker {
    FakeModel model;
    Server server;
    rpc::Socket ls = rpc::kInvalidSocket;
    std::thread th;
    std::atomic<bool> stop{false};
    std::string endpoint;
    std::vector<std::string> ended;   // why each connection ended

    Worker(const Spec& spec, int port) : server("secret", spec, model) {
        endpoint = "127.0.0.1:" + std::to_string(port);
        std::string err;
        if (!rpc::listen(endpoint, ls, err)) {
            std::fprintf(stderr, "listen %s: %s\n", endpoint.c_str(), err.c_str());
            std::exit(2);
        }
        th = std::thread([this] {
            while (!stop.load()) {
                rpc::Socket c = rpc::kInvalidSocket;
                std::string e;
                if (!rpc::accept(ls, c, e)) continue;
                if (stop.load()) {
                    rpc::close(c);
                    break;
                }
                server.serve(c, e);
                ended.push_back(e);
            }
        });
    }
    ~Worker() {
        stop.store(true);
        rpc::Socket c = rpc::kInvalidSocket;   // (wakes the accept)
        std::string e;
        rpc::connect(endpoint, c, e);
        rpc::close(c);
        th.join();
        rpc::close(ls);
    }
};

int port_base() { return 39000 + (int) (std::chrono::steady_clock::now().time_since_epoch().count() % 2000); }

// streams n rows from p0 in pieces of `piece`, as the coordinator's last layer would
bool stream(Client& c, int64_t p0, int n, int piece, std::string& err) {
    std::vector<float> h, pre;
    for (int a = 0; a < n; a += piece) {
        const int T = std::min(piece, n - a);
        make_rows(p0 + a, T, h, pre);
        if (!c.prompt_rows(p0 + a, T, h.data(), pre.data(), err)) return false;
    }
    return true;
}

std::vector<int32_t> tokens(int n, int salt) {
    std::vector<int32_t> t((size_t) n);
    for (int i = 0; i < n; ++i) t[(size_t) i] = (i * 31 + salt) % (int) kVocab;
    return t;
}

void test_fp32(int port) {
    const Spec spec = make_spec(LayerStageWireFormat::fp32);
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    // a prompt: the coordinator's pieces (700) unlike the worker's chunks (300), the last token into checkpoint 1
    const int n = 2345;
    std::vector<int32_t> t = tokens(n, 1);
    CHECK(c.prompt_begin(0, t.data(), n, 1, 1, err));
    CHECK(stream(c, 0, n, 700, err));
    int64_t pos = -1;
    CHECK(c.prompt_end(false, pos, err));
    CHECK(pos == n);
    CHECK(w.model.rows_seen == n);
    CHECK(w.model.bad_rows == 0);
    CHECK(w.model.ck[1] == n - 1);
    CHECK(w.model.last_logits_rows == 1);
    // a step: one window
    std::vector<float> h, pre;
    make_rows(n, 1, h, pre);
    const int32_t tok = 7;
    CHECK(c.window(n, &tok, 1, 1, h.data(), pre.data(), pos, err));
    CHECK(pos == n + 1);
    CHECK(w.model.bad_rows == 0);
    int s1 = -1, s2 = -1;
    Sampling sp;
    sp.temperature = 1.0f;
    sp.seed = 5;
    CHECK(c.sample(sp, s1, err));
    CHECK(c.sample(sp, s2, err));
    CHECK(s1 >= 0 && s1 == s2);
    std::vector<float> lg;
    CHECK(c.logits(lg, err));
    CHECK(lg.size() == kVocab && lg[(size_t) (n + 1) % kVocab] == 1.0f);
    // checkpoints and restore
    CHECK(c.checkpoint(0, n + 1, err));
    CHECK(w.model.ck[0] == n + 1);
    bool restored = false;
    CHECK(c.restore(n, restored, pos, err));
    CHECK(restored && pos == n - 1);   // (checkpoint 1, the one at or before n)
    CHECK(c.pin(err));
    CHECK(c.restore(5, restored, pos, err));
    CHECK(!restored && pos == n - 1);
    // a window at the wrong position is refused, and the connection closes
    CHECK(!c.window(0, &tok, 1, 1, h.data(), pre.data(), pos, err));
    CHECK(!c.connected());
    // connected again: the worker starts over at 0
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(w.model.p == 0);
    CHECK(c.reset(err));
    std::printf("fp32: %.1f MB sent, %.1f ms waiting\n", (double) c.bytes_sent() / 1e6, c.wait_ms());
}

void test_bf16_and_halt(int port) {
    const Spec spec = make_spec(LayerStageWireFormat::bf16);
    Worker w(spec, port);
    w.model.tolerance = 1.0f / 128.0f;   // (BF16: 8 bits of mantissa)
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    const int n = 1000;
    std::vector<int32_t> t = tokens(n, 2);
    CHECK(c.prompt_begin(0, t.data(), n, 1, -1, err));
    CHECK(stream(c, 0, n, 256, err));
    int64_t pos = -1;
    CHECK(c.prompt_end(false, pos, err));
    CHECK(pos == n && w.model.bad_rows == 0 && w.model.rows_seen == n);
    CHECK(c.checkpoint(0, n, err));
    // a prompt the coordinator stops after 400 of 900 rows: the worker's feed ends, the connection stays
    std::vector<int32_t> t2 = tokens(900, 3);
    CHECK(c.prompt_begin(n, t2.data(), 900, 1, -1, err));
    CHECK(stream(c, n, 400, 200, err));
    CHECK(!c.prompt_end(true, pos, err));
    CHECK(c.connected());
    bool restored = false;
    CHECK(c.restore(n, restored, pos, err));
    CHECK(restored && pos == n);
    // and the next prompt goes on from there
    CHECK(c.prompt_begin(n, t2.data(), 10, 1, -1, err));
    CHECK(stream(c, n, 10, 10, err));
    CHECK(c.prompt_end(false, pos, err));
    CHECK(pos == n + 10);
    CHECK(w.model.bad_rows == 0);
}

void test_refusals(int port) {
    const Spec spec = make_spec(LayerStageWireFormat::fp32);
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(!c.connect(w.endpoint, "wrong", spec, err));
    CHECK(err.find("STRATA_RPC_TOKEN") != std::string::npos);
    Spec other = spec;
    other.split = 14;
    CHECK(!c.connect(w.endpoint, "secret", other, err));
    CHECK(err.find("--layer-stage-split") != std::string::npos);
    other = spec;
    other.wire = LayerStageWireFormat::bf16;
    CHECK(!c.connect(w.endpoint, "secret", other, err));
    CHECK(err.find("--layer-stage-wire") != std::string::npos);
    other = spec;
    other.identity_b ^= 1;
    CHECK(!c.connect(w.endpoint, "secret", other, err));
    CHECK(err.find("different models") != std::string::npos);
    CHECK(!c.connect(w.endpoint, "", spec, err));
    // the worker's feed fails part-way: the coordinator hears it (on its rows or at the end), the connection closes
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    w.model.fail_after = 300;
    const int n = 1200;
    std::vector<int32_t> t = tokens(n, 4);
    CHECK(c.prompt_begin(0, t.data(), n, 1, -1, err));
    const bool sent = stream(c, 0, n, 256, err);
    int64_t pos = -1;
    const bool ended = sent && c.prompt_end(false, pos, err);
    CHECK(!ended);
    CHECK(!c.connected());
    CHECK(err.find("fake failed") != std::string::npos || err.find("ended the prompt") != std::string::npos ||
          err.find("receive failed") != std::string::npos || err.find("send failed") != std::string::npos);
    w.model.fail_after = -1;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(c.prompt_begin(0, t.data(), 20, 1, -1, err));
    CHECK(stream(c, 0, 20, 20, err));
    CHECK(c.prompt_end(false, pos, err));
    CHECK(pos == 20);
    // a prompt whose rows run short is not ended as complete
    CHECK(c.prompt_begin(20, t.data(), 30, 1, -1, err));
    CHECK(stream(c, 20, 10, 10, err));
    CHECK(!c.prompt_end(false, pos, err));
    CHECK(!c.connected());
}

void test_wire_codec() {
    using strata::core::layer_stage_decode_activations;
    using strata::core::layer_stage_encode_activations;
    std::vector<float> v = {0.0f, -1.5f, 3.14159f, 1e-3f, -65000.0f, 1e20f};
    std::vector<uint8_t> b;
    std::vector<float> back(v.size());
    std::string err;
    CHECK(layer_stage_encode_activations(LayerStageWireFormat::fp32, v.data(), v.size(), b, err));
    CHECK(layer_stage_decode_activations(LayerStageWireFormat::fp32, b.data(), b.size(), back.data(), back.size(), err));
    CHECK(std::memcmp(back.data(), v.data(), v.size() * 4) == 0);
    CHECK(layer_stage_encode_activations(LayerStageWireFormat::bf16, v.data(), v.size(), b, err));
    CHECK(b.size() == v.size() * 2);
    CHECK(layer_stage_decode_activations(LayerStageWireFormat::bf16, b.data(), b.size(), back.data(), back.size(), err));
    for (size_t i = 0; i < v.size(); ++i) CHECK(std::fabs(back[i] - v[i]) <= std::fabs(v[i]) / 128.0f);
    CHECK(!layer_stage_encode_activations(LayerStageWireFormat::fp16, v.data(), v.size(), b, err));   // (1e20)
    v.back() = NAN;
    CHECK(!layer_stage_encode_activations(LayerStageWireFormat::fp32, v.data(), v.size(), b, err));
    LayerStageWireFormat f;
    CHECK(parse_wire("bf16", f) && f == LayerStageWireFormat::bf16 && !parse_wire("int8", f));
}

void test_vision_sessions(int port) {
    Spec spec = make_spec(LayerStageWireFormat::bf16);
    spec.image_token = 17; spec.sessions = true;
    Worker w(spec, port);
    w.model.tolerance = 0.004f;
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(c.images(0, {{1, 2, 77}}, err));
    auto t = tokens(5, 9);
    std::vector<float> h, pre;
    make_rows(0, 5, h, pre);
    int64_t after = 0;
    CHECK(c.window(0, t.data(), 5, 1, h.data(), pre.data(), after, err));
    CHECK(w.model.image_spans.size() == 1 && w.model.image_spans[0].hash == 77);
    Sampling sampling; sampling.temperature = 1.2f; sampling.top_p = 0.9f; sampling.seed = 123; sampling.counter = 9;
    int before = -1, restored = -2;
    CHECK(c.sample(sampling, before, err));
    SnapshotRef ref{"0123456789abcdef0123456789abcdef", 5, prefix_hash(t)};
    uint64_t bytes = 0;
    CHECK(c.snapshot(true, ref, bytes, err) && bytes == 20);
    // A fresh connection resets live state, while snapshots survive it.
    c.close();
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(c.snapshot(false, ref, bytes, err));
    CHECK(c.sample(sampling, restored, err) && restored == before);
    CHECK(w.model.image_spans.size() == 1 && w.model.image_spans[0].start == 1);
    ref.prefix ^= 1;
    CHECK(!c.snapshot(false, ref, bytes, err) && !c.connected());
    ref.id = "../not-a-snapshot";
    CHECK(!valid_snapshot(ref, spec.max_context));
    CHECK(!valid_images({{3, 4, 1}, {4, 1, 2}}, spec.max_context));
    CHECK(!valid_images({{(int64_t) spec.max_context, 1, 1}}, spec.max_context));
    Spec wrong = spec; wrong.image_token = -1;
    CHECK(!same_spec(spec, wrong, err));
    wrong = spec; wrong.sessions = false;
    CHECK(!same_spec(spec, wrong, err));
}

// the split rules on DeepSeek-V4.1-Flash's own layout (pack_gguf's geometry: tests/terminal/test_pack_gguf_engram.py)
void test_split_rules() {
    Layout v41;
    v41.n_layers = 40;
    v41.kv_src = {2, 8, 14, 20};
    v41.ix_src = {2, 8, 14, 20, 24, 28, 32, 36};
    v41.eg_layers = {1, 14};
    v41.cand_src = 20;
    std::string why;
    CHECK(check_split(v41, 20, why));
    CHECK(!check_split(v41, 14, why) && why.find("Engram layer 14") != std::string::npos);
    CHECK(!check_split(v41, 8, why) && why.find("Engram") != std::string::npos);
    CHECK(!check_split(v41, 21, why) && why.find("own compressed KV") != std::string::npos);
    CHECK(!check_split(v41, 24, why) && why.find("own compressed KV") != std::string::npos);
    CHECK(!check_split(v41, 0, why) && !check_split(v41, 40, why));
    Layout early = v41;   // (a split past the candidate source whose later index layers would read its candidates)
    early.kv_src = {2, 8, 14, 20, 26};
    early.ix_src = {2, 8, 14, 20, 26, 28};
    CHECK(!check_split(early, 26, why) && why.find("candidates of layer 20") != std::string::npos);
    CHECK(cut_layers(v41.kv_src, 20, 40) == std::vector<int>({0}));
    CHECK(cut_layers(v41.kv_src, 0, 20) == std::vector<int>({2, 8, 14}));
    CHECK(cut_layers(v41.ix_src, 20, 40) == std::vector<int>({0, 4, 8, 12, 16}));
    CHECK(join(v41.eg_layers) == "1, 14");
}

}  // namespace

int main() {
    test_vision_sessions(port_base() + 15);
    test_split_rules();
    const int base = port_base();
    test_wire_codec();
    test_fp32(base);
    test_bf16_and_halt(base + 1);
    test_refusals(base + 2);
    if (failures != 0) {
        std::fprintf(stderr, "ds4_stage_test: %d failures\n", failures);
        return 1;
    }
    std::printf("ds4_stage_test: ok\n");
    return 0;
}
