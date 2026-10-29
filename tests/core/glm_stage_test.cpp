// The GLM layer-stage protocol (src/core/glm_stage.cpp) over loopback TCP, without a GPU: a fake worker model that
// takes the prompt in its own chunk size, checks every row it gets, and keeps the state a Glm5Model keeps (position,
// snapshot, checkpoints, a paired session file).
#include "strata/core/glm_stage.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace strata::glm::stage;
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

constexpr uint32_t kDim = 24, kHc = 4, kVocab = 1000, kLayers = 45;
constexpr size_t kHf = (size_t) kHc * kDim;

// the row the coordinator sends for position p, so the worker can check what it got
float h_value(int64_t p, size_t i) { return (float) std::sin(0.37 * (double) p + 0.011 * (double) i) * 3.0f + 0.5f; }

void make_rows(int64_t p0, int T, std::vector<float>& h) {
    h.resize((size_t) T * kHf);
    for (int t = 0; t < T; ++t)
        for (size_t i = 0; i < kHf; ++i) h[(size_t) t * kHf + i] = h_value(p0 + t, i);
}

bool near(float got, float want, float tol) {
    return tol == 0.0f ? got == want : std::fabs(got - want) <= tol * std::fabs(want) + 1e-6f;
}

struct FakeModel : Backend {
    int chunk_cap = 300;       // the chunk this end takes at most
    float tolerance = 0.0f;    // 0: rows must be exact (FP32)
    std::atomic<int> resets{0};
    int64_t p = 0, snap = -1;
    int64_t ck[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int64_t rows_seen = 0, bad_rows = 0;
    double sum = 0, snap_sum = 0, ck_sum[8] = {};
    std::vector<float> buf;
    int64_t pr_p0 = 0, pr_n = 0;
    std::map<int, std::vector<std::pair<int32_t, float>>> bias_tab;
    struct Saved { int64_t pos; double sum; uint64_t prefix; };
    std::map<std::string, Saved> files;
    Sampling last_sample;

    int64_t pos() const override { return p; }
    void reset() override {
        ++resets;
        p = 0;
        snap = -1;
        sum = 0;
        for (auto& c : ck) c = -1;
    }
    bool snap_save(std::string&) override { snap = p; snap_sum = sum; return true; }
    bool snap_restore(int64_t& pos, std::string& err) override {
        if (snap < 0) { err = "none"; return false; }
        p = snap; sum = snap_sum; pos = p; return true;
    }
    int64_t snap_pos() const override { return snap; }
    bool ckpt_save(int slot, bool from_snapshot, std::string&) override {
        ck[slot & 7] = from_snapshot ? snap : p;
        ck_sum[slot & 7] = from_snapshot ? snap_sum : sum;
        return true;
    }
    bool ckpt_restore(int slot, int64_t& pos, std::string&) override {
        p = ck[slot & 7]; sum = ck_sum[slot & 7]; pos = p; return true;
    }
    int64_t ckpt_pos(int slot) override { return ck[slot & 7]; }
    bool check_row(int64_t q, const float* r) {
        bool ok = true;
        for (size_t i = 0; i < kHf; ++i) {
            if (!near(r[i], h_value(q, i), tolerance)) ok = false;
            sum += r[i];
        }
        ++rows_seen;
        if (!ok) ++bad_rows;
        return ok;
    }
    bool step(int64_t q, const float* row, int& token, float* r_out, std::string& err) override {
        if (q != p) { err = "out of step"; return false; }
        check_row(q, row);
        ++p;
        if (r_out != nullptr)
            for (size_t i = 0; i < kHf; ++i) r_out[i] = 2.0f * row[i];
        token = (int) (q * 7 % kVocab);
        return true;
    }
    bool prompt_begin(int64_t p0, int64_t n, int T, int& t_bound, std::string&) override {
        pr_p0 = p0;
        pr_n = n;
        t_bound = std::min(T, chunk_cap);
        buf.assign((size_t) t_bound * kHf, 0.0f);
        return true;
    }
    float* prompt_rows() override { return buf.data(); }
    bool prompt_chunk(int64_t p0, int T, std::string& err) override {
        if (p0 != p) { err = "chunk out of order"; return false; }
        for (int t = 0; t < T; ++t) check_row(p0 + t, buf.data() + (size_t) t * kHf);
        p += T;
        return true;
    }
    bool prompt_end(bool, std::string&) override { buf.clear(); return true; }
    int sample(const Sampling& s, std::string&) override {
        last_sample = s;
        return (int) ((s.seed + s.counter + s.history.size() + (s.bias_slot >= 0 ? 100 : 0)) % kVocab);
    }
    bool bias(int slot, const std::vector<std::pair<int32_t, float>>& b, std::string&) override {
        bias_tab[slot] = b;
        return true;
    }
    bool logits(std::vector<float>& out, std::string&) override {
        out.assign(kVocab, 0.0f);
        out[(size_t) (p % kVocab)] = 1.0f;
        return true;
    }
    bool session(bool save, const SnapshotRef& ref, uint64_t& bytes, std::string& err) override {
        if (save) {
            if (snap != ref.position) { err = "no snapshot at that position"; return false; }
            files[ref.id] = {snap, snap_sum, ref.prefix};
        } else {
            const auto it = files.find(ref.id);
            if (it == files.end() || it->second.pos != ref.position || it->second.prefix != ref.prefix) {
                err = "missing or mismatched session";
                return false;
            }
            snap = p = it->second.pos;
            snap_sum = sum = it->second.sum;
        }
        bytes = (uint64_t) ref.position * 8;
        return true;
    }
};

// a worker on its own thread: accepts connections one after another until the test ends
struct Worker {
    FakeModel model;
    Server server;
    std::string endpoint;
    rpc::Socket ls = rpc::kInvalidSocket;
    std::thread th;
    std::atomic<bool> stop{false};
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
            }
        });
    }
    ~Worker() {
        stop.store(true);
        rpc::Socket c = rpc::kInvalidSocket;   // (wake the accept)
        std::string e;
        rpc::connect(endpoint, c, rpc::deadline_after(std::chrono::seconds(2)), e);
        if (c != rpc::kInvalidSocket) rpc::close(c);
        th.join();
        rpc::close(ls);
    }
};

Spec base_spec() {
    Spec s;
    s.identity_a = 11;
    s.identity_b = 22;
    s.n_layers = kLayers;
    s.split = 26;
    s.dim = kDim;
    s.hc = kHc;
    s.vocab = kVocab;
    s.max_context = 1 << 20;
    s.wire = LayerStageWireFormat::fp32;
    return s;
}

int port_base() { return 41000 + (int) (std::chrono::steady_clock::now().time_since_epoch().count() % 2000); }

// a whole prompt of n rows from the coordinator's position p0, in chunks of T (the worker may take fewer)
bool send_prompt(Client& c, int64_t p0, int64_t n, int T, std::string& err) {
    int tb = 0;
    if (!c.prompt_begin(p0, n, T, tb, err)) return false;
    std::vector<float> h;
    for (int64_t a = 0; a < n; a += tb) {
        const int t = (int) std::min<int64_t>(tb, n - a);
        make_rows(p0 + a, t, h);
        if (!c.prompt_chunk(p0 + a, t, h.data(), err)) return false;
    }
    int64_t pos = 0;
    return c.prompt_end(true, pos, err) && pos == p0 + n;
}

void test_fp32(int port) {
    Spec spec = base_spec();
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(c.epoch() == 1);
    // a prompt of 1,300 rows in the coordinator's chunks of 700: the worker takes 300 at a time, pieces of 512
    CHECK(send_prompt(c, 0, 1300, 700, err));
    CHECK(w.model.p == 1300);
    CHECK(w.model.rows_seen == 1300);
    CHECK(w.model.bad_rows == 0);
    // decode steps: the head's argmax comes back
    std::vector<float> row(kHf);
    for (int64_t q = 1300; q < 1305; ++q) {
        for (size_t i = 0; i < kHf; ++i) row[i] = h_value(q, i);
        int tok = -1;
        CHECK(c.step(q, row.data(), tok, nullptr, err));
        CHECK(tok == (int) (q * 7 % kVocab));
    }
    CHECK(w.model.p == 1305 && w.model.bad_rows == 0);
    // a step at the wrong position is refused and closes the link; reconnecting starts over at 0
    int tok = -1;
    CHECK(!c.step(999, row.data(), tok, nullptr, err));
    CHECK(!c.connected());
    CHECK(c.reconnect(err));
    CHECK(c.epoch() == 2);
    CHECK(w.model.p == 0);
    c.close();
}

void test_bf16_and_halt(int port) {
    Spec spec = base_spec();
    spec.wire = LayerStageWireFormat::bf16;
    Worker w(spec, port);
    w.model.tolerance = 1.0f / 128.0f;
    w.model.chunk_cap = 4096;
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(send_prompt(c, 0, 2000, 1024, err));
    CHECK(w.model.bad_rows == 0 && w.model.p == 2000);
    // a prompt stopped after its first chunk: the worker keeps what it read, the link stays
    int tb = 0;
    CHECK(c.prompt_begin(2000, 3000, 1000, tb, err));
    CHECK(tb == 1000);
    std::vector<float> h;
    make_rows(2000, 1000, h);
    CHECK(c.prompt_chunk(2000, 1000, h.data(), err));
    int64_t pos = 0;
    CHECK(c.prompt_end(false, pos, err));
    CHECK(pos == 3000 && w.model.p == 3000);
    CHECK(c.connected());
    // a request while a prompt streams is refused
    CHECK(c.prompt_begin(3000, 10, 10, tb, err));
    CHECK(!c.reset(err));
    CHECK(!c.connected());
    c.close();
}

void test_refusals(int port) {
    Spec spec = base_spec();
    Worker w(spec, port);
    std::string err;
    {
        Client c;
        CHECK(!c.connect(w.endpoint, "wrong", spec, err));
        CHECK(err.find("STRATA_RPC_TOKEN") != std::string::npos);
        CHECK(c.refused());                        // (the coordinator stops: waiting cannot fix the token)
        CHECK(!c.connect(w.endpoint, "", spec, err));
        CHECK(c.refused());
    }
    {
        // nothing listens there (a worker still loading): not a refusal, so the coordinator waits and tries again
        Client c;
        CHECK(!c.connect("127.0.0.1:" + std::to_string(port + 9), "secret", spec, err));
        CHECK(!c.refused());
    }
    const auto refused = [&](Spec s, const char* why) {
        Client c;
        std::string e;
        CHECK(!c.connect(w.endpoint, "secret", s, e));
        CHECK(e.find(why) != std::string::npos);
        CHECK(c.refused());
    };
    Spec s = spec;
    s.split = 20;
    refused(s, "--layer-stage-split differs");
    s = spec;
    s.wire = LayerStageWireFormat::bf16;
    refused(s, "--layer-stage-wire differs");
    s = spec;
    s.head_on_coordinator = true;
    refused(s, "--layer-stage-head differs");
    s = spec;
    s.max_context = 4096;
    refused(s, "context differs");
    s = spec;
    s.identity_b = 23;
    refused(s, "different models");
    s = spec;
    s.sessions = true;
    refused(s, "--layer-stage-sessions");
    std::string why;
    CHECK(!check_split(45, 0, why));
    CHECK(!check_split(45, 45, why));
    CHECK(check_split(45, 26, why));
}

void test_state(int port) {
    Spec spec = base_spec();
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(send_prompt(c, 0, 100, 100, err));
    bool ok = false;
    CHECK(c.snap_save(100, ok, err) && ok);
    CHECK(c.ckpt_save(3, false, 100, ok, err) && ok);
    CHECK(c.ckpt_save(4, true, 100, ok, err) && ok);
    CHECK(send_prompt(c, 100, 50, 64, err));
    CHECK(w.model.p == 150);
    // a snapshot restore asks for the position the coordinator's snapshot holds
    CHECK(c.snap_restore(90, ok, err) && !ok);   // not the worker's: nothing restored
    CHECK(w.model.p == 150);
    CHECK(c.snap_restore(100, ok, err) && ok);
    CHECK(w.model.p == 100);
    CHECK(c.ckpt_restore(3, 100, ok, err) && ok);
    CHECK(c.ckpt_restore(5, 100, ok, err) && !ok);   // an empty slot
    // a checkpoint at a position the worker does not stand at is out of step: refused, the link closes
    CHECK(!c.ckpt_save(1, false, 77, ok, err));
    CHECK(!c.connected());
    CHECK(c.reconnect(err));
    CHECK(c.reset(err));
    CHECK(w.model.resets.load() >= 3);
    c.close();
}

void test_head_on_coordinator(int port) {
    Spec spec = base_spec();
    spec.head_on_coordinator = true;
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    std::vector<float> row(kHf), out(kHf, 0.0f);
    for (size_t i = 0; i < kHf; ++i) row[i] = h_value(0, i);
    int tok = 5;
    CHECK(c.step(0, row.data(), tok, out.data(), err));
    CHECK(tok == -1);
    bool same = true;
    for (size_t i = 0; i < kHf; ++i) same = same && out[i] == 2.0f * row[i];
    CHECK(same);
    // without an output buffer the answer cannot be taken
    CHECK(!c.step(1, row.data(), tok, nullptr, err));
    c.close();
}

void test_sampling(int port) {
    Spec spec = base_spec();
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(c.bias(2, {{5, -1e30f}, {7, 2.5f}}, err));
    CHECK(w.model.bias_tab[2].size() == 2 && w.model.bias_tab[2][1].first == 7);
    Sampling s;
    s.seed = 40;
    s.counter = 2;
    s.temperature = 0.7f;
    s.top_k = 20;
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.1f;
    s.history = {1, 2, 3};
    s.bias_slot = 2;
    int tok = -1;
    CHECK(c.sample(s, tok, err));
    CHECK(tok == (40 + 2 + 3 + 100) % (int) kVocab);
    CHECK(w.model.last_sample.penalty_last_n == 64 && w.model.last_sample.temperature == 0.7f);
    CHECK(w.model.last_sample.history.size() == 3);
    std::vector<float> lg;
    CHECK(c.logits(lg, err));
    CHECK(lg.size() == kVocab && lg[0] == 1.0f);
    c.close();
}

void test_sessions(int port) {
    Spec spec = base_spec();
    spec.sessions = true;
    Worker w(spec, port);
    Client c;
    std::string err;
    CHECK(c.connect(w.endpoint, "secret", spec, err));
    CHECK(send_prompt(c, 0, 64, 64, err));
    bool ok = false;
    CHECK(c.snap_save(64, ok, err) && ok);
    SnapshotRef ref;
    ref.id = new_snapshot_id();
    CHECK(ref.id.size() == 32);
    ref.position = 64;
    ref.prefix = prefix_hash({1, 2, 3});
    uint64_t bytes = 0;
    CHECK(c.session(true, ref, bytes, err));
    CHECK(bytes == 64 * 8);
    CHECK(c.reset(err));
    CHECK(w.model.p == 0);
    // a session the worker does not hold: the request fails, the link stays
    SnapshotRef other = ref;
    other.id = new_snapshot_id();
    CHECK(!c.session(false, other, bytes, err));
    CHECK(c.connected());
    CHECK(c.session(false, ref, bytes, err));
    CHECK(w.model.p == 64 && w.model.snap == 64);
    // an id that is not 32 hex characters never reaches the worker
    SnapshotRef bad = ref;
    bad.id = "../../etc/passwd0000000000000000";
    CHECK(!c.session(false, bad, bytes, err));
    c.close();
    // sessions off on both PCs: the request is not sent
    Spec off = base_spec();
    Worker w2(off, port + 1);
    Client c2;
    CHECK(c2.connect(w2.endpoint, "secret", off, err));
    CHECK(!c2.session(true, ref, bytes, err));
    c2.close();
}

}  // namespace

int main() {
    const int p = port_base();
    test_fp32(p);
    test_bf16_and_halt(p + 10);
    test_refusals(p + 20);
    test_state(p + 30);
    test_head_on_coordinator(p + 40);
    test_sampling(p + 50);
    test_sessions(p + 60);
    if (failures != 0) {
        std::fprintf(stderr, "glm_stage_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("glm_stage_test: ok\n");
    return 0;
}
