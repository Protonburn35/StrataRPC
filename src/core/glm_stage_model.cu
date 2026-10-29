// src/core/glm_stage_model.cu - StrataRPCG: Glm5Model's side of the layer stage over two PCs (glm_stage.hpp).
//
// The coordinator's model is one part, layers [0, split): every position it reads goes through its layers, then over
// the link to the worker's model (layers split .. last), which answers with the head's argmax - or, with the head on
// the coordinator (--layer-stage-head coordinator), with its last layer's streams, which this part's head reads.  The
// prompt's chunks go over on a thread of their own (glm_prefill.cu: stage_send_chunk) while the next chunk is read
// here.  The state each request keeps between calls - the snapshot, the checkpoints, the position - is kept on both
// PCs in step: every call that moves it here moves it there (glm_fast_path.cu, glm_model.cu).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_stage.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace stage = strata::glm::stage;

namespace {

// the requests' logit-bias tables live on the worker (the head's PC); what logit_bias_table hands the serve loop
// stands for a table slot there - an address of this array, never read
constexpr int kBiasSlots = 65;
float g_bias_tags[kBiasSlots] = {};

bool env_on(const char* k) {
    const char* v = std::getenv(k);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool Glm5Model::stage_attach(stage::Client* link, bool head_here, std::string& err) {
    if (link == nullptr || !link->connected()) {
        err = "glm stage: no link to the worker";
        return false;
    }
    if (fast_ == nullptr) {
        err = "glm stage: the layer stage needs the fast path (a pack; MAYA_GLM_SLOW is set?)";
        return false;
    }
    if (split_next_ != nullptr) {
        err = "glm stage: the layer stage runs one GPU on each PC (this one has a split of its own)";
        return false;
    }
    if (l0_ != 0) {
        err = "glm stage: the coordinator holds the first layers";
        return false;
    }
    if (head_here && !head_loaded()) {
        err = "glm stage: the head was not loaded on the coordinator (set_head_here before the load)";
        return false;
    }
    stage_ = link;
    stage_head_local_ = head_here;
    stage_epoch_ = link->epoch();
    stage_row_.assign((size_t) 2 * (size_t) g_.hc * (size_t) g_.n_embd, 0.0f);
    return true;
}

bool Glm5Model::stage_link_ok(std::string& err) {
    if (stage_ == nullptr) return true;
    if (!stage_->connected()) {
        if (!stage_->reconnect(err)) {
            err = "the layer-stage worker: " + err;
            return false;
        }
        std::fprintf(stderr, "glm stage: connected to the worker again - the conversation is read again from its "
                             "start\n");
    }
    if (stage_->epoch() != stage_epoch_) {   // the worker started over: nothing it kept matches this side any more
        stage_drop_state();
        stage_epoch_ = stage_->epoch();
    }
    return true;
}

void Glm5Model::stage_drop_state() {
    ckpt_join();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        m->snap_pos_ = -1;
        for (auto& c : m->ckpt_pos_) c = -1;
    }
}

int64_t Glm5Model::ckpt_pos_of(int slot) {
    ckpt_join();
    if (slot < 0 || (size_t) slot >= ckpt_pos_.size()) return -1;
    return ckpt_pos_[(size_t) slot];
}

std::string Glm5Model::stage_identity(const std::string& pack_dir) const {
    namespace fs = std::filesystem;
    std::ostringstream id;
    const Glm5Geometry& g = g_;
    id << "stratarpcg-identity-1 layers=" << g.n_layers << " embd=" << g.n_embd << " hc=" << g.hc
       << " vocab=" << g.n_vocab << " experts=" << g.n_expert << " used=" << g.n_exp_used << " kv_lora=" << g.kv_lora
       << " kda=" << g.kda_head_dim << " dense=" << g.dense_lead;
    std::error_code ec;
    // the pack's index (every tensor, its served form and kind) and its experts' table, as texts; the dense weights'
    // file by its size
    for (const char* f : {"index.txt", "native_experts.txt"}) {
        std::ifstream in(fs::path(pack_dir) / f, std::ios::binary);
        std::ostringstream body;
        body << in.rdbuf();
        id << "\n" << f << ":" << body.str().size() << ":" << body.str();
    }
    id << "\ndense.bin:" << (unsigned long long) fs::file_size(fs::path(pack_dir) / "dense.bin", ec);
    std::vector<std::string> shards;
    for (const auto& de : fs::directory_iterator(fs::path(pack_dir).parent_path(), ec))
        if (de.path().extension() == ".gguf")
            shards.push_back(de.path().filename().string() + ":" + std::to_string(de.file_size(ec)));
    std::sort(shards.begin(), shards.end());
    for (const auto& s : shards) id << "\n" << s;
    return id.str();
}

// ---------------------------------------------------------------- the coordinator's decode step
// fast_token has run this part's layers for position p (the residual at tail->state_): the streams go over, the
// worker's layers read them; back come the head's argmax (the head there) or the last layer's streams, which go into
// state_ for this part's head
bool Glm5Model::stage_hand_over(Glm5Model* tail, int64_t p, int& token, std::string& err) {
    static const bool stats = env_on("STRATA_LAYER_STAGE_STATS");
    static thread_local uint64_t steps = 0;
    static thread_local double wait_ms = 0, all_ms = 0;
    static thread_local std::chrono::steady_clock::time_point last{};
    const auto t0 = std::chrono::steady_clock::now();
    FastState* FT = tail->fast_;
    const size_t hf = (size_t) g_.hc * (size_t) g_.n_embd;
    if (stage_row_.size() < 2 * hf) stage_row_.assign(2 * hf, 0.0f);
    cudaSetDevice(tail->dev_);
    if (!glmfast::cuda_ok(cudaMemcpyAsync(stage_row_.data(), tail->state_, hf * sizeof(float), cudaMemcpyDeviceToHost,
                                          FT->cs),
                          "glm stage row out", err) ||
        !glmfast::cuda_ok(cudaEventRecord(FT->ev_done, FT->cs), "glm stage row out", err) ||
        !glmfast::wait_event(FT->ev_done, "glm stage row out", err))
        return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    if (!stage_->connected()) {
        err = "glm stage: the link to the worker is down (the next request connects again)";
        return false;
    }
    const auto tw = std::chrono::steady_clock::now();
    std::string e;
    if (!stage_->step(p, stage_row_.data(), token, stage_head_local_ ? stage_row_.data() + hf : nullptr, e)) {
        err = "glm stage: " + e;
        return false;
    }
    const double w = ms_since(tw);
    if (stage_head_local_ &&
        !glmfast::cuda_ok(cudaMemcpyAsync(tail->state_, stage_row_.data() + hf, hf * sizeof(float),
                                          cudaMemcpyHostToDevice, FT->cs),
                          "glm stage row in", err))
        return false;
    if (stats) {
        wait_ms += w;
        if (last != std::chrono::steady_clock::time_point{}) all_ms += std::chrono::duration<double, std::milli>(t0 - last).count();
        last = t0;
        if (++steps % 256 == 0) {
            std::fprintf(stderr, "glm stage: last 256 steps %.1f ms a token, %.1f of it waiting on the worker\n",
                         all_ms / 255.0, wait_ms / 256.0);
            wait_ms = all_ms = 0;
            last = std::chrono::steady_clock::time_point{};
        }
    }
    return true;
}

int Glm5Model::stage_sample(strata::kernels::SamplerParams& sp, std::string& err) {
    stage::Sampling s;
    s.top_k = sp.top_k;
    s.top_p = sp.top_p;
    s.min_p = sp.min_p;
    s.temperature = sp.temperature;
    s.min_keep = sp.min_keep;
    s.penalty_last_n = sp.penalty_last_n;
    s.penalty_repeat = sp.penalty_repeat;
    s.penalty_freq = sp.penalty_freq;
    s.penalty_present = sp.penalty_present;
    s.seed = sp.seed;
    s.counter = sp.counter;
    s.greedy = sp.greedy;
    if (sp.logit_bias != nullptr) {
        const ptrdiff_t k = sp.logit_bias - g_bias_tags;
        if (k < 0 || k >= kBiasSlots) {
            err = "glm stage: a logit bias that is not the worker's table";
            return -1;
        }
        s.bias_slot = (int32_t) k;
    }
    if (sp.penalty_last_n > 0 && sp.host_history != nullptr) {
        const std::vector<int32_t>& h = *sp.host_history;
        const size_t keep = std::min<size_t>({h.size(), (size_t) sp.penalty_last_n, (size_t) stage::kMaxHistory});
        s.history.assign(h.end() - (ptrdiff_t) keep, h.end());
    }
    int tok = -1;
    std::string e;
    if (!stage_->sample(s, tok, e)) {
        err = "glm stage: " + e;
        return -1;
    }
    return tok;
}

const float* Glm5Model::stage_bias(int slot, const std::vector<std::pair<int32_t, float>>& bias, std::string& err) {
    if (slot < 0 || slot >= kBiasSlots) {
        err = "logit_bias: no conversation slot " + std::to_string(slot);
        return nullptr;
    }
    std::string e;
    if (!stage_->bias(slot, bias, e)) {
        err = "logit_bias: the worker's table: " + e;
        return nullptr;
    }
    return g_bias_tags + slot;
}

// ---------------------------------------------------------------- the worker's decode step
// A later part of a split on one PC reads its hop the same way (fast_token's loop over split_next_): the row into
// state_, this part's layers, then the head - or the streams back.
bool Glm5Model::stage_step(int64_t p, const float* row, int& token, float* r_out, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F == nullptr) {
        err = "the worker needs the fast path";
        return false;
    }
    if (p != pos_) {
        err = "a step at " + std::to_string(p) + ", the worker is at " + std::to_string(pos_);
        return false;
    }
    if (r_out == nullptr && !head_loaded()) {
        err = "the head is not loaded on the worker (--layer-stage-head differs?)";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    pos_ = p + 1;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
    if (!glmfast::cuda_ok(cudaMemcpyAsync(state_, row, hop, cudaMemcpyHostToDevice, F->cs), "glm stage row in", err))
        return false;
    if (!fast_layers(p, true, err)) return false;
    Glm5Model* tail = this;
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {   // (a worker of several GPUs)
        FastState* FA = tail->fast_;
        FastState* FB = B->fast_;
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(FA->hop_h, tail->state_, hop, cudaMemcpyDeviceToHost, FA->cs);
        cudaEventRecord(FA->ev_hop, FA->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
        cudaMemcpyAsync(B->state_, FA->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    cudaSetDevice(tail->dev_);
    if (r_out != nullptr) {   // the head on the coordinator: the last layer's streams go back
        cudaMemcpyAsync(r_out, tail->state_, hop, cudaMemcpyDeviceToHost, FT->cs);
    } else {
        gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x,
                      FT->head_xq, FT->cs);
        const WSlot& ow = tail->ws_map_.at("output.weight");
        gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                       g.n_vocab};
        if (ow.type == 0) o.w = ow.f32;
        if (!gf::mv(&o, 1, FT->cs)) {
            err = "glm fast: output head (type " + std::to_string(ow.type) + ")";
            return false;
        }
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
        cudaMemcpyAsync(FT->tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    }
    cudaEventRecord(FT->ev_done, FT->cs);
    if (!glmfast::wait_event(FT->ev_done, "glm stage step", err)) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm fast: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    if (r_out == nullptr) {
        token = FT->tok_h[0];
        last_tok_ = token;
    } else {
        token = -1;
    }
    F->ms += ms_since(t0);
    ++F->tokens;
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::stage_logits(std::vector<float>& out, std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    if (!tail->head_loaded()) {
        err = "no head on the worker";
        return false;
    }
    cudaSetDevice(tail->dev_);
    out.resize((size_t) g_.n_vocab);
    const bool ok = cudaMemcpy(out.data(), tail->sc_ + tail->sc_logits, (size_t) g_.n_vocab * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaSetDevice(dev_);
    if (!ok) err = "glm fast: logits copy";
    return ok;
}

}  // namespace strata::core
