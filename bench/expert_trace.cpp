// Hypothesis No. 8 (docs/MOE-VIABLE-HYPOTHESES.md): how skewed is Tiel's expert
// selection? Records the real top-8 expert ids of every MoE layer for every
// token, both while the prompt is processed and while the model generates.
//
// Output, uint8, layout [n_layer=40][n_tokens][8]:
//   <out>.prefill.u8   prompt tokens (ubatches of 512)
//   <out>.gen.u8       generated tokens (greedy, one token per decode)
// bench/expert_hist.py turns them into hit rates of a static hot set.
//
// Usage: expert_trace model.gguf prompt.txt max_prompt_tokens n_gen out_prefix [chat]
//   chat = 1 wraps the prompt as one ChatML user turn (generation then answers it).
// Experts stay on the CPU (like --n-cpu-moe 40), the rest on the GPU.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int N_LAYER = 40, K = 8;

struct trace {
    std::vector<std::vector<uint8_t>> ids = std::vector<std::vector<uint8_t>>(N_LAYER);
    std::vector<int32_t> row;
};

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return !ask;
    if (ask) return true;
    const int il = atoi(t->name + 13);
    if (il < 0 || il >= N_LAYER) return true;
    if (t->type != GGML_TYPE_I32 || t->ne[0] != K) {
        fprintf(stderr, "unexpected %s: type %s ne0 %lld\n", t->name, ggml_type_name(t->type), (long long) t->ne[0]);
        exit(2);
    }
    trace * p = (trace *) ud;
    p->row.resize(K);
    for (int64_t j = 0; j < t->ne[1]; j++) {
        ggml_backend_tensor_get(t, p->row.data(), j * t->nb[1], K * sizeof(int32_t));
        for (int k = 0; k < K; k++) p->ids[il].push_back((uint8_t) p->row[k]);
    }
    return true;
}

static void flush(trace & p, const std::string & path) {
    const size_t n = p.ids[0].size() / K;
    FILE * f = fopen(path.c_str(), "wb");
    for (int il = 0; il < N_LAYER; il++) {
        if (p.ids[il].size() != n * K) { fprintf(stderr, "layer %d has %zu ids, expected %zu\n", il, p.ids[il].size(), n * K); exit(3); }
        fwrite(p.ids[il].data(), 1, p.ids[il].size(), f);
        p.ids[il].clear();
    }
    fclose(f);
    fprintf(stderr, "wrote %s: %zu tokens\n", path.c_str(), n);
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model.gguf prompt.txt max_prompt_tokens n_gen out_prefix [chat]\n", argv[0]); return 1; }
    const int max_tokens = atoi(argv[3]);
    const int n_gen = atoi(argv[4]);
    const std::string out = argv[5];
    const bool chat = argc > 6 && atoi(argv[6]) != 0;

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    std::string text = ss.str();

    llama_backend_init();

    static llama_model_tensor_buft_override ov[2] = {
        { "\\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tok(text.size() + 64);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), true, false);
    if (nt < 0) return 1;
    tok.resize(std::min(nt, max_tokens));
    if (chat) {
        // cut on whole tokens, then rebuild as one user turn with special tokens parsed
        std::string body;
        char piece[256];
        for (llama_token t : tok) {
            int n = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
            if (n > 0) body.append(piece, n);
        }
        text = "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n";
        tok.assign(text.size() + 64, 0);
        nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), false, true);
        if (nt < 0) return 1;
        tok.resize(nt);
    }
    fprintf(stderr, "prompt tokens: %zu, chat %d\n", tok.size(), (int) chat);

    trace p;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (tok.size() + n_gen + 64);
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 6;
    cp.n_threads_batch = 6;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &p;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    for (size_t i = 0; i < tok.size(); i += 512) {
        const int n = (int) std::min<size_t>(512, tok.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tok.data() + i, n)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        fprintf(stderr, "decoded %zu\n", i + n);
    }
    flush(p, out + ".prefill.u8");

    if (n_gen > 0) {
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        std::string gen;
        char piece[256];
        for (int i = 0; i < n_gen; i++) {
            llama_token t = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, t)) { fprintf(stderr, "eog after %d\n", i); break; }
            int n = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, true);
            if (n > 0) gen.append(piece, n);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        }
        // the last sampled token was decoded too, so ids cover every generated token
        flush(p, out + ".gen.u8");
        FILE * f = fopen((out + ".gen.txt").c_str(), "wb");
        fwrite(gen.data(), 1, gen.size(), f);
        fclose(f);
        llama_sampler_free(smpl);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
