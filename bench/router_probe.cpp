// Hypothesis No. 4, step 2 (docs/MOE-VIABLE-HYPOTHESES.md): dump what an early
// router would see. For every layer of Tiel it saves, per prompt token:
//   attn_post_norm-L   the real router input of layer L     [n_tokens, 2048] f32
//   l_out-L            the residual after layer L            [n_tokens, 2048] f32
//   ffn_moe_logits-L   the real router logits of layer L     [n_tokens, 256]  f32
// bench/router_probe.py then scores predictors offline against the real top-8.
//
// Usage: router_probe model.gguf prompt.txt max_tokens out_dir
// Experts stay on the CPU (like --n-cpu-moe 40), the rest on the GPU.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct probe {
    std::string dir;
    std::map<std::string, FILE *> files;
    std::vector<uint8_t> buf;
};

static bool wanted(const char * name) {
    return strncmp(name, "attn_post_norm-", 15) == 0 ||
           strncmp(name, "l_out-", 6) == 0 ||
           strncmp(name, "ffn_moe_logits-", 15) == 0;
}

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name)) return true;
    probe * p = (probe *) ud;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "skip %s: type %s contiguous %d\n", t->name, ggml_type_name(t->type), ggml_is_contiguous(t));
        return true;
    }
    const size_t n = ggml_nbytes(t);
    p->buf.resize(n);
    ggml_backend_tensor_get(t, p->buf.data(), 0, n);
    FILE *& f = p->files[t->name];
    if (!f) f = fopen((p->dir + "/" + t->name + ".f32").c_str(), "wb");
    fwrite(p->buf.data(), 1, n, f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf prompt.txt max_tokens out_dir\n", argv[0]); return 1; }
    const int max_tokens = atoi(argv[3]);
    probe p; p.dir = argv[4];

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    const std::string text = ss.str();

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

    std::vector<llama_token> tok(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), true, false);
    if (nt < 0) return 1;
    tok.resize(std::min(nt, max_tokens));
    fprintf(stderr, "tokens: %d of %d\n", (int) tok.size(), nt);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) tok.size() + 64;
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
    for (auto & kv : p.files) fclose(kv.second);

    FILE * f = fopen((p.dir + "/meta.txt").c_str(), "w");
    fprintf(f, "n_tokens %d\nfiles %d\n", (int) tok.size(), (int) p.files.size());
    fclose(f);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
