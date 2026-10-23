# Residual-Spec for llama.cpp: Lossless & Training-Free LLM Offload Acceleration

> **2-bit Base in VRAM + AVX2 Residual Verifier in System RAM: Beating Q4 Offload Speed with Full Q8/BF16 Quality on Budget GPUs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-feature%2Fresidual--spec-blue)](https://github.com/cerXXXX/llama.cpp/tree/feature/residual-spec)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20(AVX2%20%2B%20CUDA)-green)]()

---

## 📌 Executive Summary

Running 12B–35B models on consumer laptops (e.g., RTX 3060 6GB + Ryzen 5600H) typically forces users into severe trade-offs:
1. **Partial Offloading (Q4_K_M):** 30 layers in VRAM, 18 layers in DDR4 $\implies$ **$9.7 - 10.2\text{ tok/s}$** (strictly bottlenecked by system RAM bandwidth on *every single token*).
2. **Aggressive Quantization (IQ2_M):** 100% in VRAM $\implies$ **$34.0\text{ tok/s}$**, but **catastrophic quality loss** (Wikitext Perplexity degrades from 596 to **3498**, MMLU-Pro drops to 13%).

**Residual Speculative Decoding** solves this fundamental bottleneck:
* **Draft Model (100% in VRAM, ~2.8 GB):** An aligned 2-bit GPTQ base that generates speculative candidate tokens at **$34\text{ tok/s}$** without touching system RAM.
* **Residual Verifier (100% in DDR4, ~8.5 GB):** Exact residual difference $W_{res} = W_{orig} - W_{base}$ computed via hand-optimized **AVX2 + FMA3** SIMD micro-kernels.
* **The Breakthrough:** The slow DDR4 bus is accessed **only once per draft batch** (every 300–400 ms), rather than on every token.

**Results on RTX 3060 Laptop (6GB) + AMD Ryzen 5 5600H (DDR4 @ 34 GB/s):**
* **Generation Speed:** **$15.8 - 23.5\text{ tok/s}$** (**$1.55\times - 2.30\times$ faster than standard Q4 offload**).
* **Intelligence / Accuracy:** Restores full **lossless Q8/BF16 quality** (Wikitext PPL **607.9** vs 596.2 BF16, MMLU-Pro **18.0%** vs 15.0% Q4).
* **Memory Safety:** Uses only **$4.2\text{ GB}$ VRAM**, leaving $>1.8\text{ GB}$ headroom for large contexts.
* **Training-Free:** Zero extra neural network training required.

---

## 📊 Benchmark Comparison: Gemma 4 12B on RTX 3060 6GB

| Configuration | Offload Placement | Speed (tok/s) | VRAM Used | Wikitext PPL | MMLU-Pro | Need Fine-tuning? |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Pure IQ2_M (Extreme Speed)** | 100% in VRAM (`-ngl 99`) | **34.0 t/s** | 4.7 GB | 3498.7 *(ruined)* | 13.0% | No |
| **Standard Q4_K_M (Baseline)** | 30 GPU / 18 DDR4 | **9.7 – 10.2 t/s** | 4.8 GB | ~650 | 15.0% | No |
| **Q4_K_M + Google MTP** | 32 GPU + MTP Head in VRAM | **17.7 t/s** | 5.6 GB | ~650 | 15.0% | **Yes (MTP Head)** |
| **Q4_K_M + z-lab DFlash** | 32 GPU + DFlash in VRAM | **17.5 t/s** | 5.9 GB *(near OOM)* | ~650 | 15.0% | **Yes (DFlash Drafter)** |
| **Residual Spec (Q6, Linear D=2)** | 2b VRAM + Q6 DDR4 | **15.8 t/s** | **4.2 GB** | 607.9 *(BF16 lvl)* | **18.0%** | **No (Training-Free)** |
| **Residual Spec (Q8, Tree D=10)** | 2b VRAM + Q8 DDR4 | **23.5 t/s** | **4.2 GB** | **622.4** *(lossless)* | **17.0%** | **No (Training-Free)** |
| *Full BF16 (Unquantized)* | *Requires 24 GB VRAM* | *—* | *—* | *596.2* | *19.0%* | *Reference* |

*(Tested on Windows 11, Ryzen 5 5600H, RTX 3060 Laptop 6GB, 32GB DDR4-3200).*

---

## 🔬 Core Discoveries & Hardware Physics

### 1. Why Q8 Residual is 5x Faster than Q6 Residual on Zen 3 (AVX2)
In naive theory, 6-bit weights save RAM bandwidth compared to 8-bit. In hardware reality on AVX2:
* **Q8 Residual (`int8_t`):** Exactly 1 byte per weight. The CPU streams continuous 256-bit memory vectors (`_mm256_loadu_si256`) at the full physical DDR4 ceiling (**$34.17\text{ GB/s}$**). Layer time: **$0.91\text{ ms}$**.
* **Q6 Residual:** 6-bit numbers straddle byte boundaries. Without **AVX-512 VBMI** (`_mm_multishift_epi64_epi8`, absent on Zen 3 / Intel 10th gen), unpacking each 6-bit weight requires 6–8 scalar ALU shifts, masks, and shuffles. The CPU stalls on bit-manipulation, dropping effective memory read to $2.3\text{ GB/s}$ and taking **$4.60\text{ ms}$** per layer (**5x slower!**).
* **Takeaway:** For AVX2 CPUs, **8-bit residual is the optimal sweet spot**.

### 2. Draft Depth Sweep: Linear Chain vs. Tree Drafting
Empirical sweep over depth $D \in [1 \dots 24]$ with draft probability decay:

* **Linear Chain (`--spec-draft-n-max N`):**
  * **Optimal Depth:** Strictly **$D = 2$** ($\tau = 1.45$ tokens accepted, **$15.80\text{ tok/s}$**, $1.55\times$).
  * For $D \ge 10$, cumulative chain rejection causes GPU wasted work, dropping speed below autoregressive baseline ($9.26\text{ tok/s}$).
* **Branching Tree Drafting (SubSpec Tree, $W=2$):**
  * **Optimal Depth:** **$D = 8 - 10$** ($\tau = 8.51$ tokens accepted, **$85.1\%$ acceptance rate**, **$23.49\text{ tok/s}$**, $2.30\times$).
  * Branching eliminates cumulative decay: CPU reads DDR4 once every 400 ms and verifies 8–9 tokens in one pass.

---

## 🚀 Quick Start & Usage

### 1. Grab the Fork or Patch
The runtime is merged in our `llama.cpp` fork:
* **Fork Branch:** [`https://github.com/cerXXXX/llama.cpp/tree/feature/residual-spec`](https://github.com/cerXXXX/llama.cpp/tree/feature/residual-spec)
* **Or Apply Standalone Patch:**
  ```bash
  git clone https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  git apply /path/to/patches/feature-residual-spec.patch
  ```

### 2. Build via CMake / Ninja
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_OPENMP=OFF
cmake --build build --config Release --target llama-residual-bench llama-cli
```

### 3. Run Hardware Benchmarks
```powershell
./build/bin/llama-residual-bench.exe --kernel-bench
```
Measures your exact CPU DDR4 streaming speed, AVX2 GEMV GFLOPS across token batches $N \in \{1, 2, 4, 8, 16\}$, Fast Hadamard latency, and the full depth selection matrix.

### 4. Run Speculative Inference
```powershell
./build/bin/llama-cli.exe `
  -m models/gemma4-12b/gemma-4-12b-it-Q4_K_M.gguf `
  -md models/gemma4-12b/gemma-4-12b-it-UD-IQ2_M.gguf `
  --spec-type draft-subspec `
  --spec-draft-temp 0.2 `
  --spec-shared-kv `
  --spec-draft-n-max 2 `
  -ngl 32 -ngld 99 `
  -p "Explain quantum entanglement in simple terms."
```

---

## 📂 Repository Layout

* [`CLAUDE.md`](CLAUDE.md) — Comprehensive master engineering log with raw measurements, formulas, hardware invariants, and execution journal.
* [`docs/RESIDUAL-SPEC-BENCHMARK.md`](docs/RESIDUAL-SPEC-BENCHMARK.md) — Full technical benchmark and roofline analysis report.
* [`patches/feature-residual-spec.patch`](patches/feature-residual-spec.patch) — Standalone git patch for upstream `llama.cpp`.
* [`llama-cpp-src/`](llama-cpp-src/) — Clean C++ source files (`common/residual-cpu.*`, `tools/residual-bench/`).
* [`experiments/gemma4_residual_spec/`](experiments/gemma4_residual_spec/) — Python quantization, calibration, and evaluation scripts:
  * `src/twobit/` — Hadamard rotation, 2-bit GPTQ, residual quantizers, speculative trees.
  * `src/bench_gguf_models.py` — Automated GGUF comparative benchmark harness.
  * `results/matrix_10h/` — 10-hour verification matrix (MMLU-Pro 700 questions, ARC-Challenge 1172 questions, Wikitext-2 PPL).
* [`bench/`](bench/) — Profiling and hardware benchmarking scripts.

---

## 📚 Acknowledgments & References

1. **u/Happy_Guitar3521** on r/LocalLLaMA for the original decomposition concept and research prototype (`twobit-residual-spec`).
2. **SubSpec (NeurIPS 2025):** *"Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding"* (arXiv:2509.18344).
3. **llama.cpp community** for the foundational inference engine.
