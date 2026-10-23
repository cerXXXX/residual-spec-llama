---
license: mit
tags:
- quantization
- speculative-decoding
---

# 2-bit draft + residual verifier

Experiments for one idea: keep a 2-bit copy of the model in VRAM and the residual (W minus the 2-bit weights, 6-8 bit) in system RAM. The 2-bit copy is the draft. The verifier is 2-bit + residual, where the GPU does the 2-bit part, the CPU does the residual part, and only activations go over PCIe. The draft costs no extra memory because it is half of the verifier.

It only makes sense when Q8 of the model doesn't fit in your VRAM.

This is research code, not a runtime. All numbers were measured on real weights, but the weights were dequantized and run in plain PyTorch, so there are no tok/s numbers here. The CPU kernel is the only piece that runs the real split, and it is benchmarked on its own.

## What's where

- `twobit/` - the library: Hadamard rotation, 2-bit base with GPTQ, residual quantization, calibration data, exact-KV draft attention, draft trees, three-level draft
- `experiments/` - one script per experiment
- `kernels/` - CPU residual kernel (needs AVX-512) and two benchmarks
- `baselines/` - llama.cpp baseline

## Running

    pip install -r requirements.txt

Run everything from the repo root as a module, for example:

    python -m experiments.e10_build_calibration
    python -m experiments.e10_quantize_layerwise --out weights/ldlq_chat_b2

Every script has `--help`, and the default paths are set up so one step's output is the next step's input.

Rough order:

- Qwen3.5-4B: `e00_download_data`, then `e00`-`e03`, `e02` (saves the 2-bit base), `e03b`-`e03d`, `e09`
- Gemma 4 12B: `e06_prepare_prompts`, `e06_gemma_drafter`, `e06b`-`e06d`, `e10_build_calibration`, `e10_quantize_layerwise`, then `e10_eval_base`, `e11`, `e12`, `e16`
- Gemma 4 31B: `e10_quantize_layerwise --model google/gemma-4-31B-it --out weights/gemma31_ldlq_chat_b2`, then `e13`
- `e17*` only read the rank files saved by `e11`-`e13`

CPU kernel:

    ./kernels/build.sh
    python kernels/bench_residual.py

Feel free to take the idea further, that's why it's here.
