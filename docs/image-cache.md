# Repeated-image cache and batched variations

Two independent speedups in `hairdeck.infer.image.Pipeline` for interactive use, where the same character image is
run again with other seeds, more variations or colours.

## Repeated-image cache

The pipeline keeps the preprocessing result (SAM 3 / BiRefNet masks and the model-input crop) of the last image, and
the DINOv3 features of the last crop. A new request with the same pixels (alpha included) skips preprocessing; the
image DiT and the colour model share one DINO pass instead of running it twice.

- One entry each; image pixels determine reuse (SHA-256 of the arrays). A per-call `prep=` preprocessor bypasses the
  mask cache, since custom preprocessors may be stateful.
- Off with model offload (`HAIRDECK_OFFLOAD`) and on ZeroGPU.
- `HAIRDECK_IMAGE_CACHE=1` / `0` turns it on / off. **Default: on with ROCm builds of PyTorch, off otherwise**, so
  CUDA behaviour is unchanged unless opted in. Call `Pipeline.clear_image_cache()` after changing weights or
  preprocessing settings on a live pipeline.
- Results include `cache=dict(prep_hit=...)`.

The outputs are unchanged: on both GPUs below, the cached request's tokens, card JSON and chain JSON are identical to
an uncached run of the same seed (`tools/bench/bench_image_cache.py`, which fails if they differ).

Example 1, 50 steps, CFG 2, colours, chains and GLB files, single requests after warm-up:

| GPU | Uncached | Same image again | Preprocessing |
|---|---|---|---|
| RTX 5060 Ti, Windows, torch 2.8.0+cu128 | 6.15 s | **5.25 s** | 0.99 s → 0.01 s |
| Radeon 8060S (Strix Halo), ROCm 7.2, torch 2.13 | 2.32 s | **1.18 s** | 1.09 s → 0.00 s |

## Batched variations (opt-in)

With `n_samples` > 1 the seeds are sampled as one batch (one DiT batch of size n with the condition shared) instead
of one after the other; VAE decoding, colours, chains and files stay per seed.

Batching changes GEMM shapes and therefore rounding, so the same seed gives a slightly different hairstyle than
sequential sampling (relative L2 of the tokens 0.4-8%). It is therefore **off by default** everywhere:
`HAIRDECK_BATCH_VARIATIONS=1` or `Pipeline.run(..., batch_variations=True)` opts in. Off with model offload and on
ZeroGPU. The first use of each batch size may compile when the fast sampler is enabled.

Example 2, seeds 1-4, 50 steps, CFG 2, colours, chains and GLB files (`tools/bench/bench_variations.py`):

| GPU | Variations | Sequential | Batched | DINO + DiT + decode |
|---|---|---|---|---|
| RTX 5060 Ti | 2 | 13.45 s | **11.09 s** | 6.93 s → 3.81 s |
| | 4 | 25.07 s | **16.77 s** | 13.71 s → 3.41 s |
| Radeon 8060S | 2 | 2.15 s | **1.91 s** | 1.41 s → 1.16 s |
| | 4 | 4.20 s | **3.73 s** | 2.82 s → 2.33 s |

Peak tensor allocation was unchanged on the RTX 5060 Ti (9.22 GiB) and 0.07 GiB higher at four variations on the
Radeon 8060S. Raw reports, including the per-seed token differences, are in
[measurements/image-cache](measurements/image-cache).

