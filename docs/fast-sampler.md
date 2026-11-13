# Fast sampler for small batches

`hairdeck/sampling/fast.py` runs the same Euler / CFG sampling as `fm_sample` and `euler_cfg` for the set DiTs
(point-cloud, image and chain models), restructured for batch sizes of 1-4. At those sizes the GEMMs are small
(M = 2 x 192 tokens with CFG) and a step is bound by memory bandwidth and kernel launches rather than matrix
throughput. The same model weights and operations are used; no retraining.

1. **Condition terms once per call.** The condition does not change while sampling, so each block's cross-attention
   K / V (after their RMSNorm) and the pooled condition embedding are computed once instead of once per step.
2. **adaLN modulations for all steps in one GEMM per block.** `ada(silu(t_mlp(t) + pool))` depends only on the step,
   so it is computed for all 50 steps together (M = steps x batch) instead of one bandwidth-bound M = 2 GEMM per block
   and step.
3. **bf16 weight copies**, made once per model (and remade if its parameters change) instead of autocast casting every
   fp32 weight on every call.
4. **`torch.compile` of one step.** Inductor fuses the norms, modulations, GELU, residuals and the CFG combination
   between the GEMMs. In the eager step these small kernels are about half of the time. If compilation fails (for
   example no Triton, or a GPU the installed Triton cannot target) the sampler prints one line and continues eagerly.

## Defaults and settings

| Variable | Default | |
|---|---|---|
| `HAIRDECK_FAST` | `1` on ROCm builds of PyTorch, `0` otherwise | `1` / `0` turns the sampler on / off |
| `HAIRDECK_COMPILE` | as `HAIRDECK_FAST` | `0` keeps 1-3 without compilation |
| `HAIRDECK_GRAPHS` | `0` | `1` also replays the step as a CUDA / HIP graph (no gain measured; never under TunableOp) |

**NVIDIA defaults are unchanged**: with `HAIRDECK_FAST` unset, CUDA builds run the release sampler. On an RTX 5060 Ti,
upstream `cb20f70` and this branch gave identical token, card, chain and GLB hashes for every request and seed
(example 2, seeds 1 / 2, four requests) and identical peak allocation
([upstream](measurements/fast-sampler/rtx5060ti-default-upstream.json),
[branch](measurements/fast-sampler/rtx5060ti-default-branch.json)). Set `HAIRDECK_FAST=1` to opt in on CUDA.

The first call per model and batch size compiles: about a minute cold, a few seconds from the inductor cache on disk
afterwards (`TORCHINDUCTOR_CACHE_DIR`).

## Speed

Image pipeline, demo example 2, two variations (seeds 1 and 2), 50 steps, CFG 2, colours and chains, no file writes,
warm medians of three requests (`tools/fast_sampler/bench_image.py`):

| GPU | | Request | DINO + DiT + decode |
|---|---|---|---|
| RTX 5060 Ti, Windows, torch 2.8.0+cu128 | release sampler | 9.33 s | 4.98 s |
| | `HAIRDECK_FAST=1` | **4.01 s** | **1.57 s** |
| Radeon 8060S (Strix Halo), Linux, ROCm 7.2, torch 2.13 | release sampler (`HAIRDECK_FAST=0`) | 3.25 s | 1.58 s |
| | default | **2.27 s** | **0.81 s** |

Reports: [RTX 5060 Ti](measurements/fast-sampler/rtx5060ti-windows.json),
[Radeon 8060S](measurements/fast-sampler/radeon8060s-rocm.json).

Point-cloud DiT alone, 50 steps with CFG 2 (`tools/fast_sampler/check_fast.py`):

| GPU | Batch | Release | 1-3 eager | 1-4 compiled |
|---|---|---|---|---|
| RTX 5060 Ti | 1 | 2524 ms | 1702 ms | **561 ms** |
| | 4 | 1926 ms | 1463 ms | **628 ms** |
| Radeon 8060S | 1 | 602 ms | 413 ms | **290 ms** |
| | 4 | 1824 ms | 1321 ms | **837 ms** |

On a GB10 (DGX Spark class, torch 2.8.0+cu129) the installed Triton could not compile for compute capability 12.1
(`PTXAS error`); the sampler fell back to eager as intended and the request still got faster. The GPU was shared
with another workload during that run, so its timings are not reported.

## Numerics

- One step of the restructured forward is bitwise identical to `SetDiT.forward` under the same autocast, on both the
  RTX 5060 Ti and the Radeon 8060S (`check_fast.py`, `|fast-autocast|/|v| = 0` at t = 1, 0.5 and 0.02).
- Batching the adaLN GEMM over steps and fusing with inductor change the rounding, well below autocast's own deviation
  from fp32 (0.6-1.1% of |v|). Over 50 steps that gives a different but equally likely sample for the same seed.
- The released test-set evaluations (`scripts/test.py`, seed 0) on the Radeon 8060S with the fast sampler against the
  paper's three-seed means:

| Table 1, point clouds | CD | EMD | HD | IoU | Orient. |
|---|---|---|---|---|---|
| Paper (RTX A6000) | 0.0066 | 0.0084 | 0.0242 | 0.867 | 16.8° |
| Release sampler | 0.0065 | 0.0084 | 0.0242 | 0.868 | 16.4° |
| Fast sampler | 0.0065 | 0.0084 | 0.0241 | 0.869 | 16.4° |

| Table 4, chains on artists' cards | CD | Rec. | Prec. | Motion | Ampl. | Fit |
|---|---|---|---|---|---|---|
| Paper (RTX A6000) | 0.279 | 0.625 | 0.585 | 0.702 | 0.0167 | 0.953 |
| Release sampler | 0.277 | 0.620 | 0.596 | 0.701 | 0.0168 | 0.956 |
| Fast sampler | 0.277 | 0.620 | 0.598 | 0.702 | 0.0168 | 0.958 |

## Tried and not adopted

- **CUDA / HIP graph replay** of the step (`HAIRDECK_GRAPHS=1`): 3% faster than plain launches without compilation and
  slower with it on ROCm 7.2; it deadlocks with PyTorch TunableOp, so it is never used while TunableOp is enabled.
- **PyTorch TunableOp** for the 29 GEMM shapes, and rocBLAS instead of hipBLASLt on ROCm: within run-to-run noise.
