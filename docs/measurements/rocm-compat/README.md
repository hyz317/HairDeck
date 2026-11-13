# ROCm compatibility regression evidence

Collected on 2026-11-13. Baseline is upstream `cb20f7073419a33c9ab88f22f686fcb9060a99f0`;
candidate implementation is `61c19348ea759aadd58563ba5abbf5a9558a282e` (the AMD report
uses its short SHA). Subsequent changes add this evidence, documentation, whitespace
cleanup and an explicit revision argument for containerized harness execution. No
model, preprocessing, sampler or export implementation differs between those revisions.

The same `tools/rocm/regression.py` harness and existing weights/environment were used
for each before/after pair. Example 2, seeds 1 and 2, 50 Euler steps, CFG 2, colors,
chains and GLB writes were enabled. Optional fork optimizations are absent. Each fresh
process recorded its first request and three warm requests, retaining disk caches.

On **each GPU separately**, all token-array, card/color JSON, chain JSON and GLB hashes
match between baseline and candidate across all four requests and both seeds.
Cross-vendor output identity is not asserted. Both revisions completed validated VRM
export, rest rendering and a 90-frame motion preview on Windows/5060 Ti and Linux/8060S.
The PGX check deliberately skipped exports because its existing ARM64 setup lacks the
managed Blender installation; it is not full ARM64 export validation.

| Device / environment | Upstream warm median | Candidate warm median | Peak warm allocated bytes, both revisions | Exports |
|---|---:|---:|---:|---|
| RTX 5060 Ti, Windows, torch 2.8.0+cu128 | 10.239 s | 9.212 s | 9,895,692,288 | VRM + rest + motion pass |
| GB10, Linux ARM64, torch 2.8.0+cu129 | 2.667 s | 2.655 s | 9,893,857,280 | Not run |
| Radeon 8060S, Linux, torch 2.13.0+git733fca1 / HIP 7.2 | 3.355 s | 3.385 s | 8,772,776,960 | VRM + rest + motion pass |

These small samples are regression observations, not a speedup claim or a GPU ranking.
The desktop session remained active on the 5060 Ti; the resident ROCm demo remained
loaded on Strix Halo. Runs were sequential per device, not randomized. Allocation
figures are PyTorch tensors in the benchmark process, not machine-wide memory.
Reserved memory, first-request costs, loading costs and individual timings are retained
in each raw report. PGX's existing PyTorch emitted an architecture-range warning for
SM 12.1, but both actual inference runs completed correctly.

The same prebuilt ROCm environment was used for both AMD revisions; the Dockerfile,
build scripts, dependency constraints and requirements match the already-built port
environment. A second full image rebuild was not performed for this PR. The GPU smoke
check passed BF16 attention, NMS, ROI Align, zero-offset deformable-convolution equivalence
and EGL rendering on the Radeon 8060S / Mesa 26.2.5 (2209 depth pixels).

A temporary candidate demo container with `GRADIO_SERVER_NAME=0.0.0.0` and host mapping
`127.0.0.1:17860:7860` returned HTTP 200. It was stopped and removed after the check;
the existing demo on port 7860 still returned HTTP 200. The ordinary local bind default
is preserved when the override is absent.

## Reports

- [Machine-readable comparison](comparison.json)
- RTX 5060 Ti: [upstream](5060-baseline.json), [candidate](5060-candidate.json)
- PGX GB10: [upstream](pgx-baseline.json), [candidate](pgx-candidate.json)
- Strix Halo: [upstream](amd-baseline.json), [candidate](amd-candidate.json)

Saved model outputs, videos and VRMs remain in the run directories recorded in the
reports. They are not duplicated in Git. For a worktree mounted without its Git
metadata, pass `--revision <source-commit>` to the harness; use the actual checkout
commit rather than a label. New runs require an unused `--out` directory.
