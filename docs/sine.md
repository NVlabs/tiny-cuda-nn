# FullyFusedMLP Sine

Measured repetitions, quality curves, numerical diagnostics and source identities
are included in the [measurement evidence](benchmarks/sine/README.md). The main
experiments use RTX5090; final-source cross-architecture acceptance is reported
separately so historical snapshots are not mistaken for the submitted source.

## Configuration

```json
{
  "otype": "FullyFusedMLP",
  "activation": "Sine",
  "output_activation": "None",
  "n_neurons": 128,
  "n_hidden_layers": 5,
  "sine_backward_mode": "SavedZ"
}
```

`SavedZ` is the default. Native FP16 inference, forward, parameter gradients, and
input gradients support the existing FullyFusedMLP widths 16, 32, 64, and 128.
The usual native requirements still apply: suitable shared-memory capacity and a
positive batch size divisible by `BATCH_SIZE_GRANULARITY` (256).
Input/output padding and weight initialization follow the existing network code;
this change does not introduce biases or a new Sine frequency multiplier.
Sine output activations also save their pre-activation for first-order backward.
Sine double backward is not added by this change.

## Backward Modes

For a hidden layer, native SavedZ stores the FP16 pre-activation `z` alongside the
existing FP16 activation `h`. Backward multiplies the propagated FP16 gradient by
an FP16-rounded `cos(z)`. Hidden dH and activation backward remain fused; dW uses
the original independent GEMM and split-K reduction path.

Opt-in `"sine_backward_mode": "SignRecon"` reuses the saved FP16 `h` and stores one
cosine-sign bit per activation instead of an additional FP16 `z`. The magnitude
is reconstructed in FP32 and rounded to FP16 before gradient multiplication:

```text
cosine = half(sign * sqrt(max(0, 1 - float(h) * float(h))))
```

This is an approximate derivative, not a numerically equivalent implementation.
In particular, when `sin(z)` rounds to FP16 +1 or -1, reconstruction produces zero
even if SavedZ's cosine is nonzero. Larger loss scaling can reduce downstream
FP16 gradient underflow, but cannot recover information lost in `h`.
Use SavedZ when derivative accuracy is more important than reduced state traffic.

SignRecon currently requires all of the following:

- FP16 native FullyFusedMLP, hidden width 128 and **encoded network input width 128**.
- Sine hidden activations and linear (`None`) output with 1 through 16 dimensions.
- Contiguous column-major network input for training. The built-in OneBlob 64-bin
  encoding of two coordinates produces the required 128-wide input.
- Validated device compute capability **80, 89, or 120**. This is not a promise of
  support for every GPU in the corresponding architectural families.
- Native execution: set `jit_fusion` to false. Device-function generation rejects
  SignRecon; automatic JIT error handling can warn and fall back to native execution.

The packed state uses the producer accumulator's lane/element correspondence,
consumed by the matching eight entries of the backward matrix-A fragment. This
correspondence is implementation-dependent; the coordinate probe below checks it
on each validated architecture. Unsupported configurations fail explicitly rather
than silently switching derivative modes. Inference needs neither z nor signs.
Mode serialization is explicit; old configurations without the field use SavedZ.
Validated builds used CUDA12.8 with native SM80/SM89/SM120 targets. This is not
certification of mixed-target fatbins, older-target PTX fallback, or all compiler
versions. SavedZ Sine output backward requires contiguous output gradients with
the same layout as the saved output pre-activation.

## Tests

No image, checkpoint, network access, or Python package is needed for CTest:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTCNN_BUILD_TESTS=ON \
  -DTCNN_BUILD_EXAMPLES=OFF -DTCNN_CUDA_ARCHITECTURES=89
cmake --build build --target test_sine probe_sine_fragment_layout -j 2
ctest --test-dir build -R '^(test_sine|probe_sine_fragment_layout)$' --output-on-failure
```

Choose the architecture appropriate to your GPU; the probe target requires native
FP16 FullyFusedMLP to be compiled. A portable evidence-collecting wrapper is also
available: `python scripts/run_sine_tests.py --build build --output results/tests
--cuda-architectures 89` (use a new result directory).

The suite enumerates 63,488 finite FP16 angle values, compares each derivative to
its declared scalar semantics, and uses an independent CPU-double chain rule with
explicit FP16 rounding for controlled network fixtures. Parameter-gradient gates
are relative L2 <= 1% and max absolute <= 1e-4 + 0.005 * reference max. Fixtures
cover widths, depths, outputs, input gradients, row-major SavedZ, Sine output,
null forward outputs, live contexts, zero gradients, and gradient accumulation.
The sparse controlled weights isolate layout and activation errors; passing them
does not establish arbitrary trained-model equivalence. A hidden measurement tag
can report mode differences on external image checkpoints without imposing an
equivalence gate. The fragment probe checks 393,216 consumed positions and guards.

## Reproducing Image Measurements

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTCNN_BUILD_BENCHMARK=ON \
  -DTCNN_BUILD_EXAMPLES=OFF -DTCNN_CUDA_ARCHITECTURES=89
cmake --build build --target bench_image_ours bench_image_quality -j 2
python scripts/benchmark_sine.py --executable build/bench_image_ours \
  --image input-1024.png --output results/throughput
python scripts/run_image_quality.py --build build --skip-build \
  --input-image input-1024.png --output results/quality
```

Quality analysis needs NumPy, Pillow, and Matplotlib. The quality script can also
square-resize the repository's Einstein image to 1024 x 1024 with Pillow LANCZOS;
reuse the same RGB8 PNG to compare runs. It uses the supplied OneBlob Sine config,
million-sample batches, paired seeds 1337/2026/3407, 10,000 steps, and evaluation
every 100 steps. Its 125-step warmup restores weights, optimizer state, and RNG
before timing. Training time excludes evaluation and export. It verifies final
prediction round trips and best-checkpoint quality after model reload. Gamma 2.2
RGB8 PSNR is reported separately from sampled RelativeL2 training loss.
The benchmark exports inference-only packets with a canonical SavedZ decoder
configuration (also for SignRecon-trained weights). Reload checks validate
decoding, not resumed SignRecon optimizer trajectories; ordinary network
hyperparameter serialization still records the selected training mode.

The throughput wrapper retains the original native benchmark's sampling, loss
readback, first-half timing exclusion, and cooldowns. It rotates methods across
at least three fresh-process rounds and reports medians and ranges. These are
task-level measurements, not isolated kernel latencies; the benchmark's per-batch
step budgets differ. This change does not claim that throughput determines
time-to-target-quality, or that SignRecon always preserves convergence.

## Candidate Spot Check

### RTX 5090 (SM120)

CUDA 12.8, native FP16, OneBlob64, width128, five Sine hidden layers, linear RGB.
Median training / inference ms per batch, three fresh-process rounds:

| Batch | SavedZ | SignRecon | Cutlass Sine |
| ---: | ---: | ---: | ---: |
| 65,536 | 0.376101 / 0.085126 | 0.332312 / 0.085140 | 0.665882 / 0.140429 |
| 1,048,576 | 6.336120 / 1.147856 | 5.513823 / 1.147614 | 11.196911 / 2.455307 |

Numerical CTest passed in 2.56 s (288,676 assertions, all 393,216 consumed
fragment positions). The paired seed1337, 1,000-step quality/reload check passed;
all 11 observed PSNR checkpoints and initial weights matched the historical
main-platform run. This short run does not replace the full multi-seed study.

At batch65,536, an additional same-GPU three-round old/new SignRecon comparison
found a small training regression: old4.10 0.323636 ms, final candidate
0.331643 ms (+2.47%). Inference was effectively unchanged. This cost is reported
rather than dismissed as historical timing noise; its kernel/host cause was not
isolated. The million-batch spot check was slightly faster than the old median.

### RTX 4090 (SM89)

RTX 4090 (SM89), CUDA 12.8, native FP16, width 128, five Sine hidden layers,
OneBlob64, linear RGB output; three independent-process rounds. Median
training / inference ms per batch, without a profiler:

| Batch | SavedZ | SignRecon | Cutlass Sine |
| ---: | ---: | ---: | ---: |
| 65,536 | 0.676922 / 0.096906 | 0.546720 / 0.097037 | 0.732367 / 0.149431 |
| 1,048,576 | 10.422452 / 1.489790 | 9.029717 / 1.490551 | 17.159800 / 3.827695 |

SignRecon's training speedups over SavedZ are 1.24x / 1.15x; over Cutlass,
1.34x / 1.90x. Inference uses the same no-state Sine path for both modes;
small inference timing differences are not advertised as mode improvements.
One-seed, 1,000-step image reload smoke checks passed; they do not replace
multi-seed convergence evaluation. In that short run, SignRecon used less
training time but first reached 20 dB later in wall-clock training time
(5.411 s versus 5.199 s), illustrating why throughput alone is insufficient.

### A100 (SM80)

The same final core/test source passed a fresh A100-PCIE-40GB (SM80) build with
CUDA 12.8: CTest 5.21 s, 288,676 numerical assertions and all 393,216 consumed
fragment positions. Million-batch median training / inference times across
three fresh-process rounds were SavedZ 9.186769 / 2.374392 ms, SignRecon
8.483124 / 2.372817 ms, and Cutlass 15.308207 / 3.602230 ms. SignRecon training
speedups were 1.083x versus SavedZ and 1.805x versus Cutlass. This minimal final
acceptance does not replace the historical multi-seed quality study, or imply
that the cleaned source has been revalidated on every supported device/toolchain.
