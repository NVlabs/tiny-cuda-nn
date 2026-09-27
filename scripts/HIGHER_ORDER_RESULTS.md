# Parameter-gradient second-order boundary

Tested against NVlabs/tiny-cuda-nn `0109538c37ac0bf613f2bac8de6cda48352feca7`
with the locally built SM120 extension, CUDA 13.0, PyTorch 2.13.0+cu130,
and one RTX 5090. Existing PR #370 proposes CutlassMLP second derivatives;
this change is limited to the current Python binding's unsupported
parameter-gradient direction.

## Observed support

D0 is forward; D1 is input VJP; D2–D4 are derivatives of the input VJP
with respect to its cotangent, input, and parameters. D5 differentiates a
parameter gradient again.

| Backend | Requested JIT | D0/D1 | D2–D4 | D5 before change |
|---|---|---|---|---|
| HashGrid | off | finite | runs; norms 0.0113 / 0.1993 / 241.77, numerical parity unvalidated | silently returns zero |
| HashGrid | on | finite | runs; similar norms, numerical parity unvalidated | silently returns zero |
| FullyFusedMLP | off / on | forward and first backward run | native `NetworkWithInputEncoding does not support double backward` | untested after D2–D4 failure |
| CutlassMLP | off / on | forward and first backward run | same native unsupported error on this upstream commit | untested after D2–D4 failure |

For HashGrid with repeated `x=[0.33,0.44,0.55]`, the baseline derivative
of `||∂(v·f)/∂params||²` with respect to x was exactly zero. Centered
finite differences along the first input coordinate were 104085, 53898,
and 77035 at steps 0.001, 0.003, and 0.01. FP16 quantization makes their
magnitudes step-dependent, but the direction is clearly not zero. The
binding had materialized missing gradients as zero tensors, then ignored
the parameter-gradient cotangent. The change disables materialization for
that autograd function and raises `NotImplementedError` on D5.

The input-gradient derivative-loss step still produces a finite,
nonzero parameter gradient with JIT off and on. The new three-test
regression passes on the changed Python binding and fails twice on the
unchanged binding, once for each JIT mode.

## Hot-path timing

HashGrid derivative-loss gradient, 128 inputs, JIT off, one GPU. Ten
paired runs of 200 calls each followed 20 warmup calls. The timer includes
Python, native forward/backward, and final CUDA synchronization.

| Binding | Median per call | Relative throughput |
|---|---:|---:|
| Baseline | 4617.6 μs | 1.00× |
| Boundary check | 4574.9 μs | 1.01× |

The small difference is within run variation; this change makes no
speedup claim. Individual medians were measured with identical initial
parameters and outputs/parameter gradients checked for parity.
