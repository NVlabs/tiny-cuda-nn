# FullyFusedMLP Sine

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

`SavedZ` is the default native FP16 backward mode. It supports the existing
FullyFusedMLP widths 16, 32, 64, and 128, including parameter and input gradients.
It saves pre-activations and evaluates cosine during backward. Input/output
padding, initialization, and positive batch granularity (256) follow existing
FullyFusedMLP conventions. No bias or frequency multiplier is introduced.

## Optional SignRecon

`"sine_backward_mode": "SignRecon"` reuses saved FP16 activations and stores
cosine-sign bits instead of additional FP16 hidden pre-activations. It reconstructs
`half(sign * sqrt(max(0, 1 - float(h) * float(h))))` before gradient multiplication.
This is approximate: when `h` rounds to +/-1, a nonzero SavedZ derivative can
become zero. Loss scaling cannot recover this discarded information.

The formula is not inherently limited to width128. This implementation currently
enables the validated native path for hidden/encoded input width128, contiguous
column-major training input, Sine hidden layers and linear output of 1-16 channels.
Validated native-target builds use CUDA12.8 and SM80/SM89/SM120. Other configurations
are rejected; other widths remain available through SavedZ. Fragment correspondence
is implementation-dependent and checked by tests, not universally guaranteed.

Disable JIT fusion for SignRecon; device-function generation rejects that mode.
Sine output backward uses SavedZ and requires contiguous gradients matching the
saved output layout. Sine double backward is not provided.

## Tests And Benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTCNN_BUILD_TESTS=ON \
  -DTCNN_BUILD_EXAMPLES=OFF -DTCNN_CUDA_ARCHITECTURES=89
cmake --build build --target test_sine probe_sine_fragment_layout -j 2
ctest --test-dir build -R '^(test_sine|probe_sine_fragment_layout)$' --output-on-failure
```

Choose the native architecture for your GPU. The probe target requires native FP16
support. `scripts/run_sine_tests.py` optionally records test metrics and hashes;
`scripts/benchmark_sine.py` repeats the existing native image benchmark without
changing its timing protocol. `data/config_oneblob_sine.json` provides its config.
Numerical tests require no image or external checkpoint. Controlled references
do not establish arbitrary dense-GEMM or optimizer equivalence.

Full experimental reports, raw data, figures and quality-study programs are kept
on the contributor's [separate evidence branch](https://github.com/RichealYoung/tiny-cuda-nn-sine/tree/evidence/fully-fused-sine),
not included in the upstream change. See its [measurement index](https://github.com/RichealYoung/tiny-cuda-nn-sine/blob/63faa6e20ba1f57bec0a61310aad8ffcfd601f51/docs/benchmarks/sine/README.md)
for measured benefits, limitations, and the small-batch SignRecon performance cost.
