# JSON Configuration Documentation

This document lists the JSON parameters of all components of __tiny-cuda-nn__.

For each component, we provide a sample configuration that lists each parameter's default value.

## Networks

### Activation Functions

Activation functions are specified by string, e.g. as follows:
```json5
{
	"activation": "ReLU",
}
```

The following activation functions are supported:
- `"None"` (identity)
- `"ReLU"`
- `"LeakyReLU"` (defined as `max(0, x) + 0.01 * min(0, x)`)
- `"SiLU"` (defined as `x * sigmoid(x)`, also known as Swish)
- `"Exponential"`
- `"Sine"`
- `"Sigmoid"` (the logistic function)
- `"Squareplus"` (defined as `X = 10*x; 0.5 * (X + sqrt(X*X + 4)) / 10`)
- `"Softplus"` (defined as `X = 10*x; log(exp(X) + 1) / 10`)
- `"Tanh"` (defined as `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`)

The factor and divisor `10` in the `Squareplus` and `Softplus` activations can be thought of as "zooming out" such that these smooth activations more closely resembly the ReLU. If this is undesired in your use case, you can change the compile-time constant `K_ACT` in `include/tiny-cuda-nn/common_device.h`.

### Fully Fused MLP

Lightning fast implementation of small multi-layer perceptrons (MLPs). Restricted to hidden layers of size 16, 32, 64, or 128.

```json5
{
	"otype": "FullyFusedMLP",    // Component type.
	"activation": "ReLU",        // Activation of hidden layers.
	"output_activation": "None", // Activation of the output layer.
	"n_neurons": 128,            // Neurons in each hidden layer.
	                             // May only be 16, 32, 64, or 128.
	"n_hidden_layers": 5,        // Number of hidden layers.
}
```

### CUTLASS MLP

Multi-layer perceptron (MLP) based on [CUTLASS](https://github.com/NVIDIA/cutlass)' GEMM routines. Slower than the fully fused MLP, but allows for arbitrary numbers of hidden and output neurons. Like the fully fused MLP, it outperforms TensorFlow for small networks.

```json5
{
	"otype": "CutlassMLP",       // Component type.
	"activation": "ReLU",        // Activation of hidden layers.
	"output_activation": "None", // Activation of the output layer.
	"n_neurons": 128,            // Neurons in each hidden layer.
	"n_hidden_layers": 5         // Number of hidden layers.
}
```

## Encodings


### Composite

Allows composing multiple encodings. The following example replicates the Neural Radiance Caching [[Müller et al. 2021]](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf) encoding by composing the `TriangleWave` encoding for the first 3 (spatial) dimensions, the `OneBlob` encoding for the following 5 non-linear appearance dimensions, and the `Identity` for all remaining dimensions.

```json5
{
	"otype": "Composite",
	"nested": [
		{
			"n_dims_to_encode": 3, // Spatial dims
			"otype": "TriangleWave",
			"n_frequencies": 12
		},
		{
			"n_dims_to_encode": 5, // Non-linear appearance dims.
			"otype": "OneBlob",
			"n_bins": 4
		},
		{
			// Number of remaining linear dims is automatically derived
			"otype": "Identity"
		}
	]
}
```

### Frequency

From NeRF [[Mildenhall et al. 2020]](https://www.matthewtancik.com/nerf). Works better than OneBlob encoding if the dynamic range of the encoded dimension is high. However, suffers from stripe artifacts.

The number of encoded dimensions is twice the specified number of frequencies for each input dimension. E.g. with `n_frequencies == 4`, an input dimension `x` becomes `sin(πx), cos(πx), sin(2πx), cos(2πx), sin(4πx), cos(4πx), sin(8πx), cos(8πx)`.

Note that many NeRF implementations (including the official ones) omit the factor of `π` from eq. (4) of the paper. This makes little difference in practice as coordinate normalization usually differs by similar amounts. Due to the logarithmic scaling of this encoding, this means that one or two fewer or additional frequency bands might be required to match results across implementations.

```json5
{
	"otype": "Frequency", // Component type.
	"n_frequencies": 12   // Number of frequencies (sin & cos)
	                      // per encoded dimension.
}
```

### Grid

Encoding based on trainable multiresolution grids.
Used for [Instant Neural Graphics Primitives [Müller et al. 2022]](https://nvlabs.github.io/instant-ngp/). The grids can be backed by hashtables, dense storage, or tiled storage.

The number of encoded dimensions is `n_levels * n_features_per_level`.

```json5
{
	"otype": "Grid",           // Component type.
	"type": "Hash",            // Type of backing storage of the
	                           // grids. Can be "Hash", "Tiled"
	                           // or "Dense".
	"n_levels": 16,            // Number of levels (resolutions)
	"n_features_per_level": 2, // Dimensionality of feature vector
	                           // stored in each level's entries.
	"log2_hashmap_size": 19,   // If type is "Hash", is the base-2
	                           // logarithm of the number of elements
	                           // in each backing hash table.
	"base_resolution": 16,     // The resolution of the coarsest le-
	                           // vel is base_resolution^input_dims.
	"per_level_scale": 2.0,    // The geometric growth factor, i.e.
	                           // the factor by which the resolution
	                           // of each grid is larger (per axis)
	                           // than that of the preceding level.
	"interpolation": "Linear"  // How to interpolate nearby grid
	                           // lookups. Can be "Nearest", "Linear",
	                           // or "Smoothstep" (for smooth deri-
	                           // vatives).
}
```

### Permuto

`Permuto` is a trainable N-dimensional permutohedral lattice encoding. The
factory accepts input widths `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`,
`10`, `12`, `16`, and `24`. The input width passed to the factory is
authoritative. Callers must normalize each input value to `[0, 1]`. The
encoding does not scan device inputs to enforce this range.

Each level produces `n_features_per_level` adjacent output features. The
supported feature widths are `1`, `2`, `4`, and `8`. Level `l` uses the scale
`base_scale * per_level_scale^l`. The unpadded output width is
`n_levels * n_features_per_level`. An alignment request can add zero-valued
padding after these features. The required output alignment is
`n_features_per_level`. The encoded features have no fixed value range because
the lattice entries are trainable parameters.

```text
[x0, ..., xN-1]
       |
       v
levels 0 ... n_levels - 1
       |
       v
[level 0 feature 0, ..., level L feature F-1]
```

```json5
{
	"otype": "Permuto",            // Component type.
	"n_levels": 16,                // Number of levels. Must be in [1, 32].
	"n_features_per_level": 2,     // Must be 1, 2, 4, or 8.
	"log2_hashmap_size": 19,       // Base-2 logarithm of the entries per level.
	"base_scale": 16.0,            // Finite scale of level 0.
	"per_level_scale": 2.0,        // Positive finite scale multiplier.
	"interpolation": "Linear",      // Case-insensitive. Must be "Linear".
	"max_input_grad_dims": 5,      // Leading input dimensions with gradients.
	"seed": 1337                   // Unsigned seed for per-level lattice shifts.
}
```

`interpolation` accepts only `Linear`, case-insensitively.
`log2_hashmap_size` must be a nonnegative integer smaller than 32. The
parameter count must also fit in an unsigned 32-bit integer. Every derived
level scale must be finite and safe for the kernel's integer lattice
coordinates. `max_input_grad_dims` defaults to the input width and must not
exceed that width.
`seed` must be an integer in `[0, 2^32 - 1]`.

The flat parameter array contains
`n_levels * 2^log2_hashmap_size * n_features_per_level` values. The layout
order is level, hashed entry, then feature. The features for one entry are
adjacent. Construction, `hyperparams()`, and reload preserve this layout.
`hyperparams()` includes the configured `seed`. Reloading the emitted
hyperparameters and the same flat parameter array therefore preserves the
lattice shifts and learned values.
`scales_table` and `shifts_table` are diagnostics derived from `seed`,
`base_scale`, and `per_level_scale`. They do not independently configure the
lattice. The factory accepts the tables only when they match those behavioral
fields. Each table contains `n_levels * input_width` values.

`n_features` and `n_grid_features` are aliases for the total unpadded output
width. Either alias can replace `n_levels`. The aliases are mutually exclusive
with each other and with `n_levels`. An alias value must be a positive integer
that is divisible by `n_features_per_level`. `hyperparams()` always emits the
canonical `n_levels` and `n_features_per_level` fields. It does not emit either
alias. The factory ignores unrelated compatibility keys, including
`base_resolution`, `max_resolution`, and a JSON `n_dims_to_encode` field.

The encoding supports first-order parameter and input gradients. It also
supports non-JIT double backward for parameter gradients and upstream
gradients. The input Hessian is zero inside each linear lattice simplex.
`max_input_grad_dims` applies at both derivative orders. It enables gradients
only for the leading input dimensions. The remaining input gradients are zero.
Parameter gradients honor `Overwrite`, `Accumulate`, and `Ignore` at both
derivative orders.

### MultiLevelEncodingLoD

`MultiLevelEncodingLoD` adds per-element level control to a multilevel
encoding. The public Grid and Permuto encodings are supported bases. The
wrapper accepts one more input dimension than its base. The first N values are
the base input. The final value is a level ratio. Callers normally provide a
finite ratio in `[0, 1]`. NaN and infinite ratios are outside the public
contract.

```text
[x0, ..., xN-1, level_ratio]
          |             |
          v             v
       base input      LoD weights
          \_____________/
                 |
                 v
          weighted features
```

```json5
{
	"otype": "MultiLevelEncodingLoD",
	"lod_type": "Soft",           // Optional. Defaults to "Hard".
	"base": {
		"otype": "Permuto",        // Can also be a Grid encoding.
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 19,
		"base_scale": 16.0,
		"per_level_scale": 2.0,
		"interpolation": "Linear",
		"max_input_grad_dims": 5,
		"seed": 1337
	}
}
```

`lod_type` accepts `Hard`, `Discontinuous`, `Soft`, and `Continuous`,
case-insensitively. `Discontinuous` is an alias for `Hard`. `Continuous` is an
alias for `Soft`. `hyperparams()` serializes the canonical value `Hard` or
`Soft` and preserves the base configuration as nested JSON.

Both modes calculate the per-element level coordinate as follows:

```text
level_f = level_ratio * n_levels + 1e-3
level_i = floor(level_f)
```

Hard mode enables a complete zero-based level when the following predicate is
true:

```text
level < level_f
```

Soft mode assigns weight `1` to levels below `level_i`. Soft mode assigns
weight `level_f - level_i` to level `level_i`. Soft mode assigns weight `0` to
finer levels. A ratio below the first boundary can disable all levels. A ratio
at or above the final boundary preserves all levels. The implementation does
not clamp the ratio.

An inactive level produces zero output features and contributes no parameter
or position-input gradient. A partially active soft level applies the same
weight to its output and upstream gradient. The wrapper retains the unweighted
base output because native backward can require that value. Native double
backward applies the same soft weights to the upstream-gradient result. The
level-ratio input is scheduler state. Its input gradient is exactly zero in
both modes.

The wrapper forwards the base level count, position dimensions, features per
level, parameter offsets, output alignment, layout, and padding. The wrapper
also preserves the base logical output order, flat parameter layout, inference
parameters, and `Overwrite`, `Accumulate`, and `Ignore` gradient modes. The
wrapper passes each gradient mode unchanged to the base at both derivative
orders. The base encoding therefore defines the final gradient-mode behavior
and any base-specific limitations. In particular, a half-precision Grid with
`n_features_per_level=1` uses a temporary floating-point parameter-gradient
accumulator. Its `Accumulate` path does not initialize that scratch buffer from
the existing half-precision gradients. A Grid-backed wrapper inherits this
pre-existing limitation. A nested `MultiLevelEncodingLoD` base is rejected
because nested wrappers cannot safely share the native mutable level-selection
state.

A nonempty context-only forward is supported when the output pointer is null.
Soft mode also retains the unweighted base output for a later backward call. A
zero-element batch is valid for forward, backward, and double backward.
Empty-batch derivative calls clear parameter gradients in `Overwrite` mode and
preserve them in `Accumulate` and `Ignore` modes.

#### Thread Safety

`MultiLevelEncodingLoD` does not support concurrent host calls on one
instance. Do not start a call whose device work can overlap earlier work on
that instance. Use one CUDA stream, or insert and wait for an event dependency
before another stream accesses the same parameters, gradients, or per-call
state. Host-call serialization without a device dependency is insufficient.
These encodings do not establish a broader host-thread-safety contract for
TCNN.

#### Build and JIT Support

`Permuto` and `MultiLevelEncodingLoD` are available only in builds that contain
the offline forward and backward implementations. A build configured with
`TCNN_BUILD_NO_FWD_BWD=ON` does not register either encoding type.

Both encodings provide generated CUDA device functions for inference. The
generated `Permuto` function reproduces the native lattice lookup. The
generated `MultiLevelEncodingLoD` function calls its generated base function
and applies the hard or soft weights to the returned vector. The built-in
`Permuto` and Grid bases implement generated forward, so automatic JIT
inference remains enabled for those wrapper combinations. A custom multilevel
base must also implement generated forward, or the normal JIT fallback applies.
Direct `generate_device_function()` calls use the standard three-argument
device-function interface.

Generated training is not supported. The generated functions do not allocate
or populate a forward context. Their forward-context size remains unsupported.
An automatic JIT training `forward()` therefore reports the unsupported context
size, disables JIT fusion, and falls back before it allocates a generated
context. Native backward then consumes the native forward context. Generated
backward and generated double backward are also unsupported. The
no-forward-and-backward build cannot use the native fallback because the
encoding types are not registered.

The non-JIT `NetworkWithInputEncoding` double-backward path currently supports
one specific network shape: a `FullyFusedMLP` with one hidden layer, `ReLU`, and
no output activation. That path computes parameter and upstream gradients. It
does not compute a network input Hessian. Other network shapes report an
explicit runtime error instead of returning an incomplete second-order result.

### Identity

Leaves values untouched. Optionally, multiplies each dimension by a scalar and adds an offset.

```json5
{
	"otype": "Identity", // Component type.
	"scale": 1.0,        // Scaling of each encoded dimension.
	"offset": 0.0        // Added to each encoded dimension.
}
```

### OneBlob

From Neural Importance Sampling [[Müller et al. 2019]](https://tom94.net/data/publications/mueller18neural/mueller18neural-v4.pdf) and Neural Control Variates [[Müller et al. 2020]](https://tom94.net/data/publications/mueller20neural/mueller20neural.pdf). When the dynamic range of the encoded dimension is limited, it results in a more accurate fit than the identity encoding while not suffering from stripe artifacts like the Frequency encoding.

For performance reasons, the encoding uses a quartic kernel rather than a Gaussian kernel to compute blob integrals. We measured no loss of reconstruction quality.

```json5
{
	"otype": "OneBlob", // Component type.
	"n_bins": 16        // Number of bins per encoded dimension.
}
```

### Spherical Harmonics

A frequency-space encoding that is more suitable to direction vectors than component-wise `Frequency` or `TriangleWave` encodings.
Expects 3D inputs that represent normalized vectors `v` transformed into the unit cube as `(v+1)/2`.

The number of encoded dimensions is the degree squared.

```json5
{
	"otype": "SphericalHarmonics", // Component type.
	"degree": 4                    // The SH degree up to which
	                               // to evaluate the encoding.
	                               // Produces degree^2 encoded
	                               // dimensions.
}
```

### TriangleWave

Similar to the `Frequency` encoding, but replaces the sine function with a cheaper-to-compute triangle wave. Also omits the cosine function. Proposed in [[Müller et al. 2021]](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf). Works better than OneBlob encoding if the dynamic range of the encoded dimension is high. However, suffers from stripe artifacts.

The number of encoded dimensions is the specified number of frequencies for each input dimension.

```json5
{
	"otype": "TriangleWave", // Component type.
	"n_frequencies": 12      // Number of frequencies (triwave)
	                         // per encoded dimension.
}
```


## Losses

### L1

Standard L1 loss.

```json5
{
	"otype": "L1" // Component type.
}
```

### Relative L1

Relative L1 loss normalized by the network prediction.

```json5
{
	"otype": "RelativeL1" // Component type.
}
```

### MAPE

Mean absolute percentage error (MAPE). The same as Relative L1, but normalized by the target.

```json5
{
	"otype": "MAPE" // Component type.
}
```

### SMAPE

Symmetric mean absolute percentage error (SMAPE). The same as Relative L1, but normalized by the mean of the prediction and the target.

```json5
{
	"otype": "SMAPE" // Component type.
}
```

### L2

Standard L2 loss.

```json5
{
	"otype": "L2" // Component type.
}
```

### Relative L2

Relative L2 loss normalized by the network prediction [[Lehtinen et al. 2018]](https://github.com/NVlabs/noise2noise).

```json5
{
	"otype": "RelativeL2" // Component type.
}
```

### Relative L2 Luminance

Same as above, but normalized by the luminance of the network prediction. Only applicable when network prediction is RGB. Used in Neural Radiance Caching [[Müller et al. 2021]](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf).

```json5
{
	"otype": "RelativeL2Luminance" // Component type.
}
```

### Cross Entropy

Standard cross entropy loss. Only applicable when the network prediction is a probability density function.

```json5
{
	"otype": "CrossEntropy" // Component type.
}
```

### Variance

Standard variance loss. Only applicable when the network prediction is a probability density function.

```json5
{
	"otype": "Variance" // Component type.
}
```


## Optimizers

### Adam

Implementation of Adam [[Kingma and Ba 2014]](https://arxiv.org/abs/1412.6980), generalized to AdaBound [[Luo et al. 2019]](https://github.com/Luolc/AdaBound).

```json5
{
	"otype": "Adam",       // Component type.
	"learning_rate": 1e-3, // Learning rate.
	"beta1": 0.9,          // Beta1 parameter of Adam.
	"beta2": 0.999,        // Beta2 parameter of Adam.
	"epsilon": 1e-8,       // Epsilon parameter of Adam.
	"l2_reg": 1e-8,        // Strength of L2 regularization
	                       // applied to the to-be-optimized params.
	"relative_decay": 0,   // Percentage of weights lost per step.
	"absolute_decay": 0,   // Amount of weights lost per step.
	"adabound": false      // Whether to enable AdaBound.
}
```

### Novograd

Implementation of Novograd [[Ginsburg et al. 2019]](https://arxiv.org/abs/1905.11286).

```json5
{
	"otype": "Novograd",   // Component type.
	"learning_rate": 1e-3, // Learning rate.
	"beta1": 0.9,          // Beta1 parameter of Novograd.
	"beta2": 0.999,        // Beta2 parameter of Novograd.
	"epsilon": 1e-8,       // Epsilon parameter of Novograd.
	"relative_decay": 0,   // Percentage of weights lost per step.
	"absolute_decay": 0    // Amount of weights lost per step.
}
```

### Stochastic Gradient Descent (SGD)

Standard stochastic gradient descent (SGD).

```json5
{
	"otype": "SGD",        // Component type.
	"learning_rate": 1e-3, // Learning rate.
	"l2_reg": 1e-8         // Strength of L2 regularization.
}
```

### Shampoo

Implementation of the 2nd order Shampoo optimizer [[Gupta et al. 2018]](https://arxiv.org/abs/1802.09568) with home-grown optimizations as well as those by [Anil et al. [2020]](https://arxiv.org/abs/2002.09018).

```json5
{
	"otype": "Shampoo",              // Component type.
	"learning_rate": 1e-3,           // Learning rate.
	"beta1": 0.9,                    // Beta1 parameter similar to Adam.
	                                 // Used to exponentially average the
	                                 // first gradient moment.
	"beta2": 0.99,                   // Beta2 parameter similar to Adam.
	                                 // Used to exponentially average the
	                                 // second gradient moment.
	"beta3": 0.9,                    // Used to exponentially average L and R.
	"beta_shampoo": 0.9,             // Used to exponentially average
	                                 // Shampoo updates.
	"epsilon": 1e-8,                 // Epsilon parameter similar Adam.
	                                 // Used to avoid singularity when computing
	                                 // momentum.
	"identity": 0.01,                // Blends L and R with I*identity for
	                                 // numerical stability.
	"cg_on_momentum": true,          // Whether to estimate L and R from the
	                                 // estimated momentum or from the raw
	                                 // gradients.
	"l2_reg": 1e-5,                  // Strength of L2 regularization
	                                 // applied to the to-be-optimized params.
	"relative_decay": 0,             // Percentage of weights lost per step.
	"absolute_decay": 0,             // Amount of weights lost per step.
	"frobenius_normalization": true, // Whether to normalize update
	                                 // steps by the would-be Adam
	                                 // update's Frobenius norm.
}
```

### Average

Wraps another optimizer and computes a linear average of the optimized parameters over the last N training steps. The average is used for inference only (does not feed back into training).

```json5
{
	"otype": "Average", // Component type.
	"n_samples": 128,   // The number of steps to be averaged over.
	"nested": {         // The nested optimizer.
		"otype": "Adam"
	}
}
```

### Batched

Wraps another optimizer, invoking the nested optimizer once every N steps on the averaged gradient. Has the same effect as increasing the batch size but requires only a constant amount of memory.

```json5
{
	"otype": "Batched",          // Component type.
	"batch_size_multiplier": 16, // N from the above description
	"nested": {                  // The nested optimizer.
		"otype": "Adam"
	}
}
```

### Exponential Moving Average (EMA)

Wraps another optimizer and computes an exponential moving average of optimized parameters. The average is used for inference only (does not feed back into training).

```json5
{
	"otype": "EMA", // Component type.
	"decay": 0.99,  // The EMA's decay per step.
	"nested": {     // The nested optimizer.
		"otype": "Adam"
	}
}
```


### Exponential Decay

Wraps another optimizer and performs piecewise-constant exponential learning-rate decay.

```json5
{
	"otype": "ExponentialDecay", // Component type.
	"decay_base": 0.1,           // The amount per decay step.
	"decay_start": 10000,        // The training step at which
	                             // to start the decay.
	"decay_end": 10000000,       // The training step at which
	                             // to end the decay.
	"decay_interval": 10000,     // Training steps inbetween decay.
	"nested": {                  // The nested optimizer.
		"otype": "Adam"
	}
}
```


### Lookahead

Wraps another optimizer, implementing the lookahead algorithm [[Zhang et al. 2019]](https://arxiv.org/abs/1907.08610).

```json5
{
	"otype": "Lookahead", // Component type.
	"alpha": 0.5,         // Fraction of lookahead distance to
	                      // traverse.
	"n_steps": 16,        // Nested optimizer steps for each
	                      // Lookahead step.
	"nested": {           // The nested optimizer.
		"otype": "Adam"
	}
}
```
