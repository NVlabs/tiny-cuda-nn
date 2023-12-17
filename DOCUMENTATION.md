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
