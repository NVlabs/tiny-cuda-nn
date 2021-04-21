# JSON Configuration Documentation

This documentation so far only contains the JSON parameters for configuring each component of __tiny-cuda-nn__.

For each component, we provide a sample configuration with each parameter's default value.

## Networks

### Fully Fused MLP

Lightning fast implementation of small multi-layer perceptrons (MLPs). Restricted to hidden layers of size 32, 64, or 128 and outputs of 16 or fewer dimensions.

```json5
{
	"otype": "FullyFusedMLP",    // Component type.
	"activation": "ReLU",        // Activation of hidden layers.
	                             // Can be "ReLU" or "Sigmoid".
	"output_activation": "None", // Activation of the output layer.
	                             // Can be "None", "ReLU", "Sigmoid",
	                             // or "Exponential".
	"n_neurons": 128,            // Neurons in each hidden layer.
	                             // May only be 32, 64 or 128.
	"n_hidden_layers": 5,        // Number of hidden layers.
	"feedback_alignment": false  // Use feedback alignment
	                             // [Lillicrap et al. 2016].
}
```

### CUTLASS MLP

Multi-layer perceptron (MLP) based on [CUTLASS](https://github.com/NVIDIA/cutlass)' GEMM routines. Slower than the fully fused MLP, but allows for arbitrary numbers of hidden and output neurons. Like the fully fused MLP, it outperforms TensorFlow for small networks.

```json5
{
	"otype": "CutlassMLP",       // Component type.
	"activation": "ReLU",        // Activation of hidden layers.
	                             // Can be "None", "ReLU", or "Sine",
	                             // or "Exponential".
	"output_activation": "None", // Activation of the output layer.
	                             // Can be "None", "ReLU", "Exponential".
	"n_neurons": 128,            // Neurons in each hidden layer.
	"n_hidden_layers": 5         // Number of hidden layers.
}
```

### CUTLASS ResNet

Fully connected residual network based on [CUTLASS](https://github.com/NVIDIA/cutlass)' GEMM routines. Only ReLU activations in the hidden layers and no activations on the output.

```json5
{
	"otype": "CutlassResNet", // Component type.
	"n_neurons": 128,         // Neurons in each hidden layer.
	"n_blocks": 2,            // Number of residual blocks
	                          // with one skip link each.
	"n_matrices_per_block": 2 // Number of ReLU->MatMul operations
	                          // per residual block.
}
```

## Encodings

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

```json5
{
	"otype": "OneBlob", // Component type.
	"n_bins": 16        // Number of bins per encoded dimension.
}
```

### Frequency

From NeRF [[Mildenhall et al. 2020]](https://www.matthewtancik.com/nerf). Works better than OneBlob encoding if the dynamic range of the encoded dimension is high. However, suffers from stripe artifacts.

```json5
{
	"otype": "Frequency", // Component type.
	"n_frequencies": 12   // Number of frequencies (sin & cos)
	                      // per encoded dimension.
}
```

### NRC Encoding

The encoding used in Neural Radiance Caching [Müller et al. 2021] (to appear). Uses an optimized Frequency encoding with triangle waves for the first 3 dimensions (assumed to be 3D position) and OneBlob encoding with quartic kernels for the remaining dimensions.

```json5
{
	"otype": "NRC",      // Component type.
	"n_frequencies": 12, // Number of frequencies (tri wave)
	                     // per encoded 3D dimension.
	"n_bins": 4          // Number of bins per oneblob-encoded
	                     // remaining dimension.
}
```


## Losses

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

Same as above, but normalized by the luminance of the network prediction. Only applicable when network prediction is RGB. Used in Neural Radiance Caching [Müller et al. 2021] (to appear).

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
