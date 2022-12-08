#!/usr/bin/env python3

import torch
from tqdm import tqdm
import numpy as np
try:
	import tinycudann as tcnn
except:
	pass

# This script stress-tests the GPU memory arena of tiny-cuda-nn with randomly sized allocations and helped
# find a bug in its interval arithmetic in the past.

class TcnnFCBlock(tcnn.Network):
	def __init__(
		self, in_features, out_features,
		num_hidden_layers, hidden_features,
		activation:str='ReLU', last_activation:str='None',
		seed=42):
		assert hidden_features in [16, 32, 64, 128], "hidden_features can only be 16, 32, 64, or 128."
		super().__init__(in_features, out_features, network_config={
			"otype": "FullyFusedMLP",               # Component type.
			"activation": activation,               # Activation of hidden layers.
			"output_activation": last_activation,   # Activation of the output layer.
			"n_neurons": hidden_features,           # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
			"n_hidden_layers": num_hidden_layers,   # Number of hidden layers.
		}, seed=seed)

	def forward(self, x: torch.Tensor):
		prefix = x.shape[:-1]
		return super().forward(x.flatten(0,-2)).unflatten(0, prefix)

device = torch.device('cuda:0')
mlp = TcnnFCBlock(3, 256, 8, 128)

for _ in range(10000):
	for n, p in mlp.named_parameters():
		p.grad = None
	_x = np.random.randint(200, 1000, 1)[0]
	x = torch.rand([_x,1000,3], dtype=torch.float, device=device) # random setting
	#x = torch.rand([torch.randint(200,800,[1]).item(),100,3], dtype=torch.float, device=device) # setting 2
	y = mlp.forward(x)
	y.mean().backward()
