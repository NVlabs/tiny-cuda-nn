# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from tinycudann_bindings import _C

class _module_func(torch.autograd.Function):
	@staticmethod
	def forward(ctx, native_tcnn_module, input, params, loss_scale):
		# params is just a dummy param for autograd, we got an internal pointer in native_tcnn_module
		output = native_tcnn_module.fwd(input, params)
		ctx.save_for_backward(input, params, output)
		ctx.native_tcnn_module = native_tcnn_module
		ctx.loss_scale = loss_scale
		return output

	@staticmethod
	def backward(ctx, doutput):
		input, weights, output = ctx.saved_tensors
		with torch.no_grad():
			scaled_grad = doutput * ctx.loss_scale
			input_grad, weight_grad = ctx.native_tcnn_module.bwd(input, weights, output, scaled_grad)
		return None, None if input_grad is None else (input_grad / ctx.loss_scale), weight_grad / ctx.loss_scale, None

class Module(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()

		initial_params = self.native_tcnn_module.initial_params(seed).cuda()
		self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		self.register_parameter(name="params", param=self.params)

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

	def _unpad_output(self, x, batch_size):
		return x[:batch_size, :self.n_output_dims]

	def forward(self, x):
		batch_size = x.shape[0]
		output = _module_func.apply(
			self.native_tcnn_module,
			x.to(torch.float).contiguous(),
			self.params.to(torch.half if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else torch.float32).contiguous(),
			self.loss_scale
		)
		return self._unpad_output(output, batch_size)

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

class NetworkWithInputEncoding(Module):
	def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, seed=1337):
		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.encoding_config = encoding_config
		self.network_config = network_config

		super(NetworkWithInputEncoding, self).__init__()

	def _native_tcnn_module(self):
		return _C.Module(self.n_input_dims, self.n_output_dims, self.encoding_config, self.network_config)

class Network(Module):
	def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.network_config = network_config

		super(Network, self).__init__()

	def _native_tcnn_module(self):
		return _C.Module(self.n_input_dims, self.n_output_dims, self.network_config)

class Encoding(Module):
	def __init__(self, n_input_dims, encoding_config, seed=1337):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config

		super(Encoding, self).__init__()

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.Module(self.n_input_dims, self.encoding_config)
