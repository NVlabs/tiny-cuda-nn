#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import autograd
from torch.optim import Adam
import torch.nn.functional as F

import sys

try:
	import tinycudann as tcnn
except ImportError:
	print("This script requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

class SDF(nn.Module):
	def __init__(self, hash=True, n_levels=12, log2_hashmap_size=15, base_resolution=16, smoothstep=False) -> None:
		super().__init__()
		self.encoder = tcnn.Encoding(3, {
			"otype": "HashGrid" if hash else "DenseGrid",
			"n_levels": n_levels,
			"n_features_per_level": 2,
			"log2_hashmap_size": log2_hashmap_size,
			"base_resolution": base_resolution,
			"per_level_scale": 1.5,
			"interpolation": "Smoothstep" if smoothstep else "Linear"
		})
		self.decoder = nn.Sequential(
			nn.Linear(self.encoder.n_output_dims, 64),
			nn.ReLU(True),
			nn.Linear(64, 1)
		)

	def forward(self, x):
		encoded = self.encoder(x).to(dtype=torch.float)
		sdf = self.decoder(encoded)
		return sdf

	def forward_with_nablas(self, x):
		with torch.enable_grad():
			x = x.requires_grad_(True)
			sdf = self.forward(x)
			nablas = autograd.grad(
				sdf,
				x,
				torch.ones_like(sdf, device=x.device),
				create_graph=True,
				retain_graph=True,
				only_inputs=True)[0]
		return sdf, nablas

if __name__ == '__main__':
	"""
	NOTE: Jianfei:  I provide three testing tools for backward_backward functionality.
					Play around as you want :)
					1. test_train(): train a toy SDF model with eikonal term.
					2. grad_check(): check backward_backward numerical correctness via torch.autograd.gradcheck.
					3. vis_graph(): visualize torch compute graph
	"""

	def test_():
		device = torch.device("cuda")
		model = SDF(True, n_levels=1, log2_hashmap_size=15, base_resolution=4, smoothstep=False).to(device)
		x = (torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float, device=device)).requires_grad_(True)
		sdf, nablas = model.forward_with_nablas(x)
		autograd.grad(
			nablas,
			x,
			torch.ones_like(nablas, device=x.device),
			create_graph=False,
			retain_graph=False,
			only_inputs=True)[0]

	def test_train():
		"""
		train a toy SDF model with eikonal term.
		"""
		from tqdm import tqdm
		device = torch.device("cuda")
		model = SDF(True, 4, base_resolution=12).to(device)
		# model = SDF(False, 4, base_resolution=12).to(device)
		optimizer = Adam(model.parameters(), 2.0e-3)
		with tqdm(range(10000)) as pbar:
			for _ in pbar:
				x = torch.rand([51200,3], dtype=torch.float, device=device)
				sdf, nablas = model.forward_with_nablas(x)
				nablas_norm: torch.Tensor = nablas.norm(dim=-1)

				# eikonal term
				loss = F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				pbar.set_postfix(loss=loss.item())

	def grad_check():
		"""
		check backward_backward numerical correctness via torch.autograd.gradcheck
		"""
		import numpy as np
		from types import SimpleNamespace
		from tinycudann.modules import _module_function_backward, _module_function, _torch_precision, _C
		dtype = _torch_precision(_C.preferred_precision())
		device = torch.device("cuda")
		# NOTE: need a smaller net when gradcheck, otherwise will OOM
		model = SDF(True, n_levels=4, log2_hashmap_size=19, base_resolution=4, smoothstep=True).to(device)
		# model = SDF(True, n_levels=1, log2_hashmap_size=15, base_resolution=8, smoothstep=False).to(device)

		def apply_on_x(x):
			params = model.encoder.params.to(_torch_precision(model.encoder.native_tcnn_module.param_precision())).contiguous()
			return _module_function.apply(
				model.encoder.native_tcnn_module, x, params, 128.0
			)

		#     ✓ y       w.r.t. x        i.e. dy_dx          (passed)
		autograd.gradcheck(
			apply_on_x,
			# (torch.rand([1,3], dtype=torch.float, device=device)).requires_grad_(True),
			(torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)).requires_grad_(True),
			eps=1.0e-3)

		#     ✓ dL_dx   w.r.t. x        i.e. ddLdx_dx       (passed)
		#     ✓ dL_dx   w.r.t. dL_dy    i.e. ddLdx_ddLdy    (passed)
		autograd.gradgradcheck(
			apply_on_x,
			# (torch.rand([1,3], dtype=torch.float, device=device)).requires_grad_(True),
			(torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)).requires_grad_(True),
			eps=1.0e-3,
			nondet_tol=0.001 # due to non-determinism of atomicAdd
			)


		def backward_apply_on_x(x):
			dL_dy = torch.ones([*x.shape[:-1], model.encoder.n_output_dims], dtype=dtype, device=device)
			params = model.encoder.params.to(_torch_precision(model.encoder.native_tcnn_module.param_precision())).contiguous()
			native_ctx, y = model.encoder.native_tcnn_module.fwd(x, params)
			dummy_ctx_fwd = SimpleNamespace(
				native_tcnn_module=model.encoder.native_tcnn_module,
				loss_scale=model.encoder.loss_scale,
				native_ctx=native_ctx)
			return _module_function_backward.apply(dummy_ctx_fwd, dL_dy, x, params, y)

		def backward_apply_on_params(params):
			x = (torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)).requires_grad_(True)
			dL_dy = torch.ones([*x.shape[:-1], model.encoder.n_output_dims], dtype=dtype, device=device)
			params = params.to(_torch_precision(model.encoder.native_tcnn_module.param_precision())).contiguous()
			native_ctx, y = model.encoder.native_tcnn_module.fwd(x, params)
			dummy_ctx_fwd = SimpleNamespace(
				native_tcnn_module=model.encoder.native_tcnn_module,
				loss_scale=model.encoder.loss_scale,
				native_ctx=native_ctx)
			return _module_function_backward.apply(dummy_ctx_fwd, dL_dy, x, params, y)

		def backward_apply_on_dLdy(dL_dy):
			x = (torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)).requires_grad_(True)
			# params = model.encoder.params.data.to(_torch_precision(model.encoder.native_tcnn_module.param_precision())).contiguous()
			params = model.encoder.params.to(_torch_precision(model.encoder.native_tcnn_module.param_precision())).contiguous()
			native_ctx, y = model.encoder.native_tcnn_module.fwd(x, params)
			dummy_ctx_fwd = SimpleNamespace(
				native_tcnn_module=model.encoder.native_tcnn_module,
				loss_scale=model.encoder.loss_scale,
				native_ctx=native_ctx)
			return _module_function_backward.apply(dummy_ctx_fwd, dL_dy, x, params, y)

		# NOTE: partial passed (Jacobian mismatch for output 1 with respect to input 0, which is ddLdgrid_dx)
		#     ✓ dL_dx       w.r.t. x     i.e. ddLdx_dx (passed)
		#     ✓ dL_dgrid    w.r.t. x     i.e. ddLdgrid_dx (currently do not support second order gradients from grid's gradient.)
		# autograd.gradcheck(
		#     backward_apply_on_x,
		#     # (torch.rand([1,3], dtype=torch.float, device=device)).requires_grad_(True),
		#     (torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)).requires_grad_(True),
		#     eps=1.0e-4
		# )

		# NOTE: passed
		#     ✓ dL_dx       w.r.t. grid     i.e. ddLdx_dgrid (passed)
		#     ✓ dL_dgrid    w.r.t. grid     i.e. ddLdgrid_dgrid (all zero)
		autograd.gradcheck(
			backward_apply_on_params,
			model.encoder.params,
			eps=1.0e-3
		)

		# NOTE: partial passed (Jacobian mismatch for output 1 with respect to input 0, which is ddLdgrid_ddLdy)
		#     ✓ dL_dx       w.r.t. dL_dy     i.e. ddLdx_ddLdy (passed)
		#     x dL_dgrid    w.r.t. dL_dy     i.e. ddLdgrid_ddLdy (currently do not support second order gradients from grid's gradient.)
		autograd.gradcheck(
			backward_apply_on_dLdy,
			torch.randn([1,model.encoder.n_output_dims], dtype=dtype, device=device).requires_grad_(True),
			eps=1.0e-3, atol=0.01, rtol=0.001
		)

	def vis_graph():
		"""
		visualize torch compute graphs
		"""
		import torchviz
		device = torch.device("cuda")
		# NOTE: need a smaller net when gradcheck, otherwise will OOM
		model = SDF(True, n_levels=4, log2_hashmap_size=15, base_resolution=4).to(device)
		x = torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float, device=device)
		sdf, nablas = model.forward_with_nablas(x)
		torchviz.make_dot(
			(nablas, sdf, x, model.encoder.params, *list(model.decoder.parameters())),
			{'nablas': nablas, 'sdf': sdf, 'x': x, 'grid_param': model.encoder.params,
			 **{n:p for n, p in model.decoder.named_parameters(prefix='decoder')}
			}).render("attached", format="png")

	def check_throw():
		network = tcnn.Network(3, 1, network_config={
			"otype": "FullyFusedMLP",               # Component type.
			"activation": 'ReLU',               # Activation of hidden layers.
			"output_activation": 'None',   # Activation of the output layer.
			"n_neurons": 64,           # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
			"n_hidden_layers": 5,   # Number of hidden layers.
		}, seed=42)


if __name__ == "__main__":
	# test_()
	test_train()
	# grad_check()
	# vis_graph()
	# check_throw()
