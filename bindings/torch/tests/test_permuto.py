# Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest

import torch
import tinycudann as tcnn


ENCODING_CONFIG = {
	"otype": "MultiLevelEncodingLoD",
	"lod_type": "Hard",
	"base": {
		"otype": "Permuto",
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 19,
		"per_level_scale": 1.4472692374403782,
		"base_scale": 16.0,
		"interpolation": "Linear",
		"max_input_grad_dims": 3,
	},
}

NETWORK_CONFIG = {
	"otype": "FullyFusedMLP",
	"activation": "ReLU",
	"output_activation": "none",
	"n_neurons": 64,
	"n_hidden_layers": 1,
}


class TestPermuto(unittest.TestCase):
	def test_first_order_construction_backward_and_optimizer(self) -> None:
		self.assertTrue(torch.cuda.is_available())

		module = tcnn.NetworkWithInputEncoding(
			n_input_dims=6,
			n_output_dims=16,
			encoding_config=ENCODING_CONFIG,
			network_config=NETWORK_CONFIG,
			seed=42,
		)
		module.jit_fusion = False

		self.assertEqual(module.params.numel(), 16_780_288)

		torch.manual_seed(7)
		positions = torch.rand((128, 5), device="cuda", dtype=torch.float32)
		ratios = torch.tensor(
			[0.0, 10 / 16, 12 / 16, 15 / 16, 1.0],
			device=positions.device,
			dtype=torch.float32,
		).repeat(26)[:128, None]
		inputs = torch.cat((positions, ratios), dim=1).requires_grad_()

		optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
		params_before = module.params.detach().clone()
		output = module(inputs)
		self.assertEqual(output.shape, (128, 16))
		self.assertTrue(torch.isfinite(output).all().item())

		output.float().square().mean().backward()
		self.assertIsNotNone(inputs.grad)
		self.assertTrue(torch.isfinite(inputs.grad).all().item())
		self.assertGreater(torch.count_nonzero(inputs.grad[:, :3]).item(), 0)
		self.assertEqual(torch.count_nonzero(inputs.grad[:, 3:]).item(), 0)

		self.assertIsNotNone(module.params.grad)
		self.assertTrue(torch.isfinite(module.params.grad).all().item())
		self.assertGreater(torch.count_nonzero(module.params.grad).item(), 0)

		optimizer.step()
		self.assertTrue(torch.isfinite(module.params).all().item())
		self.assertFalse(torch.equal(params_before, module.params.detach()))

	def test_double_backward_and_no_input_gradient_property(self) -> None:
		module = tcnn.NetworkWithInputEncoding(
			n_input_dims=6,
			n_output_dims=16,
			encoding_config=ENCODING_CONFIG,
			network_config=NETWORK_CONFIG,
			seed=42,
		)
		module.jit_fusion = False
		self.assertFalse(module.backward_backward_input_no_input_grad)

		torch.manual_seed(11)
		inputs = torch.rand((128, 6), device="cuda", dtype=torch.float32)
		inputs[:, -1] = 12 / 16
		first_order_probe = torch.randn((128, 16), device=inputs.device, dtype=module.dtype) * 0.01
		second_order_probe = torch.randn_like(inputs) * 0.01

		gradients = []
		for skip_input_gradient in (False, True):
			module.backward_backward_input_no_input_grad = skip_input_gradient
			x = inputs.detach().clone().requires_grad_()
			output = module(x)
			input_gradient = torch.autograd.grad(
				output, x, first_order_probe, create_graph=True
			)[0]
			gradient_loss = (input_gradient * second_order_probe).sum()
			gradients.append(
				torch.autograd.grad(
					gradient_loss, (module.params, x), allow_unused=True
				)
			)

		(params_gradient, input_gradient), (params_gradient_optimized, input_gradient_optimized) = gradients

		self.assertTrue(torch.isfinite(params_gradient).all().item())
		self.assertGreater(torch.count_nonzero(params_gradient).item(), 0)
		self.assertTrue(torch.allclose(params_gradient, params_gradient_optimized, atol=1e-5, rtol=1e-3))
		self.assertIsNotNone(input_gradient)
		self.assertEqual(torch.count_nonzero(input_gradient).item(), 0)
		self.assertIsNone(input_gradient_optimized)

	def test_double_backward_network_parameter_endpoints(self) -> None:
		for encoded_width, hidden_width, output_width, batch_size in (
			(16, 16, 1, 1),
			(48, 128, 17, 257),
		):
			with self.subTest(
				encoded_width=encoded_width,
				hidden_width=hidden_width,
				output_width=output_width,
				batch_size=batch_size,
			):
				encoding_config = {
					**ENCODING_CONFIG,
					"base": {
						**ENCODING_CONFIG["base"],
						"n_levels": encoded_width // 2,
						"log2_hashmap_size": 8,
						"base_scale": 4.0,
						"per_level_scale": 1.2,
					},
				}
				network_config = {**NETWORK_CONFIG, "n_neurons": hidden_width}
				module = tcnn.NetworkWithInputEncoding(
					n_input_dims=6,
					n_output_dims=output_width,
					encoding_config=encoding_config,
					network_config=network_config,
					seed=42,
				)
				module.jit_fusion = False
				module.backward_backward_input_no_input_grad = True
				self.assertEqual(module.dtype, torch.float16)

				padded_output_width = (output_width + 15) // 16 * 16
				input_weight_count = hidden_width * encoded_width
				network_param_count = input_weight_count + padded_output_width * hidden_width
				encoding = tcnn.Encoding(6, encoding_config, seed=42, dtype=torch.float16)
				encoding.jit_fusion = False
				encoding.backward_backward_input_no_input_grad = True
				self.assertEqual(module.params.numel(), network_param_count + encoding.params.numel())

				torch.manual_seed(encoded_width)
				# Keep alternating ReLU masks away from zero so TCNN and PyTorch
				# cannot select different branches because of FP16 accumulation order.
				with torch.no_grad():
					controlled_input_weights = module.params[:input_weight_count].view(hidden_width, encoded_width)
					controlled_input_weights.zero_()
					hidden_indices = torch.arange(hidden_width, device=module.params.device)
					controlled_input_weights[hidden_indices, hidden_indices % encoded_width] = 1.0
					controlled_input_weights[1::2].mul_(-1.0)
					module.params[input_weight_count:network_param_count].fill_(0.0625)
					module.params[network_param_count:].uniform_(0.05, 0.15)
					encoding.params.copy_(module.params[network_param_count:])
				encoding.params.requires_grad_(False)

				inputs = torch.rand((batch_size, 6), device="cuda", dtype=torch.float32)
				inputs[:, -1] = 1.0
				inputs.requires_grad_()
				input_direction = torch.randn_like(inputs) * 0.01
				output_probe = torch.randn(
					(batch_size, output_width), device=inputs.device, dtype=torch.float16
				) * 0.01
				input_gradient = torch.autograd.grad(
					module(inputs), inputs, output_probe, create_graph=True
				)[0]
				actual = torch.autograd.grad(
					(input_gradient * input_direction).sum(), module.params
				)[0][:network_param_count]

				# Obtain the encoded input direction for an independent network oracle.
				encoding_inputs = inputs.detach().requires_grad_()
				encoded = encoding(encoding_inputs)
				jacobian_probe = torch.zeros_like(encoded, requires_grad=True)
				encoded_input_gradient = torch.autograd.grad(
					encoded, encoding_inputs, jacobian_probe, create_graph=True
				)[0]
				encoded_direction = torch.autograd.grad(
					(encoded_input_gradient * input_direction).sum(), jacobian_probe
				)[0]

				network_params = module.params[:network_param_count].detach().to(torch.float16)
				input_weights = network_params[:input_weight_count].view(hidden_width, encoded_width)
				output_weights = network_params[input_weight_count:].view(padded_output_width, hidden_width)
				preactivation = encoded.detach().matmul(input_weights.T)
				self.assertGreater(preactivation.abs().min().item(), 0.01)
				hidden_mask = preactivation > 0
				self.assertGreater(torch.count_nonzero(hidden_mask).item(), 0)
				self.assertGreater(torch.count_nonzero(~hidden_mask).item(), 0)
				padded_output_probe = torch.nn.functional.pad(
					output_probe, (0, padded_output_width - output_width)
				)
				hidden_direction = encoded_direction.matmul(input_weights.T) * hidden_mask
				hidden_probe = padded_output_probe.matmul(output_weights) * hidden_mask
				expected_input_gradient = hidden_probe.T.matmul(encoded_direction)
				expected_output_gradient = padded_output_probe.T.matmul(hidden_direction)
				actual_input_gradient = actual[:input_weight_count].view(hidden_width, encoded_width)
				actual_output_gradient = actual[input_weight_count:].view(padded_output_width, hidden_width)

				for gradient in (
					encoded_direction,
					expected_input_gradient,
					expected_output_gradient[:output_width],
					actual_input_gradient,
					actual_output_gradient[:output_width],
				):
					self.assertGreater(torch.count_nonzero(gradient).item(), 0)
				torch.testing.assert_close(actual_input_gradient, expected_input_gradient.float(), atol=1e-5, rtol=1e-3)
				torch.testing.assert_close(
					actual_output_gradient[:output_width],
					expected_output_gradient[:output_width].float(),
					atol=1e-5,
					rtol=1e-3,
				)
				self.assertEqual(torch.count_nonzero(actual_output_gradient[output_width:]).item(), 0)

	def test_upstream_double_gradient_without_parameter_or_input_result(self) -> None:
		module = tcnn.Encoding(
			n_input_dims=5,
			encoding_config={
				"otype": "Permuto",
				"n_levels": 4,
				"n_features_per_level": 2,
				"log2_hashmap_size": 8,
				"base_scale": 4.0,
				"per_level_scale": 1.5,
				"max_input_grad_dims": 3,
			},
			seed=17,
			dtype=torch.float32,
		)
		module.params.requires_grad_(False)
		module.backward_backward_input_no_input_grad = True

		torch.manual_seed(19)
		inputs = torch.rand((256, 5), device="cuda", dtype=torch.float32, requires_grad=True)
		output_probe = torch.randn((256, 8), device=inputs.device, dtype=module.dtype, requires_grad=True)
		second_order_probe = torch.randn((5, 256), device=inputs.device, dtype=inputs.dtype).T
		self.assertFalse(second_order_probe.is_contiguous())
		input_gradient = torch.autograd.grad(module(inputs), inputs, output_probe, create_graph=True)[0]
		upstream_gradient = torch.autograd.grad((input_gradient * second_order_probe).sum(), output_probe)[0]

		self.assertTrue(torch.isfinite(upstream_gradient).all().item())
		self.assertGreater(torch.count_nonzero(upstream_gradient).item(), 0)


if __name__ == "__main__":
	unittest.main()
