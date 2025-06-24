#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import autograd
from torch.optim import Adam
import torch.nn.functional as F

import sys
import os
import numpy as np

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
			"n_features_per_level": 8,
			"log2_hashmap_size": log2_hashmap_size,
			"base_resolution": base_resolution,
			"per_level_scale": 1.5,
			"interpolation": "Smoothstep" if smoothstep else "Linear"
		})

		# 8 layers in total, 7 hidden layers
		self.decoder = nn.Sequential(
			nn.Linear(self.encoder.n_output_dims, 64),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64),
			nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 1),
			#nn.Softplus(beta = 10.0)
		)
		
		for i in range(0, len(self.decoder), 2):
			if i % 2 == 0:
				self.decoder[i].weight.data.fill_(0.005)
				#self.decoder[i].bias.data.fill_(0.0)
		
	def forward(self, x):
		enc_output = self.encoder(x).float()
		mlp_input = enc_output

		p = []
		output = []
		for i in range(0, len(self.decoder), 2):
			tmp_p = self.decoder[i](mlp_input)
			tmp_output = self.decoder[i+1](tmp_p)

			p.append(tmp_p)
			output.append(tmp_output)
			mlp_input = tmp_output

		return enc_output, p, output

	def forward_with_nablas(self, x):
		with torch.enable_grad():
			x = x.requires_grad_(True)
			mlp_input, p, output = self.forward(x)
			output_last = output[-1]

			device = torch.device('cuda:0')
			L1 = 2.0 * torch.sum(output_last) #torch.sum(torch.pow(output - GT_64, 2.0))
			dL1doutput = (autograd.grad(
				L1,
				output_last,
				retain_graph=True
			)[0]).requires_grad_(True)

			nablas = autograd.grad( # dL1dinput aka dL1dmlp_input
				output_last,
				mlp_input, # x
				dL1doutput,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0].requires_grad_(True)

			dL1_dinput = autograd.grad( # dL1dinput aka dL1dmlp_input
				output_last,
				mlp_input, # x
				dL1doutput,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0].requires_grad_(True)

			dL1doutput_list = []
			for i in range(0, len(self.decoder), 2):
				idx = int(i / 2)
				if idx == int(len(self.decoder)/2 - 1):
					input = mlp_input
				else:
					input = output[-(idx+2)]
				output_current = output[-(idx+1)]

				dL1doutput_list.append(dL1doutput)
				dL1dinput = (autograd.grad(
					output_current,
					input,
					dL1doutput,
					retain_graph=True,
					create_graph=True,
				)[0]).requires_grad_(True)

				dL1doutput = dL1dinput
			
			dL1_dinput = dL1doutput
			
			# second order
			L2 = torch.sum(dL1_dinput) #torch.sum(dL1_dinput * dL1_dinput)
			dL2_ddL1dinput = autograd.grad(
				L2,
				dL1_dinput,
				create_graph=True,
				retain_graph=True,
			)[0].requires_grad_(True)
			
			dL2_ddL1doutput_list = []
			dL2_dinput_list = []
			for i in range(0, len(self.decoder), 2):
				idx = int(i / 2)
				dL1doutput = dL1doutput_list[-(idx+1)]
				if idx == 0:
					input = mlp_input
				else:
					input = output[idx-1]

				dL2_ddL1doutput = autograd.grad(
					L2,
					dL1doutput,
					#create_graph=True,
					retain_graph=True
				)[0].requires_grad_(True)

				dL2_dinput = autograd.grad(
					L2,
					input,
					#create_graph=True,
					retain_graph=True
				)[0].requires_grad_(True)

				dL2_ddL1doutput_list.append(dL2_ddL1doutput)
				dL2_dinput_list.append(dL2_dinput)
			
		return mlp_input, p, output, nablas, dL1doutput_list, dL2_ddL1doutput_list, dL2_dinput_list

class SDF_multi(nn.Module): # for derivation
	def __init__(self, hash=True, n_levels=12, log2_hashmap_size=15, base_resolution=16, smoothstep=False) -> None:
		super().__init__()

		self.encoder = tcnn.Encoding(3, {
			"otype": "HashGrid" if hash else "DenseGrid",
			"n_levels": 16, #n_levels,
			"n_features_per_level": 2, #8,
			"log2_hashmap_size": 19, #log2_hashmap_size,
			"base_resolution": 16, #base_resolution,
			"per_level_scale": np.exp2(np.log2(2048 * 1 * 1 / 16) / (16 - 1)), #1.5,
			#"interpolation": "Smoothstep" if smoothstep else "Linear"
		})

		# 8 layers in total, 7 hidden layers
		b_flag = False
		self.decoder = nn.Sequential(
			nn.Linear(self.encoder.n_output_dims, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 1, bias=b_flag),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 64),
			#nn.Softplus(beta = 10.0),
			#nn.Linear(64, 1),
			#nn.Softplus(beta = 10.0)
		)
		'''
		# read params from numpy files
		params_input = np.load('params_input.npy', allow_pickle=True) # init for input layers
		params = np.load('params.npy', allow_pickle=True) # init for hidden + output layers
		idx = 0
		for m in self.decoder.modules():
			if isinstance(m, nn.Linear):
				if idx == 0:
					m.weight.data = torch.from_numpy(params_input[idx])
				else:
					m.weight.data = torch.from_numpy(params[idx-1])
					#m.bias.data = torch.from_numpy(params[idx + 1])
				idx += 1
		
		#import numpy as np
		file_dir = os.getcwd()
		enc_params = np.load(os.path.join(file_dir, 'params_encoder.npy'))
		self.encoder.params = torch.nn.Parameter(torch.from_numpy(enc_params))
		#print("encoder.params: ", self.encoder.params)
		'''

		params_enc = np.load('numpy/params_sdf_enc.npy')
		self.encoder.params.data = torch.from_numpy(params_enc)

		idx = 0
		for m in self.decoder.modules():
			if isinstance(m, nn.Linear):
				if idx == 0:
					params_input = np.load("numpy/params_input.npy")
					m.weight.data = torch.from_numpy(params_input)
				elif idx == 1:
					params_hidden_1 = np.load("numpy/params_hidden_1.npy")
					m.weight.data = torch.from_numpy(params_hidden_1)
				elif idx == 2:
					params_hidden_2 = np.load("numpy/params_hidden_2.npy")
					m.weight.data = torch.from_numpy(params_hidden_2)

				else:
					params_output = np.load("numpy/params_output.npy")
					m.weight.data = torch.from_numpy(params_output)
				idx += 1
		
		
	def forward(self, x):
		enc_output = self.encoder(x).float()
		mlp_input = enc_output

		p = []
		output = []
		for i in range(0, len(self.decoder), 2):
			tmp_p = self.decoder[i](mlp_input)
			if (len(self.decoder) % 2 and i+1 == len(self.decoder)):
				tmp_output = tmp_p
			else:
				tmp_output = self.decoder[i+1](tmp_p)

			p.append(tmp_p)
			output.append(tmp_output)
			mlp_input = tmp_output

		return enc_output, p, output

	def forward_with_nablas(self, x):
		with torch.enable_grad():
			x = x.requires_grad_(True)
			mlp_input, p, output = self.forward(x)
			output_last = output[-1]

			device = torch.device('cuda:0')
			#L1 = 2.0 * torch.sum(output_last)
			L1 = torch.sum(output_last)
			dL1doutput = (autograd.grad(
				L1,
				output_last,
				retain_graph=True
			)[0]).requires_grad_(True)

			nablas = autograd.grad( # dL1dinput aka dL1dmlp_input
				output_last,
				x, # mlp_input
				dL1doutput,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0].requires_grad_(True)

			dL1dinput = autograd.grad( # dL1dinput aka dL1dmlp_input
				output_last,
				x, # mlp_input
				dL1doutput,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0].requires_grad_(True)
			# compute dL1denc_input
			dL1denc_input = dL1dinput

			# compute dL1doutput in backward order
			dL1doutput_list = []
			for i in range(0, len(self.decoder), 2):
				idx = int(i / 2)
				if idx == int((len(self.decoder)+1)/2 - 1): #if idx == int(len(self.decoder)/2 - 1):
					input = mlp_input
				else:
					input = output[-(idx+2)]
				output_current = output[-(idx+1)]

				dL1doutput_list.append(dL1doutput)

				dL1dinput = (autograd.grad(
					output_current,
					input,
					dL1doutput,
					retain_graph=True,
					create_graph=True,
				)[0]).requires_grad_(True)

				dL1doutput = dL1dinput
			# append dL1doutput for encoder output
			dL1denc_output = dL1doutput

			dL1denc_input = autograd.grad( # dL1dinput aka dL1dmlp_input
				mlp_input,
				x, #mlp_input, # x
				dL1denc_output,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0].requires_grad_(True)			

			# second order
			dL1_dinput = dL1denc_input
			#L2 = torch.sum(nablas)
			L2 = torch.sum(torch.abs(torch.norm(nablas, p=2, dim=1) - 1))
			
			dL2_ddL1dinput = autograd.grad(
				L2,
				nablas, # x: the original input
				create_graph=True,
				retain_graph=True,
			)[0].requires_grad_(True)

			# encoder second order
			dL2_ddL1denc_input = dL2_ddL1dinput
			
			w = self.encoder.params
			dL2_denc_w = autograd.grad(
				L2,
				w,
				create_graph=True, # False
				retain_graph=True
			)[0]
			
			dL2_ddL1denc_output = autograd.grad(
				dL1_dinput,
				dL1denc_output,
				dL2_ddL1dinput,
				create_graph=True,
				retain_graph=True
			)[0].requires_grad_(True)

			dL2_denc_input = autograd.grad(
				L2,
				x,
				create_graph=True,
				retain_graph=True
			)[0].requires_grad_(True)

			# mlp second order
			dL1_dinput = dL1denc_output
			dL2_ddL1dinput = dL2_ddL1denc_output
			
			dL2_ddL1doutput_list = []
			dL2_dinput_list = []
			dL2_dw_list = []
			for i in range(0, len(self.decoder), 2):
				idx = int(i / 2)
				if idx == 0:
					input = mlp_input
					dL1_dinput = dL1denc_output
				else:
					input = output[idx-1]
					dL1dinput = dL1doutput
				dL1doutput = dL1doutput_list[-(idx+1)]

				w = self.decoder[i].weight
				
				dL2_dw = autograd.grad(
					L2,
					w,
					create_graph=False,
					retain_graph=True
				)[0]

				#print("dL1dinput.shape: ", dL1dinput.shape)
				#print("dL1doutput.shape: ", dL1doutput.shape)
				dL2_ddL1doutput = autograd.grad(
					dL1dinput,
					dL1doutput,
					dL2_ddL1dinput,
					create_graph=True,
					retain_graph=True
				)[0].requires_grad_(True)
				dL2_ddL1dinput = dL2_ddL1doutput

				if (len(self.decoder) % 2 and idx == int((len(self.decoder)+1)/2 - 1)):
					dL2_dinput = torch.zeros_like(input).to(device)
				else:
					dL2_dinput = autograd.grad(
						L2,
						input,
						create_graph=True,
						retain_graph=True
					)[0].requires_grad_(True)

				dL2_ddL1doutput_list.append(dL2_ddL1doutput)
				dL2_dinput_list.append(dL2_dinput)
				dL2_dw_list.append(dL2_dw)

		# writing outputs to file
		file_dir = os.getcwd()
		file_path = os.path.join(file_dir, "python_output.txt")
		f = open(file_path, "w")
		print("&&&&&&& python output &&&&&&&", file=f)
		print("x = ", x, file=f)
		print("mlp_input = ", mlp_input, file=f)
		print("p = ", p, file=f)
		print("output = ", output, file=f)
		print("nablas = ", nablas, file=f)
		print("dL2_ddL1denc_input = ", dL2_ddL1denc_input, file=f)
		print("dL2_ddL1denc_output = ", dL2_ddL1denc_output, file=f)
		#print("dL2_denc_input = ", dL2_denc_input, file=f)
		#print("dL2_denc_w = ", dL2_denc_w, file=f)
		#print("dL1denc_output = ", dL1denc_output, file=f)
		#print("dL1doutput = ", dL1doutput_list, file=f)
		#print("dL1denc_input = ", dL1denc_input, file=f)
		print("dL2_ddL1doutput_list: ", dL2_ddL1doutput_list, file=f)
		print("dL2dw = ", dL2_dw_list[-1], file=f)
		for i in range(dL2_dw_list[-1].shape[0]):
			print("dL2dw[",i,"]:", dL2_dw_list[-1][i], file=f)
		print("end of dL2dw", file=f)
		print("dL2_dinput_list = ", dL2_dinput_list, file=f)
		print("dL2_ddL1doutput = ", dL2_ddL1doutput, file=f)
		# mlp_input, p, output
		print("&&&&&&& &&&&&& &&&&&& &&&&&&&", file=f)

		return mlp_input, p, output, nablas, dL1doutput_list, dL2_ddL1doutput_list, dL2_dinput_list, dL2_dw_list, dL1denc_input, dL2_ddL1denc_output, dL2_denc_w

if __name__ == '__main__':
	"""
	NOTE: Jianfei:  I provide three testing tools for backward_backward functionality.
					Play around as you want :)
					1. test_train(): train a toy SDF model with eikonal term.
					2. grad_check(): check backward_backward numerical correctness via torch.autograd.gradcheck.
					3. vis_graph(): visualize torch compute graph
	"""

	def test_grad_grad_mlp_():
		## ================ params declaration ================
		device = torch.device("cuda")
		model = SDF_multi(True, n_levels=1, log2_hashmap_size=15, base_resolution=4, smoothstep=False).to(device)

		print("======= within python verification ======= ")
		print("model.encoder.params.shape(): ", len(model.encoder.params))
		print("model.encoder.params: ", model.encoder.params[0:8])

		#x = (torch.tensor([[0.0679, 0.1012, 0.1586]], dtype=torch.float, device=device)).requires_grad_(True)
		#x = torch.tensor([[-0.4805,  0.2734,  0.3652], [ 0.3096,  0.1665, -0.1348]], device='cuda', dtype=torch.float16).requires_grad_(True)
		x = torch.tensor([[-0.4805,  0.2734,  0.3652]], device='cuda', dtype=torch.float16).requires_grad_(True)
		#x = (torch.rand((1, 3), dtype=torch.float, device=device)).requires_grad_(True)

		mlp_input, p_list, output_list, nablas, dL1doutput_n_torch_list, \
		    dL2_ddL1doutput_list, dL2_dinput_list, dL2_dw_list, \
			dL1denc_input, dL2_ddL1denc_output, dL2_denc_w = model.forward_with_nablas(x)

		## ================== file output =====================
		file_dir = os.getcwd()
		# save model params
		import numpy as np
		enc_params = model.encoder.params.clone().detach().cpu().numpy()
		
		np.save(os.path.join(file_dir, 'params_encoder.npy'), np.float32(enc_params))
		enc_params = np.load(os.path.join(file_dir, 'params_encoder.npy'))
		print("enc_params: ", enc_params)

		file_path = os.path.join(file_dir, "output.txt")
		f = open(file_path, "w")


		## ================ first order backward: dL1dinput comparison ================
		output = output_list[-1]

		# L1 loss
		L1 = torch.sum(output) # torch.sum(torch.pow(output - GT_64, 2.0)) # 2.0 * torch.sum(output)
		dL1doutput = autograd.grad(L1, output, create_graph=True, retain_graph=True)[0].requires_grad_(True)
		K_ACT = 10.0

		dL1doutput_list = []
		dL1dp_list = []

		for i in range(0, len(model.decoder), 2):
			idx = int(i / 2)
			if idx == int((len(model.decoder)+1)/2 - 1):
				input = mlp_input
			else:
				input = output_list[-(idx+2)]
			p = p_list[-(idx+1)]
			output = output_list[-(idx+1)]
			#w = model.decoder[-(i+2)].weight # last layer with activation
			w = model.decoder[-(i+1)].weight # last layer without activation
			dL1doutput_list.append(dL1doutput)

			#print("==================== input ====================", file=f)
			#print("idx: ", idx, "input.shape: ", input.shape, "p.shape: ", p.shape, "output.shape: ", output.shape, file=f)

			# if it's the last layer, p equals to output and there is no activation
			# therefore we don't have to compute dL1dp
			if idx == 0:
				dL1dp = dL1doutput
				dL1dp_torch = (autograd.grad(L1, p, retain_graph=True)[0]).requires_grad_(True)
				dL1dp_list.append(dL1dp)
			else:
				# compute doutputdp
				p_m = p.shape[1]
				doutputdp = torch.zeros([p_m, p_m], dtype=torch.float32, device=device)
				for j in range(p_m):
					# 1 – 1/(e^(p * K_ACT) + 1.0)
					doutputdp_j = 1.0 - 1.0/(torch.exp(p[0, j] * K_ACT) + 1.0)
					doutputdp[j, j] = doutputdp_j

				dL1dp_torch = (autograd.grad(L1, p, retain_graph=True)[0]).requires_grad_(True)
				dL1dp = torch.matmul(dL1doutput, doutputdp)
				# save in reverse order
				dL1dp_list.append(dL1dp)

			#print("==================== first order derivative ====================")
			#print("==================== first order derivative ====================", file=f)
			er = torch.allclose(dL1dp, dL1dp_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL1dp and dL1dp_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL1dp and dL1dp_torch in atol=1e-07 rtol=1e-07: ", er, file=f)

			# compare the difference between dL1doutput and dL1doutput_torch
			er = torch.allclose(dL1doutput, dL1doutput_n_torch_list[idx], rtol=1e-07, atol=1e-07)
			#print("allclose of dL1doutput and dL1doutput_n_torch_list in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL1doutput and dL1doutput_n_torch_list in atol=1e-07 rtol=1e-07: ", er, file=f)

			
			# compute dpdinput = w
			input_m = input.shape[1]
			p_m = p.shape[1]
			dpdinput = torch.ones([p_m, input_m], dtype=torch.float32, device=device)
			for j in range(p_m):
				dpdinput_j = autograd.grad(p[0, j], input, retain_graph=True)[0]
				dpdinput[j] = dpdinput_j
		
			## compute dL1dinput
			dL1dinput = torch.matmul(dL1dp, dpdinput).requires_grad_(True)
			# pytorch dL1dinput
			dL1dinput_torch = autograd.grad(L1, input, retain_graph=True, create_graph=True)[0].requires_grad_(True)

			er = torch.allclose(dL1dinput, dL1dinput_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL1dinput and dL1dinput_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL1dinput and dL1dinput_torch in atol=1e-07 rtol=1e-07: ", er, file=f)

			## compute dL1dw
			dL1dw = torch.matmul(torch.transpose(dL1dp, 0, 1), input)
			dL1dw_torch = autograd.grad(L1, w, retain_graph=True)[0]
			er = torch.allclose(dL1dw, dL1dw_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL1dw and dL1dw_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL1dw and dL1dw_torch in atol=1e-07 rtol=1e-07: ", er, file=f)
			#print("==================== ====================== ====================")
			#print("==================== ====================== ====================", file=f)
			

			dL1doutput = dL1dinput

		## ================ second order backward: dL2d_dL1dp comparison ================
		# L2 loss
		L2 = torch.sum(nablas) #torch.sum(nablas * nablas) #torch.sum(nablas)
		print("==================== Loss 2 ====================", file=f)
		print("L2: ", L2, file=f)
		print("==================== ====== ====================", file=f)

		# MLP 2nd order derivative
		dL2d_dL1dinput = autograd.grad(
			L2,
			nablas,
			retain_graph=True,
			create_graph=False
		)[0]
		dL2d_dL1dinput = dL2_ddL1denc_output

		dL2dinput_self_list = []
		dL2dw_self_list = []
		dL2dw_torch_list = []
		'''
		print("================= dL1dp_list ================", file=f)
		print("dL1dp_list: ", dL1dp_list, file=f)
		print("+++++++++++++++++++++++++++++++++++++++++++++", file=f)
		'''
		for i in range(0, len(model.decoder), 2):
			idx = int(i / 2)
			if i == 0:
				input = mlp_input
			else:
				input = output_list[idx-1] # after activation
			p = p_list[idx]
			output = output_list[idx]
			w = model.decoder[i].weight

			dL1dp = dL1dp_list[-(idx+1)]
			dL1doutput = dL1doutput_n_torch_list[-(idx+1)]
			dL1doutput_torch = dL1doutput_n_torch_list[-(idx+1)]
			input_m = input.shape[1]
			p_m = p.shape[1]
			
			## compute dL2dw
			# the 1st term: dL1doutput x diag[e^(p*K_ACT) / (e^p*K_ACT + 1.0)^2] * K_ACT * p
			d_dL1dinput_dw = torch.transpose(dL1dp, 0, 1)
			dL2dw_1 = torch.matmul(d_dL1dinput_dw, dL2d_dL1dinput)

			# the 2nd term: dL1doutput x d_doutputdp_dw x dpdinput
			# compute d_doutputdp_dw

			# =========== new modification ============ #
			# dL2_ddL1dinput_2 x w_2 shape: [1, 64]
			wT = w.transpose(0, 1)
			dL2_ddL1dinput_x_w = torch.matmul(dL2d_dL1dinput, wT)
			dL1doutput_x_dL2_ddL1dinput_x_w = torch.mul(dL1doutput, dL2_ddL1dinput_x_w)

			doutputdp_2 = torch.zeros([p_m, p_m], dtype=torch.float32, device=device)
			for i in range(doutputdp_2.shape[0]):
				tmp = torch.exp(p[0, i] * K_ACT)/torch.pow(torch.exp(p[0, i] * K_ACT)+1.0, 2.0) * K_ACT
				doutputdp_2[i, i] = tmp

			ddoutputdp_dp_2 = torch.matmul(dL1doutput_x_dL2_ddL1dinput_x_w, doutputdp_2)
			dL2dw_2 = torch.matmul(torch.transpose(ddoutputdp_dp_2, 0, 1), input)

			if idx == int(len(model.decoder)/2):
				dL2dw_2 = torch.zeros_like(dL2dw_2)

			# dL2dw add up		
			dL2dw = dL2dw_1 + dL2dw_2
			dL2dw_self_list.append(dL2dw)

			dL2dw_torch = autograd.grad(
				L2,
				w,
				retain_graph=True,
				create_graph=False)[0]
			dL2dw_torch_list.append(dL2dw_torch)

			er = torch.allclose(dL2dw, dL2dw_torch, rtol=1e-07, atol=1e-07)
			
			'''
			print("++++++++++ validation of dL2dw ++++++++++", file=f)
			#print("p: ", p, file=f)
			#print("d_dL1dinput_dw: ", d_dL1dinput_dw, file=f)
			#print("dL1doutput: ", dL1doutput, file=f)
			print("dL2dw_1: ", dL2dw_1, file=f)
			#print("ddoutputdp_dp_2: ", ddoutputdp_dp_2, file=f)
			#print("dL2_ddL1dinput_x_w: ", dL2_ddL1dinput_x_w, file=f)
			#print("dL1doutput_x_dL2_ddL1dinput_x_w: ", dL1doutput_x_dL2_ddL1dinput_x_w, file=f)
			#print("doutputdp_2: ", doutputdp_2, file=f)
			
			#print("input: ", input, file=f)
			#print("output: ", output, file=f)
			#print("ddoutputdp_dp_2: ", ddoutputdp_dp_2, file=f)
			print("dL2dw_2: ", dL2dw_2, file=f)
			print("dL2dw: ", file=f)
			for ii in range(dL2dw.shape[0]):
				print(dL2dw[ii], file = f)
					

			print("dL2dw_torch: ", dL2dw_torch, file=f)
			print("++++++++++ =================== ++++++++++", file=f)
			'''
			
			## compute dL2dinput
			dL2dinput_torch = dL2_dinput_list[idx] #autograd.grad(L2, input, retain_graph=True, create_graph=False)[0]

			d_doutputdp_dinput = torch.zeros([p_m, input_m], dtype=torch.float32, device=device)
			#print("================== validation of dL2dinput_1 doutputdp ===================", file=f)
			#print("p: ", p, file=f)
			for j in range(p_m):
				#print("within compute_dL2dinput - doutputdp_2 = ", torch.exp(p[0, j] * K_ACT)/torch.pow(torch.exp(p[0, j] * K_ACT)+1.0, 2.0) * K_ACT)
				tmp = torch.exp(p[0, j] * K_ACT)/torch.pow(torch.exp(p[0, j] * K_ACT)+1.0, 2.0) * K_ACT * dL1doutput[0, j]
				#print("doutputdp_2[",j,"] = ", torch.exp(p[0, j] * K_ACT)/torch.pow(torch.exp(p[0, j] * K_ACT)+1.0, 2.0) * K_ACT * dL1doutput[0, j], file=f)
				for k in range(input_m):
					d_doutputdp_dinput[j, k] = tmp * w[j, k] #torch.sum(w[i, :]) #x[0, j]
			#print("===========================================================================", file=f)
			dL2dinput_1 = torch.matmul(dL2d_dL1dinput, torch.matmul(torch.transpose(d_doutputdp_dinput, 0, 1), w))

			if idx == int(len(model.decoder)/2):
				dL2dinput_1 = torch.zeros_like(dL2dinput_1)

			
			print("======== validation of dL2dinput_1: 2nd order term ========", file=f)
			print("p: ", p, file=f)
			print("dL1doutput: ", dL1doutput, file=f)
			print("d_doutputdp_dinput: ", d_doutputdp_dinput, file=f)
			print("dL2dinput_1: ", dL2dinput_1, file=f)
			print("dL2dinput_torch: ", dL2dinput_torch, file=f)
			#print("dL2d_dL1dinput: ", dL2d_dL1dinput, file=f)
			#print("torch.matmul(torch.transpose(d_doutputdp_dinput, 0, 1), w): ", torch.matmul(torch.transpose(d_doutputdp_dinput, 0, 1), w), file=f)
			print("======== ================================ ========", file=f)
			

			# einsum for dL2_dinput
			ddoutputdp_dp = torch.zeros([p_m, p_m, p_m], dtype=torch.float32, device=device)
			for j in range(p_m):
				ddoutputdp_dp[j, j, j] = torch.exp(p[0, j] * K_ACT)/torch.pow(torch.exp(p[0, j] * K_ACT)+1.0, 2.0) * K_ACT
			# ddoutputdp_dinput = ddoutputdp_dp x w
			ddoutputdp_dinput = torch.zeros([p_m, p_m, input_m], dtype=torch.float32, device=device)
			ddoutputdp_dinput = torch.einsum("ijk, kh->ijh", ddoutputdp_dp, w)
			ddL1doutput_dinput = torch.einsum("bi, ijh->jh", dL1doutput, ddoutputdp_dinput)
			wT = w.transpose(0, 1)
			ddL1dinput_dinput = torch.matmul(wT, ddL1doutput_dinput)
			dL2dinput = torch.matmul(dL2d_dL1dinput, ddL1dinput_dinput)

			if idx == int(len(model.decoder)/2): # last layer without activation
				dL2dinput = torch.zeros_like(dL2dinput)

			dL2dinput_self_list.append(dL2dinput)

			er = torch.allclose(dL2dinput_1, dL2dinput_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL2dinput and dL2dinput_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL2dinput and dL2dinput_torch in atol=1e-07 rtol=1e-07: ", er, file=f)
			
			print("======== validation of dL2dinput 2nd order term ========", file=f)
			print("dL2dinput_1: ", dL2dinput_1, file=f)
			print("dL2dinput_torch: ", dL2dinput_torch, file=f)
			print("======== dL2dinput 2nd order term ends ========", file=f)
			
			
			# compute doutputdp
			p_m = p.shape[1]
			doutputdp = torch.zeros([p_m, p_m], dtype=torch.float32, device=device)
			for j in range(p_m):
				# 1 – 1/(e^(p * K_ACT) + 1.0)
				doutputdp_j = 1.0 - 1.0/(torch.exp(p[0, j] * K_ACT) + 1.0)
				doutputdp[j, j] = doutputdp_j

			## compute dL2d_dL1doutput
			dL2d_dL1dp = torch.matmul(dL2d_dL1dinput, torch.transpose(w, 0, 1))
			dL2d_dL1doutput = torch.matmul(dL2d_dL1dp, doutputdp)

			if idx == int(len(model.decoder)/2): # last layer without activation
				dL2d_dL1doutput = dL2d_dL1dp
			

			# pytorch dL2d_dL1doutput
			dL2d_dL1doutput_torch = dL2_ddL1doutput_list[idx]
			er = torch.allclose(dL2d_dL1doutput, dL2d_dL1doutput_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL2d_dL1doutput and dL2d_dL1doutput_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL2d_dL1doutput and dL2d_dL1doutput_torch in atol=1e-07 rtol=1e-07: ", er, file=f)

			# for iteration
			dL2d_dL1dinput = dL2d_dL1doutput_torch # dL2d_dL1doutput

			
			print("======== validation of dL2d_dL1doutput ========", file=f)
			#print("dL2d_dL1dp i.e. 1st step: ", dL2d_dL1dp, file=f)
			#print("doutputdp: ", doutputdp, file=f)
			print("dL2d_dL1doutput: ", dL2d_dL1doutput, file=f)
			print("dL2d_dL1doutput_torch: ", dL2d_dL1doutput_torch, file=f)
			print("======== ================================ ========", file=f)
			

		dL2dinput_2_list = []
		dL2dinput_2_list.append(dL2dinput_self_list[-1])
		print("dL2_dinput_list: ", dL2_dinput_list, file=f)
		#print("=============== add up L1 backward within L2 loss ===============", file=f)
		for i in range(0, len(model.decoder)-2, 2):
			idx = int(i / 2)
			print("within 1st order in 2nd order derivative:")
			print("idx: ", idx)
			print("len(output_list): ", len(output_list))
			print("-(idx+2+1): ", -(idx+2+1))
			print("=================================================================")
			print("i: ", i, "len(model.decoder) - 2 - 2: ", len(model.decoder) - 2 - 2)
			if i == len(model.decoder) - 1 - 2:
				input = mlp_input
			else:
				input = output_list[-(idx+2+1)]

			dL2doutput = dL2dinput_self_list[-(idx+1)] # the last dL2dinput
			
			p = p_list[-(idx+1+1)]
			w = model.decoder[-(i+1+2)].weight
			p_m = p.shape[1]

			doutputdp = torch.zeros(p_m, p_m)
			doutputdp = torch.zeros([p_m, p_m], dtype=torch.float32, device=device)
			for j in range(p_m):
				# 1 – 1/(e^(p * K_ACT) + 1.0)
				doutputdp_j = 1.0 - 1.0/(torch.exp(p[0, j] * K_ACT) + 1.0)
				doutputdp[j, j] = doutputdp_j

			dL2dp = torch.matmul(dL2doutput, doutputdp)
			dL2dinput_2 = torch.matmul(dL2doutput, torch.matmul(doutputdp, w))
			dL2dinput_2_list.append(dL2dinput_2)			
			dL2dinput_1 = dL2dinput_self_list[-(idx+2)]

			dL2dinput = dL2dinput_1 + dL2dinput_2
			dL2dinput_self_list[-(idx+2)] = dL2dinput

			dL2dinput_torch = dL2_dinput_list[-(idx+2)]

			er = torch.allclose(dL2dinput, dL2dinput_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL2dinput and dL2dinput_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL2dinput and dL2dinput_torch in atol=1e-07 rtol=1e-07: ", er, file=f)	

			dL2dp = torch.matmul(dL2doutput, doutputdp)
			dL2dw_2 = torch.matmul(dL2dp.transpose(0, 1), input)
			dL2dw_1 = dL2dw_self_list[-(idx+2)]

			#print("dL2dw_1: ", dL2dw_1, file=f)
			#print("dL2dw_2: ", dL2dw_2, file=f)
			dL2dw = dL2dw_1 + dL2dw_2

			dL2dw_torch = dL2_dw_list[-(idx+2)] #dL2dw_torch_list[-(idx+2)]
			#print("dL2dw: ", dL2dw)
			#print("dL2dw_torch: ", dL2dw_torch)
			#print("dL2dw: ", dL2dw, file=f)
			#print("dL2dw_torch: ", dL2dw_torch, file=f)
			er = torch.allclose(dL2dw, dL2dw_torch, rtol=1e-07, atol=1e-07)
			#print("allclose of dL2dw and dL2dw_torch in atol=1e-07 rtol=1e-07: ", er)
			#print("allclose of dL2dw and dL2dw_torch in atol=1e-07 rtol=1e-07: ", er, file=f)

			
			print("======== validation of dL2dinput_2 ========", file=f)
			print("dL2dinput_1: ", dL2dinput_1, file=f)
			#print("doutputdp: ", doutputdp, file=f)
			print("dL2doutput: ", dL2doutput, file=f)
			#print("dL2dp: ", dL2dp, file=f)
			print("dL2dinput_2: ", dL2dinput_2, file=f)
			print("dL2dinput: ", dL2dinput, file=f)
			print("dL2dinput_torch: ", dL2dinput_torch, file=f)
			print("======== ================================ ========", file=f)
			
			print("======== validation of dL2dw ========", file=f)
			print("dL2dp: ", dL2dp, file=f)
			print("input: ", input, file=f)
			print("dL2dw_2: ", dL2dw_2, file=f)
			print("dL2dw: ", dL2dw, file=f)
			print("dL2dw_torch: ", file=f)
			for ii in range(dL2dw_torch.shape[0]):
				print(dL2dw_torch[ii], file=f)
			print("======== ================================ ========", file=f)
			

		output = output_list[-1]
		

if __name__ == "__main__":
	# test encoding + multi-layer perceptron
	test_grad_grad_mlp_()


