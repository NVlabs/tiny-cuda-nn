#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import autograd
from torch.optim import Adam, SGD
import torch.nn.functional as F

import sys
import os
import numpy as np
import time
NoLog = False

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
		'''
		self.encoder = tcnn.Encoding(3, {
			"otype": "HashGrid" if hash else "DenseGrid",
			"n_levels": n_levels,
			"n_features_per_level": 8,
			"log2_hashmap_size": log2_hashmap_size,
			"base_resolution": base_resolution,
			"per_level_scale": 1.5,
			"interpolation": "Smoothstep" if smoothstep else "Linear"
		})
		'''
		self.encoder = tcnn.Encoding(3, {
			"otype": "HashGrid" if hash else "DenseGrid",
			"n_levels": 16, #n_levels,
			"n_features_per_level": 2, #8,
			"log2_hashmap_size": 19, #log2_hashmap_size,
			"base_resolution": 16, #base_resolution,
			"per_level_scale": np.exp2(np.log2(2048 * 1 * 1 / 16) / (16 - 1)), #1.5,
			#"interpolation": "Smoothstep" if smoothstep else "Linear"
		})
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
		)

		# write into numpy
		#params_enc = self.encoder.params.data.clone().cpu().numpy()
		#np.save('numpy/params_sdf_enc.npy', params_enc)
		#print("params_enc[0:512]: ", params_enc[0:512])

		params_enc = np.load('numpy/params_sdf_enc.npy')
		#print("after loading from numpy: params[0:512]: ", params_enc[0:512])
		#print("params_enc.shape: ", params_enc.shape)
		self.encoder.params.data = torch.from_numpy(params_enc)

		idx = 0
		for m in self.decoder.modules():
			if isinstance(m, nn.Linear):
				if idx == 0:
					#params_input = m.weight.data.clone().cpu().numpy()
					#np.save("numpy/params_input.npy", params_input)
					#print("params_input.shape: ", params_input.shape)

					params_input = np.load("numpy/params_input.npy")
					m.weight.data = torch.from_numpy(params_input)
				elif idx == 1:
					#params_hidden_1 = m.weight.data.clone().cpu().numpy()
					#np.save("numpy/params_hidden_1.npy", params_hidden_1)
					#print("params_hidden_1.shape: ", params_hidden_1.shape)

					params_hidden_1 = np.load("numpy/params_hidden_1.npy")
					m.weight.data = torch.from_numpy(params_hidden_1)
				elif idx == 2:
					#params_hidden_2 = m.weight.data.clone().cpu().numpy()
					#np.save("numpy/params_hidden_2.npy", params_hidden_2)
					#print("params_hidden_2.shape: ", params_hidden_2.shape)

					params_hidden_2 = np.load("numpy/params_hidden_2.npy")
					m.weight.data = torch.from_numpy(params_hidden_2)

				else:
					#params_output = m.weight.data.clone().cpu().numpy()
					#np.save("numpy/params_output.npy", params_output)
					#print("params_output.shape: ", params_output.shape)

					params_output = np.load("numpy/params_output.npy")
					m.weight.data = torch.from_numpy(params_output)
				idx += 1

	def set_cuda_fun(func):
		num = 0  # 初始化次数
		total_time = 0
		
		def call_fun(*args, **kwargs):
			nonlocal num 
			nonlocal total_time
			if NoLog:
				res = func(*args, **kwargs) 
				return res
			
			torch.cuda.synchronize()
			start = time.perf_counter()  # 代码执行开始时间
			num += 1 # 每次调用次数加1
			res = func(*args, **kwargs) 
			torch.cuda.synchronize()
			end = time.perf_counter() # 代码执行结束时间
            # longtime = end - start
			total_time += end - start
			print("SDF in Pytorch 前向耗时： ", func.__name__, " 调用次数： ", num, " 累计时间：", total_time)
			return res
		return call_fun
	
	def forward(self, x):
		encoded = self.encoder(x).to(dtype=torch.float)
		sdf = self.decoder(encoded)
		return sdf

	#@set_cuda_fun
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
	
class SDF_MLP(nn.Module):
	def __init__(self, hash=True, n_levels=12, log2_hashmap_size=15, base_resolution=16, smoothstep=False) -> None:
		super().__init__()
		
		b_flag = False
		self.decoder = nn.Sequential(
			nn.Linear(8, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
		)

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
				idx += 1
				#print("m.weight.data: ", m.weight.data)

	def forward(self, x):
		#encoded = self.encoder(x).to(dtype=torch.float)
		sdf = self.decoder(x)
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

class SDF_TCNN(nn.Module):
	def __init__(self, hash=True, n_levels=12, log2_hashmap_size=15, base_resolution=16, smoothstep=False) -> None:
		super().__init__()

		self.encoder_decoder = tcnn.NetworkWithInputEncoding(
			encoding_config = {
				"otype": "HashGrid" if hash else "DenseGrid",
				"n_levels": 16, #n_levels,
				"n_features_per_level": 2, #8,
				"log2_hashmap_size": 19, #log2_hashmap_size,
				"base_resolution": 16, #base_resolution,
				"per_level_scale": np.exp2(np.log2(2048 * 1 * 1 / 16) / (16 - 1)), #1.5,
				#"interpolation": "Smoothstep" if smoothstep else "Linear"
			},
			n_input_dims=3, #self.encoder.n_output_dims, #3
			n_output_dims=1, #64,
            network_config={
                "otype": "CutlassMLP", #"FullyFusedMLP",
                "activation": "Softplus",
                "output_activation": "None", #"Softplus",
                "n_neurons": 64,
                "n_hidden_layers": 3 #7
            },
			#dtype=torch.float32,
		)

		
		# init encoder params from file
		file_dir = os.getcwd()
		
		enc_params = np.load(os.path.join(file_dir, 'numpy/params_sdf_enc.npy'))
		#print("enc_params: ", enc_params)
		enc_p_torch = torch.from_numpy(enc_params)

		num_params = len(self.encoder_decoder.params)
		num_enc_params = 12196240
		start_params = num_params - num_enc_params
		# for i in range(num_enc_params):
		# 	tmp = enc_p_torch[i].detach()
		# 	self.encoder_decoder.params.data.index_fill_(0, torch.tensor(i+start_params, dtype=torch.int64).cuda(), torch.tensor(tmp).cuda())
		#print("self.encoder_decoder.params.data of encoder: ", self.encoder_decoder.params[-512:])
		

		# init decoder params from file
		params_input = np.load(os.path.join(file_dir, 'numpy/params_input.npy'))
		params_input_tensor = torch.from_numpy(params_input)
		print("params_input.shape: ", params_input.shape)
		print("total params: ", self.encoder_decoder.params.shape)

		## init input layer, notice NOT to fill unneccessary blanks (0.0)
		idx_tcnn = 0
		for i in range(params_input_tensor.shape[0]):
			for j in range(params_input_tensor.shape[1]):
				self.encoder_decoder.params[j+idx_tcnn].data.copy_(params_input_tensor[i, j])
				#print("params_input[",j+idx_tcnn,"] = ", self.encoder_decoder.params[j+idx_tcnn])
			idx_tcnn += 32 # column num in TCNN
		
		## init hidden layers
		params = np.load(os.path.join(file_dir, 'numpy/params_hidden_1.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 2048 # skip the input layer params
		print("params_hidden_1.shape: ", params_tensor.shape)
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]

		params = np.load(os.path.join(file_dir, 'numpy/params_hidden_2.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 6144 # skip the input layer params
		print("params_hidden_2.shape: ", params_tensor.shape)
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]

		params = np.load(os.path.join(file_dir, 'numpy/params_output.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 10240 # skip the input layer params
		print("params_output.shape: ", params_tensor.shape)
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]
		#print("params from numpy: ", params_tensor[-1, -1])
		#print("self.encoder_decoder.params.data of output layer: ", self.encoder_decoder.params[-576:-512])
		
	def set_cuda_fun(func):
		num = 0  # 初始化次数
		total_time = 0
		
		def call_fun(*args, **kwargs):
			nonlocal num 
			nonlocal total_time
			if NoLog:
				res = func(*args, **kwargs) 
				return res
			
			torch.cuda.synchronize()
			start = time.perf_counter()  # 代码执行开始时间
			num += 1 # 每次调用次数加1
			res = func(*args, **kwargs) 
			torch.cuda.synchronize()
			end = time.perf_counter() # 代码执行结束时间
            # longtime = end - start
			total_time += end - start
			print("SDF in TCNN 前向耗时： ", func.__name__, " 调用次数： ", num, " 累计时间：", total_time)
			return res
		return call_fun

	
	def forward(self, x):
		sdf = self.encoder_decoder(x)
		return sdf

	#@set_cuda_fun
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
			#print("sdf.shape: ", sdf.shape, "x.shape: ", x.shape, "nablas.shape: ", nablas.shape)

		return sdf, nablas

class SDF_MLP_TCNN(nn.Module):
	def __init__(self, hash=True, n_levels=12, log2_hashmap_size=15, base_resolution=16, smoothstep=False) -> None:
		super().__init__()
		
		self.decoder = tcnn.Network(
			n_input_dims=8,
            n_output_dims=64,
            network_config={
                "otype": "CutlassMLP",
                "activation": "Softplus",
                "output_activation": "Softplus",
                "n_neurons": 64,
                "n_hidden_layers": 3
            }
		)
		
		# init decoder params from file
		file_dir = os.getcwd()
		params_input = np.load(os.path.join(file_dir, 'params_input.npy'))
		params_input_tensor = torch.from_numpy(params_input)

		## init input layer, notice NOT to fill unneccessary blanks (0.0)
		idx_tcnn = 0
		for i in range(params_input_tensor.shape[1]):
			for j in range(params_input_tensor.shape[2]):
				self.decoder.params[j+idx_tcnn].data.copy_(params_input_tensor[0, i, j])
				#print("params_input[",j+idx_tcnn,"] = ", self.encoder_decoder.params[j+idx_tcnn])
			idx_tcnn += 16 # column num in TCNN
		
		## init hidden + output layers
		params = np.load(os.path.join(file_dir, 'params.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 1024 # skip the input layer params
		for i in range(params_tensor.shape[0]):
			for j in range(params_tensor.shape[1]):
				for k in range(params_tensor.shape[2]):
					self.decoder.params[k+idx_tcnn].data.copy_(params_tensor[i, j, k])
				idx_tcnn += params_tensor.shape[2]
		
		#for i in range(1024, 2048, 64):
		#	print("params_input[",i,",",i+63,"]: ", self.decoder.params[i:i+64])
		
	def forward(self, x):
		#sdf = self.encoder_decoder(x)
		sdf = self.decoder(x)

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
			#print("sdf.shape: ", sdf.shape, "x.shape: ", x.shape, "nablas.shape: ", nablas.shape)

		return sdf, nablas



if __name__ == '__main__':
	"""
	NOTE: Jianfei:  I provide three testing tools for backward_backward functionality.
					Play around as you want :)
					1. test_train(): train a toy SDF model with eikonal term.
					2. grad_check(): check backward_backward numerical correctness via torch.autograd.gradcheck.
					3. vis_graph(): visualize torch compute graph
	"""

	def print_torch_SDF(model):
		# for pytorch SDF
		print("encoder grad: ", model.encoder.params.grad)
		print("========= ========= ========= ========= ========= ========")
		print("decoder grad: ")
		for i in range(0, len(model.decoder), 2):
			if i % 2 == 0:
				#print("decoder w grad ", i, "-th layer:", model.decoder[i].weight.grad)
				#print("decoder b grad ", i, "-th layer:", model.decoder[i].bias.grad)
				w_grad = model.decoder[i].weight.grad
				print("w_grad.shape: ", w_grad.shape)
				r, c = w_grad.shape
				if c >= 64:
					r = 3
				else:
					r = 8
				for i in range(0, r):
					print("grad[",i*c,", ",(i+1)*c,"]: ", w_grad[i])
				print("========= ========= ========= ========= ========= ========")
		
		return

	def print_torch_SDF_MLP(model):
		# for pytorch SDF_MLP
		print("decoder grad: ")
		for i in range(0, len(model.decoder), 2):
			if i % 2 == 0:
				print("decoder w ", i, "-th layer:")
				w = model.decoder[i].weight.data
				for d in range(2):
					print("w[",d,"]:", w[d])
				print("========= ========= ========= ========= ========= ========")

				'''
				#print("decoder w grad ", i, "-th layer:", model.decoder[i].weight.grad)
				print("decoder w grad ", i, "-th layer:")
				w_grad = model.decoder[i].weight.grad
				for d in range(2): #range(w_grad.shape[0]):
					print("w_grad[",d,"]:", w_grad[d])
				print("========= ========= ========= ========= ========= ========")
				'''

		return

	def print_TCNN_layer_weight(prefix, weight, row, col):
		# for printing weight of each linear layer in TCNN
		print("decoder grad of layer - ", prefix, " - [",row,",", col,"]: ")
		idx = 0
		for i in range(row):
			print(prefix,"[", idx, ":", idx+col, "]: ", weight[idx:idx+col])
			idx = idx + col
			
			if idx > 128:
				break
		print("========= ========= ========= ========= ========= ========")

		return

	def print_TCNN_SDF(model):
		# for SDF in TCNN
		# print grad
		print("encoder_decoder grad.shape: ", model.encoder_decoder.params.grad.shape)
		enc_grad = model.encoder_decoder.params.grad[-512:]
		print("encoder grad: ", enc_grad)
		
		dec_grad_0 = model.encoder_decoder.params.grad[0:1024]
		dec_grad_1 = model.encoder_decoder.params.grad[1024:5120] # 1st hidden layer
		dec_grad_2 = model.encoder_decoder.params.grad[5120:9216] # 2nd hidden layer
		dec_grad_3 = model.encoder_decoder.params.grad[9216:13312] # 3rd hidden layer
		
		print_TCNN_layer_weight("dec_grad_0", dec_grad_0, 16, 64)
		print_TCNN_layer_weight("dec_grad_1", dec_grad_1, 64, 64)
		print_TCNN_layer_weight("dec_grad_2", dec_grad_2, 64, 64)
		print_TCNN_layer_weight("dec_grad_3", dec_grad_3, 64, 64)

		# print weight
		weight_0 = model.encoder_decoder.params.data[0:1024]
		weight_1 = model.encoder_decoder.params.data[1024:5120]
		weight_2 = model.encoder_decoder.params.data[5120:9216]
		weight_3 = model.encoder_decoder.params.data[9216:13312]

		#print_TCNN_layer_weight("weight_0", weight_0, 16, 64)
		#print_TCNN_layer_weight("weight_1", weight_1, 64, 64)
		#print_TCNN_layer_weight("weight_2", weight_2, 64, 64)
		#print_TCNN_layer_weight("weight_3", weight_3, 64, 64)

		return

	def print_TCNN_SDF_MLP(model):
		# for SDF MLP in TCNN
		print("decoder grad.shape: ", model.decoder.params.grad.shape)
		
		dec_grad_0 = model.decoder.params.grad[0:1024]
		dec_grad_1 = model.decoder.params.grad[1024:5120] # 1st hidden layer
		dec_grad_2 = model.decoder.params.grad[5120:9216] # 1st hidden layer
		dec_grad_3 = model.decoder.params.grad[9216:13312] # 1st hidden layer
		
		print_TCNN_layer_weight("dec_grad_0", dec_grad_0, 16, 64)
		print_TCNN_layer_weight("dec_grad_1", dec_grad_1, 64, 64)
		print_TCNN_layer_weight("dec_grad_2", dec_grad_2, 64, 64)
		print_TCNN_layer_weight("dec_grad_3", dec_grad_3, 64, 64)

		return
	
	def print_bwdbwd_time(scaler, loss, optimizer, total_time, num):

		torch.cuda.synchronize()
		start = time.perf_counter()  # 代码执行开始时间
		num += 1 # 每次调用次数加1

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
				
		torch.cuda.synchronize()
		end = time.perf_counter() # 代码执行结束时间
		total_time += end - start
		print("SDF in backward_backward耗时: ", " 调用次数： ", num, " 累计时间：", total_time)

		return total_time, num
	
	def compute_normal(x:torch.Tensor, y): #[N,3]
		x.requires_grad_(True)
		y.requires_grad_(True)
		d_output = torch.ones_like(y, requires_grad=False, device=y.device)
		gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
		
		return gradients

	def test_train():
		"""
		train a toy SDF model with eikonal term.
		"""
		from tqdm import tqdm
		device = torch.device("cuda")
		#model = SDF_TCNN(True, n_levels=1, log2_hashmap_size=15, base_resolution=4, smoothstep=False).to(device)
		model = SDF(True, n_levels=1, base_resolution=4).to(device)

		torch.cuda.nvtx.range_push('training_preparation')
		torch.manual_seed(0)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		optimizer = Adam(model.parameters(), 2.0e-4)
		
		iter_i = 0
		fake_nablas = torch.ones([1], device='cuda')
		torch.cuda.nvtx.range_pop()

		from torch.cuda.amp import GradScaler
		from torch.cuda.amp import autocast
		scaler = GradScaler()

		num = 0
		total_time = 0
		#with tqdm(range(10000)) as pbar:
		with tqdm(range(10000)) as pbar:
			for _ in pbar:
				torch.cuda.nvtx.range_push('var_declaration')
				# pytorch input
				#x = torch.rand([102400, 3], dtype=torch.float32, device=device) - 0.5
				# TCNN input
				x = torch.rand([102400, 3], dtype=torch.float16, device=device) - 0.5

				# for detailed number comparison
				#x = (torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float, device=device))#.requires_grad_(True)
				#x = (torch.tensor([[0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.3, 0.4, 0.5]], dtype=torch.float16, device=device)).requires_grad_(True)
				torch.cuda.nvtx.range_pop()

				torch.cuda.nvtx.range_push('python_forward_start')
				sdf, nablas = model.forward_with_nablas(x)
				torch.cuda.nvtx.range_pop()

				torch.cuda.nvtx.range_push('python_nablas_norm')
				nablas_norm: torch.Tensor = nablas.norm(dim=-1)
				torch.cuda.nvtx.range_pop()

				torch.cuda.nvtx.range_push('python_loss_compute')
				optimizer.zero_grad()
				ref_value  = torch.sqrt((x**2).sum(-1)) - 0.5
				#normal_surface = compute_normal(x, sdf)
				
				eikonal_loss = torch.sum(torch.abs(torch.norm(nablas, p=2, dim=1) - 1)) * 0.0001
				#loss = eikonal_loss
				loss = eikonal_loss #F.mse_loss(sdf[..., 0], ref_value) + eikonal_loss

				#loss.backward()
				#optimizer.step()
				
				# print("===== within the ", iter_i, "-th iteration: =====")
				# print("x: ", x)
				# print("sdf: ", sdf)
				# print("nablas: ", nablas)
				# print("normal_surface: ", normal_surface)
				# print("eikonal_loss: ", eikonal_loss)
				#print("loss: ", loss)
				# print("===== ========= ======== ========= ======== =====")

				'''
				# autocast training
				optimizer.zero_grad()
				print("sdf.shape: ", sdf.shape, "ref_value.shape: ", ref_value.shape)
				with autocast():
					#loss = F.mse_loss(nablas_norm, fake_nablas, reduction='mean')
					loss = F.mse_loss(sdf[..., 0], ref_value) * 100

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				'''

				# print time consumed
				# NOTE: if use print_bwdbwd_time, one needs to comment loss.backward() and optimizer.step()
				total_time, num = print_bwdbwd_time(scaler, loss, optimizer, total_time, num)

				torch.cuda.nvtx.range_pop()

				# print grad details for pytorch SDF
				#print_torch_SDF(model)

				# print grad details for TCNN SDF
				#print_TCNN_SDF(model)

				pbar.set_postfix(loss=loss.item())

				if (torch.isnan(loss)):
					break
				
				iter_i = iter_i + 1
				# if iter_i > 2:
				# 	break

		return
	
	def save_model_param(model):
		file_dir = os.getcwd()
		params = []
		params_input = []
		flag_layer = 0
		for m in model.decoder.modules():
			if isinstance(m, nn.Linear):
				if flag_layer == 0:
					params_input.append(m.weight.clone().detach().cpu().numpy())
					print("params of input layer m.weight: ", m.weight.clone().detach().cpu().numpy())
				elif flag_layer > 0:
					params.append(m.weight.clone().detach().cpu().numpy())
					print("params of m.weight: ", m.weight.clone().detach().cpu().numpy())
					# params.append(m.bias.detach().numpy())
				flag_layer = 1
		np.save("test_params_iter_1_input.npy", params_input)
		np.save("test_params_iter_1.npy", params)

		return

if __name__ == "__main__":

	# test tcnn
	test_train()


