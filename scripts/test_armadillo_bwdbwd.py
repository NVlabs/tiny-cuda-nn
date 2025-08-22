#!/usr/bin/env python3

import torch
import trimesh
import torch.nn as nn
from torch import autograd
from torch.optim import Adam, SGD
import torch.nn.functional as F
import skimage
import random
import apex

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

torch.set_printoptions(precision=10)

def GenerateRasterPoints(midpoint, extents, resolution): 
    max_axis = max(extents)
    voxel_size = max_axis / resolution
# 021 120 201 210
    points = np.meshgrid(
        np.linspace(midpoint[0] - extents[0]*.501, midpoint[0] + extents[0]*.501,  int(resolution * extents[0]/max_axis)),
        np.linspace(midpoint[1] - extents[1]*.501, midpoint[1] + extents[1]*.501,  int(resolution * extents[1]/max_axis)),
        np.linspace(midpoint[2] - extents[2]*.501, midpoint[2] + extents[2]*.501,  int(resolution * extents[2]/max_axis))
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    res = np.array([int(resolution * extents[0]/max_axis),int(resolution * extents[1]/max_axis),int(resolution * extents[2]/max_axis)])
    return points, res#voxel_size, res

class SDF(nn.Module):
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
		b_flag = False
		self.decoder = nn.Sequential(
			nn.Linear(self.encoder.n_output_dims, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 64, bias=b_flag),
			nn.Softplus(beta = 10.0),
			nn.Linear(64, 1, bias=b_flag)
		)

		# write into numpy
		#params_enc = self.encoder.params.data.clone().cpu().numpy()
		#np.save('numpy/params_sdf_enc.npy', params_enc)

		params_enc = np.load('./numpy/params_sdf_enc.npy')
		self.encoder.params.data = torch.from_numpy(params_enc)

		idx = 0
		for m in self.decoder.modules():
			if isinstance(m, nn.Linear):
				if idx == 0:
					params_input = np.load("./numpy/params_input.npy")
					m.weight.data = torch.from_numpy(params_input)
				elif idx == 1:
					params_hidden_1 = np.load("./numpy/params_hidden_1.npy")
					m.weight.data = torch.from_numpy(params_hidden_1)
				elif idx == 2:
					params_hidden_2 = np.load("./numpy/params_hidden_2.npy")
					m.weight.data = torch.from_numpy(params_hidden_2)
				else:
					params_output = np.load("./numpy/params_output.npy")
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
	
	def loadMesh(self,mesh,batch,iteration):
		sdf=[]
		points=[]
		self.mesh = trimesh.load_mesh(mesh)
		self.name = mesh
		scale_fac = np.max(self.mesh.extents) 
		self.mesh = trimesh.Trimesh.apply_translation(self.mesh,-1 * (self.mesh.bounds[0]+self.mesh.bounds[1])/2)
		self.mesh = trimesh.Trimesh.apply_scale(self.mesh,1/scale_fac)


		if not os.path.exists(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz"):
			print("Sample SDF form Model")
			for i in range(iteration):
				points_ = np.random.rand(batch,3) - 0.5
				query = trimesh.proximity.ProximityQuery(self.mesh)
				sdf_ = query.signed_distance(points_)
				sdf.append(sdf_)
				points.append(points_)
				print("Batch Size:",i)
			self.sdf = np.array(sdf).reshape(-1,1) * -1
			self.points = np.array(points).reshape(-1,3)
			np.savetxt(mesh+"_points"+"_"+str(batch)+"_"+str(iteration)+".gz",self.points)
			np.savetxt(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz",self.sdf)
		else:
			self.sdf = np.loadtxt(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz")
			self.points = np.loadtxt(mesh+"_points"+"_"+str(batch)+"_"+str(iteration)+".gz")
		self.sdf = self.sdf.reshape(iteration,-1,1)
		self.points = self.points.reshape(iteration,-1,3)

	def saveMesh(self, gridres, file, it, level=0):

		grid,res = GenerateRasterPoints(np.array([0,0,0]), np.array([1,1,1]), gridres)
		print(grid.shape)
		print(res)
		grid_size = 64
		sdf = []
		spacing = 64*64*64
		for i in range(0,grid.shape[0],spacing):               
			grid_portion = grid[i:i+spacing]
			sdf_portion = self(torch.tensor(grid_portion, dtype=torch.float32).cuda()).detach().cpu().numpy()[:,0]
			sdf.append(sdf_portion)
		sdf = np.concatenate(sdf)
		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=0)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it) +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=5e-5)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it)+'_+5e-5' +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=5e-4)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it)+'_+5e-4' +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

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
		)

		
		# init encoder params from file
		file_dir = os.getcwd()
		
		#enc_params = np.load(os.path.join(file_dir, 'numpy/params_sdf_enc.npy'))
		#enc_p_torch = torch.from_numpy(enc_params)

		num_params = len(self.encoder_decoder.params)
		num_enc_params = 12196240
		start_params = num_params - num_enc_params
		# for i in range(num_enc_params):
		# 	tmp = enc_p_torch[i].detach()
		# 	self.encoder_decoder.params.data.index_fill_(0, torch.tensor(i+start_params, dtype=torch.int64).cuda(), torch.tensor(tmp).cuda())

		# init decoder params from file
		params_input = np.load(os.path.join(file_dir, './numpy/params_input.npy'))
		params_input_tensor = torch.from_numpy(params_input)

		## init input layer, notice NOT to fill unneccessary blanks (0.0)
		idx_tcnn = 0
		for i in range(params_input_tensor.shape[0]):
			for j in range(params_input_tensor.shape[1]):
				self.encoder_decoder.params[j+idx_tcnn].data.copy_(params_input_tensor[i, j])
			idx_tcnn += 32 # column num in TCNN
		
		## init hidden layers
		params = np.load(os.path.join(file_dir, './numpy/params_hidden_1.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 2048 # skip the input layer params
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]

		params = np.load(os.path.join(file_dir, './numpy/params_hidden_2.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 6144 # skip the input layer params
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]

		params = np.load(os.path.join(file_dir, './numpy/params_output.npy'))
		params_tensor = torch.from_numpy(params)
		idx_tcnn = 10240 # skip the input layer params
		for j in range(params_tensor.shape[0]):
			for k in range(params_tensor.shape[1]):
					self.encoder_decoder.params[k+idx_tcnn].data.copy_(params_tensor[j, k])
			idx_tcnn += params_tensor.shape[1]
		
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

	def loadMesh(self,mesh,batch,iteration):
		sdf=[]
		points=[]
		self.mesh = trimesh.load_mesh(mesh)
		self.name = mesh
		scale_fac = np.max(self.mesh.extents) 
		self.mesh = trimesh.Trimesh.apply_translation(self.mesh,-1 * (self.mesh.bounds[0]+self.mesh.bounds[1])/2)
		self.mesh = trimesh.Trimesh.apply_scale(self.mesh,1/scale_fac)

		if not os.path.exists(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz"):
			print("Sample SDF form Model")
			for i in range(iteration):
				points_ = np.random.rand(batch,3) - 0.5
				query = trimesh.proximity.ProximityQuery(self.mesh)
				sdf_ = query.signed_distance(points_)
				sdf.append(sdf_)
				points.append(points_)
				print("Batch Size:",i)
			self.sdf = np.array(sdf).reshape(-1,1) * -1
			self.points = np.array(points).reshape(-1,3)
			np.savetxt(mesh+"_points"+"_"+str(batch)+"_"+str(iteration)+".gz",self.points)
			np.savetxt(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz",self.sdf)
		else:
			self.sdf = np.loadtxt(mesh+"_sdf"+"_"+str(batch)+"_"+str(iteration)+".gz")
			self.points = np.loadtxt(mesh+"_points"+"_"+str(batch)+"_"+str(iteration)+".gz")
		self.sdf = self.sdf.reshape(iteration,-1,1)
		self.points = self.points.reshape(iteration,-1,3)	

	def saveMesh(self, gridres, file, it, level=0):

		grid,res = GenerateRasterPoints(np.array([0,0,0]), np.array([1,1,1]), gridres)
		print(grid.shape)
		print(res)
		grid_size = 64
		sdf = []
		spacing = 64*64*64
		for i in range(0,grid.shape[0],spacing):               
			grid_portion = grid[i:i+spacing]
			sdf_portion = self(torch.tensor(grid_portion, dtype=torch.float32).cuda()).detach().cpu().numpy()[:,0]
			sdf.append(sdf_portion)
		sdf = np.concatenate(sdf)
		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=0)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it) +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=5e-5)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it)+'_+5e-5' +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

		try:
			vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(sdf.reshape(res[0],res[1],res[2]), allow_degenerate=True, level=5e-4)
		except Exception as e:
			vertices = np.array(((0,0,0),(0.5,0.5,0.5),(0.2,0.3,0.8)))
			faces = np.array([0,1,2])
			normals = np.array((0.6,0.5,0.6))
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
		file_ = open(file+str(it)+'_+5e-4' +'.ply','wb')
		file_.write(trimesh.exchange.ply.export_ply(mesh))

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
		#print("encoder grad: ", model.decoder.params.grad)
		print("========= ========= ========= ========= ========= ========")
		print("decoder grad: ")
		for i in range(0, len(model.decoder), 2):
			if i % 2 == 0:
				w = model.decoder[i].weight.grad
				print("grad.shape: ", w.shape)
				output, input = w.shape
				if output == 1:
					row = 1
				else:
					row = 3
				for j in range(0, row):
					print("grad[",j*input,", ",(j+1)*input,"]: ", w[j])
				print("========= ========= ========= ========= ========= ========")
		
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
		
		dec_grad_0 = model.encoder_decoder.params.grad[0:2048]
		dec_grad_1 = model.encoder_decoder.params.grad[2048:6144] # 1st hidden layer
		dec_grad_2 = model.encoder_decoder.params.grad[6144:10240] # 2nd hidden layer
		dec_grad_3 = model.encoder_decoder.params.grad[10240:11264] # 3rd hidden layer
		
		print_TCNN_layer_weight("dec_grad_0", dec_grad_0, 32, 64)
		print_TCNN_layer_weight("dec_grad_1", dec_grad_1, 64, 64)
		print_TCNN_layer_weight("dec_grad_2", dec_grad_2, 64, 64)
		print_TCNN_layer_weight("dec_grad_3", dec_grad_3, 64, 1)

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

		# leverage TCNN SDF
		# NOTE: if leveraging TCNN, please leverage print_TCNN_SDF(model) to print temporary variables
		model = SDF_TCNN(True, n_levels=1, log2_hashmap_size=15, base_resolution=4, smoothstep=False).to(device)
		
		# leverage Pytorch SDF
		# NOTE: if leveraging Pytorch, please leverage print_torch_SDF(model) to print temporary variables
		#model = SDF(True, n_levels=1, base_resolution=4).to(device)

		model.loadMesh("./data/Armadillo.ply", 1024, 3000)

		torch.cuda.nvtx.range_push('training_preparation')
		torch.manual_seed(0)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		#optimizer = Adam(model.parameters(), 2.0e-4)
		optimizer = apex.optimizers.FusedAdam(model.parameters(), lr = 2.0e-4)
		
		iter_i = 0
		fake_nablas = torch.ones([1], device='cuda')
		torch.cuda.nvtx.range_pop()

		from torch.cuda.amp import GradScaler
		from torch.cuda.amp import autocast
		scaler = GradScaler()

		num = 0
		total_time = 0

		with tqdm(range(10000)) as pbar:
			for i in pbar:
				sdf_list = []
				point_list = []
				for _ in range(1):
					it = int(random.random() * 3000)
					sdf_list.append(model.sdf[0])
					point_list.append(model.points[0])
				ref = torch.tensor(np.concatenate(sdf_list),device='cuda',dtype=torch.float32).squeeze()
				x = torch.tensor(np.concatenate(point_list),device='cuda',dtype=torch.float32)

				torch.cuda.nvtx.range_push('python_forward_start')
				sdf, nablas = model.forward_with_nablas(x)
				torch.cuda.nvtx.range_pop()

				torch.cuda.nvtx.range_push('python_nablas_norm')
				nablas_norm: torch.Tensor = nablas.norm(dim=-1)
				torch.cuda.nvtx.range_pop()

				torch.cuda.nvtx.range_push('python_loss_compute')
				optimizer.zero_grad()

				r = torch.rand([1024, 3], device='cuda', dtype=torch.float32) - 0.5
				sdf_, nablas = model.forward_with_nablas(r)
				eikonal_loss = torch.sum(torch.abs(torch.norm(nablas, p=2, dim=1) - 1)) * 0.0001
				loss = F.mse_loss(sdf[..., 0], ref) + eikonal_loss
				
				loss.backward()
				optimizer.step()

				# print time consumed
				# NOTE: if use print_bwdbwd_time, one needs to comment loss.backward() and optimizer.step()
				#total_time, num = print_bwdbwd_time(scaler, loss, optimizer, total_time, num)

				torch.cuda.nvtx.range_pop()

				pbar.set_postfix(loss=loss.item())

				if (torch.isnan(loss)):
					break
				
				if(iter_i%1000==0):
					model.saveMesh(128, "result/test", iter_i)
				iter_i = iter_i + 1


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


