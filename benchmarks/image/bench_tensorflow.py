# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   bench_tensorflow.py
# @author Thomas Müller, NVIDIA
# @brief  Generates performance data for comparison with our fully fused network.

import argparse
import commentjson as json
import glob
import math
import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class Function:
	def __init__(self, domain, n_channels, n_dims, wraparound_dims, n_conditionals, n_raw_conditionals):
		self.domain = domain
		self.n_channels = n_channels
		self.n_dims = n_dims
		self.wraparound_dims = wraparound_dims
		self.n_conditionals = n_conditionals
		self.n_raw_conditionals = n_raw_conditionals

	def __call__(self, xs):
		raise NotImplementedError

class Image(Function):
	def __init__(self, filename):
		self.filename = filename
		paths = glob.glob(os.path.join(IMAGES_DIR, self.filename + ".*"))
		if not paths:
			raise ValueError(f"Invalid image name '{filename}''")
		path = paths[0] # Use first path that exists
		self.data = read_image(path)
		if self.data.shape[-1] > 3:
			self.data = self.data[:,:,0:3]
		self.data_tf = tf.constant(self.data, dtype=tf.float32)
		super().__init__('unit', self.data.shape[-1], 2, {}, 0, 0)

	def __call__(self, xs):
		shape = self.data.shape
		indices = (xs * np.array([shape[1], shape[0]])).astype(np.uint32)
		indices[:, 0] = np.clip(indices[:, 0], a_min=0, a_max=shape[1]-1)
		indices[:, 1] = np.clip(indices[:, 1], a_min=0, a_max=shape[0]-1)
		return self.data[indices[:, 1], indices[:, 0]]

	def eval_tf(self, xs):
		shape = self.data_tf.shape
		indices = tf.cast(xs * tf.constant([shape[1], shape[0]], dtype=tf.float32), tf.int32)
		indices_clipped = tf.stack([
			tf.clip_by_value(indices[:, 1], 0, shape[0]-1),
			tf.clip_by_value(indices[:, 0], 0, shape[1]-1),
		], axis=-1)
		return tf.gather_nd(self.data_tf, indices_clipped)

class OneBlob:
	def __init__(self, n_bins, n_levels):
		self.n_bins = n_bins
		self.n_levels = n_levels
		self.radius = 0.5 / n_bins

	def __call__(self, inputs, wraparound, name, dtype=None):
		def gaussian_cdf_approx(x, radius):
			return 0.5 * (1 + tf.tanh(1.12 * x / (math.sqrt(2.) * radius)))

		def gaussian_cdf(x, radius):
			return 0.5 * (1 + tf.erf(x / (math.sqrt(2.) * radius)))

		dims = inputs.shape[-1]
		with tf.name_scope(name):
			# When there are no input dims, there is nothing to encode.
			# This special case is needed because tf.reshape does strange
			# things when 0-dims are involved.
			if dims == 0:
				return inputs
			results = []
			boundaries = tf.linspace(0., 1., self.n_bins + 1)
			boundaries = tf.reshape(boundaries, [1 for _ in inputs.shape] + [-1])

			for level in range(self.n_levels):
				with tf.name_scope(f"level{level}"):
					scale = self.n_bins**level

					# We use the absolute value here just in case the inputs are erroneously negative.
					# Even a negative epsilon would totally wreck the following code.
					if level == 0:
						scaled = tf.abs(inputs)
					else:
						scaled = tf.abs(inputs * scale) % 1

					diffs = boundaries - scaled[..., tf.newaxis]
					cdfs = gaussian_cdf_approx(diffs, self.radius)
					result = cdfs[...,1:] - cdfs[...,:-1]

					# print_op = tf.print("result: ", result)

					# In the outermost level we don't want to carry over...
					# otherwise we introduce ambiguities.
					if level != 0 or wraparound:
						cdfs_right = gaussian_cdf_approx(diffs + 1., self.radius)
						cdfs_left = gaussian_cdf_approx(diffs - 1., self.radius)
						result = result + cdfs_right[...,1:] - cdfs_right[...,:-1] + cdfs_left[...,1:] - cdfs_left[...,:-1]

					# with tf.control_dependencies([print_op]):
					result = result / scale

					results.append(result)

			result = tf.concat(results, axis=-1)
			result = tf.reshape(result, [-1, self.n_bins * self.n_levels * dims])
			return result

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using TensorFlow.")

	parser.add_argument("-c", "--config", default="config_oneblob.json", type=str, help="JSON config filename")
	parser.add_argument("-i", "--image", default="albert", type=str, help="Image to match")

	args = parser.parse_args()
	return args

def linear_layer(inputs, units, dtype, name, use_biases=True):
	# inputs: 2d Tensor, shape=(batch, in_units).
	# units: Integer, dimensionality of the output space.

	assert len(inputs.shape) == 2
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
		weights = tf.get_variable("weights", (inputs.shape[1], units),
									initializer=tf.glorot_uniform_initializer())

		if use_biases:
			biases = tf.get_variable("biases", (units),
											initializer=tf.constant_initializer())

	result = tf.matmul(tf.cast(inputs, dtype), tf.cast(weights, dtype))
	if use_biases:
		result = result + tf.cast(biases, dtype)

	return tf.cast(result, tf.float32)

def activation(tensor, kind):
	kind = kind.lower()
	if kind == "relu":
		return tf.nn.relu(tensor)
	elif kind == "relu6":
		return tf.nn.relu6(tensor)
	elif kind == "elu":
		return tf.nn.elu(tensor)
	elif kind == "selu":
		return tf.nn.selu(tensor)
	elif kind == "leaky_relu":
		return tf.nn.leaky_relu(tensor)
	elif kind == "none":
		return tensor
	else:
		assert(False)

def compute_gradients(loss, variables, loss_scale):
	with tf.name_scope("gradient_computation"):
		gradients = tf.gradients(loss * loss_scale, variables)
		# Create zero gradients for None entries
		zeros = [tf.zeros_like(var) for var in variables]
		gradients = [grad / loss_scale if grad is not None else None for grad in gradients]
		finites = [tf.reduce_all(tf.is_finite(grad)) if grad is not None else None for grad in gradients]
		gradients = [tf.where(finite, grad, zero) if grad is not None else None for finite, grad, zero in zip(finites, gradients, zeros)]

		all_finite = tf.reduce_all([f for f in finites if f is not None])

	return gradients, all_finite

def get_train_op(config, variables, gradients, optimizer, clip_norm=0):
	if clip_norm > 0:
		gradients, gradients_norm = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)
	else:
		gradients_norm = tf.global_norm(gradients)

	if gradients and not all(grad is None for grad in gradients):
		train_op = optimizer.apply_gradients(zip(gradients, variables), name="apply_gradients")
	else:
		train_op = tf.no_op(name="apply_gradients")

	return train_op, gradients_norm

def make_graph():
	uniform = tfp.distributions.Uniform()
	input_tensor = uniform.sample((batch_size_tensor, target_fun.n_dims))
	target_tensor = target_fun.eval_tf(input_tensor)

	current_tensor = encoding(input_tensor, False, "encoding")

	for i in range(config["network"]["n_hidden_layers"]):
		current_tensor = linear_layer(current_tensor, config["network"]["n_neurons"], tf.float16, f"fc{i}", False)
		current_tensor = activation(current_tensor, config["network"]["activation"])

	output_tensor = linear_layer(current_tensor, target_fun.n_channels, tf.float16, f"fc_out", False)
	output_tensor = activation(output_tensor, config["network"]["output_activation"])

	relative_l2_error = (target_tensor - output_tensor)**2 / (tf.stop_gradient(output_tensor)**2 + 0.01)
	loss = tf.math.reduce_mean(relative_l2_error)

	LOSS_SCALE = 128
	variables = tf.trainable_variables()
	gradients, _ = compute_gradients(loss, variables, LOSS_SCALE)
	train_op, _ = get_train_op(config, variables, gradients, optimizer)

	return train_op, loss, input_tensor, output_tensor

if __name__ == "__main__":
	tf.disable_eager_execution()
	args = get_args()

	# Initialize non-TF stuff
	with open(os.path.join(DATA_DIR, args.config)) as config_file:
		config = json.load(config_file)

	target_fun = Image(os.path.join(IMAGES_DIR, args.image))
	encoding = OneBlob(config["encoding"]["n_bins"], 1)

	# Initialize TF graph
	batch_size_tensor = tf.placeholder(tf.int32, shape=[])
	optimizer = tf.train.AdamOptimizer(config["optimizer"]["learning_rate"], config["optimizer"]["beta1"], config["optimizer"]["beta2"], config["optimizer"]["epsilon"])
	train_op, loss, input_tensor, output_tensor = make_graph()

	# Variables for saving/displaying image results
	resolution = 1024
	img_shape = (resolution, resolution, target_fun.n_channels)

	half_dx = 0.5 / resolution
	xs = np.linspace(half_dx, 1-half_dx, resolution)
	xv, yv = np.meshgrid(xs, xs)

	xy = np.stack((xv.flatten(), yv.flatten())).transpose()
	gt = np.reshape(target_fun(xy), img_shape)
	write_image("reference.jpg", gt)

	# Enable XLA compiler (important for good TensorFlow performance)
	session_config = tf.ConfigProto()
	session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

	timer = time.perf_counter()

	# Run the network
	with tf.Session(config=session_config) as sess:
		PRINT_INTERVAL = 100

		bench_result = { "tensorflow": [] }

		for batch_size in [2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21]:
			N_ITERS = 1000
			PRINT_INTERVAL = 100

			output_dummy_variable = tf.Variable(tf.zeros(shape=[batch_size, target_fun.n_channels], dtype=tf.float32), trainable=False)
			sess.run(tf.initialize_all_variables())

			# Training
			c = lambda it, _, __: tf.less(it, PRINT_INTERVAL)
			def body(it, sequencer, _):
				with tf.control_dependencies([sequencer]):
					local_train_op, local_loss, _, _ = make_graph()
				with tf.control_dependencies([local_train_op]):
					next_sequencer = tf.ones([])
					return it+1, next_sequencer, local_loss

			train_op, _, loss = tf.while_loop(c, body, [0, 1., 0.], parallel_iterations=1)

			throughputs = []
			for i in range(0, N_ITERS, PRINT_INTERVAL):
				if i % PRINT_INTERVAL == 0:
					_, loss_val = sess.run([train_op, loss], feed_dict={ batch_size_tensor: batch_size })
					old_time = timer
					timer = time.perf_counter()
					elapsed_time = timer - old_time
					throughput = PRINT_INTERVAL * batch_size / elapsed_time
					throughputs.append(throughput)
					print(f"Iteration#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs] thp={throughput}/s")
				else:
					sess.run([train_op], feed_dict={ batch_size_tensor: batch_size })


			img = np.reshape(sess.run(output_tensor, feed_dict={ input_tensor: xy, batch_size_tensor: xy.shape[0] }), img_shape)
			filename = f"{batch_size}-after-{N_ITERS}-iters-tensorflow.jpg"
			print(f"Saving {filename}")
			write_image(filename, img)

			mean_training_throughput = np.mean(throughputs[1:])

			print(f"Finished training benchmark. Mean throughput is {mean_training_throughput}/s. Waiting 10s for GPU to cool down.")
			time.sleep(10)

			# Inference
			_, _, _, tmp_out = make_graph()
			inference_op = output_dummy_variable.assign(tmp_out)

			N_ITERS *= 2
			PRINT_INTERVAL *= 2

			throughputs = []
			for i in range(N_ITERS):
				sess.run(inference_op, feed_dict={ batch_size_tensor: batch_size })
				if i % PRINT_INTERVAL == 0:
					old_time = timer
					timer = time.perf_counter()
					elapsed_time = timer - old_time
					throughput = PRINT_INTERVAL * batch_size / elapsed_time
					throughputs.append(throughput)
					print(f"Iteration#{i}: time={int(elapsed_time*1000000)}[µs] thp={throughput}/s")

			mean_inference_throughput = np.mean(throughputs[1:])

			print(f"Finished inference benchmark. Mean throughput is {mean_inference_throughput}/s. Waiting 10s for GPU to cool down.")
			time.sleep(10)

			# Mean throughput (discounting the first one due to XLA compilation)
			bench_result["tensorflow"].append(
				{
					"batch_size" : batch_size,
					"training_throughput" : mean_training_throughput,
					"inference_throughput" : mean_inference_throughput,
				}
			)

		with open("bench_result_tensorflow.json", "w") as f:
			json.dump(bench_result, f)
