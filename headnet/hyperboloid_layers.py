import numpy as np

import keras.backend as K
from keras.layers import Input, Layer, Concatenate
from keras.models import Model
from keras.initializers import Constant, RandomNormal
from keras.regularizers import l2
from keras import regularizers, activations


import tensorflow as tf
from tensorflow.python.framework import function

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops, control_flow_ops
# from tensorflow.python.training import optimizer
# from tensorflow.keras.optimizers import SGD

# def kullback_leibler_divergence(x, sigmas):

# 	# ignore zero t coordinate
# 	x = x[...,:-1]

# 	k = K.int_shape(x)[-1]

# 	sigmas = K.maximum(sigmas, K.epsilon())

# 	source_sigma = sigmas[:,:1]
# 	target_sigma = sigmas[:,1:]

# 	sigma_ratio = target_sigma / source_sigma

# 	trace_fac = K.sum(sigma_ratio,
# 		axis=-1, keepdims=True)

# 	mu_sq_diff = K.sum(x ** 2 / \
# 		source_sigma,
# 		axis=-1, keepdims=True) # assume sigma inv is diagonal

# 	log_det = K.sum(K.log(sigma_ratio), axis=-1, keepdims=True)

# 	return 0.5 * (trace_fac + mu_sq_diff - k - log_det)

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def parallel_transport(p, q, x):
	# p goes to q
	alpha = -minkowski_dot(p, q)

	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
		K.maximum(alpha + 1, K.epsilon())

@function.Defun(K.floatx(), K.floatx())
def norm_grad(x, dy):
	return dy*(x/(tf.norm(x, axis=-1, keepdims=True) + K.epsilon() ))

@function.Defun(K.floatx(), 
	grad_func=norm_grad, 
	shape_func=lambda op: \
		[op.inputs[0].get_shape().as_list()[:-1] + [1]])
def norm(x, ):
    return tf.norm(x, axis=-1, keepdims=True)

def minkowski_norm(x):
	return K.sqrt( K.maximum(minkowski_dot(x, x), 0.) )

def normalise_to_hyperboloid(x):
	x = x[...,:-1]
	t = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) + 1)
	return K.concatenate([x, t], axis=-1)
	# return x / K.sqrt( - minkowski_dot(x, x) )
	# return x / K.maximum( \
	# 	K.sqrt( -minkowski_dot(x, x) )  ,
	# 	K.epsilon())

# def exponential_mapping( p, x ):

# 	# minkowski unit norm
# 	r = minkowski_norm(x)
# 	# r = norm(x[...,:-1]) # use euclidean norm since x has a 0 t co-ordinate

# 	# r = tf.verify_tensor_all_finite(r, "fail after r")

# 	# x = x / K.maximum(r, K.epsilon())

# 	# exp_map = tf.cosh(r) * p + tf.sinh(r) * x

# 	######################################

# 	idx = tf.where(r > K.epsilon())[:,0]

# 	cosh_r = tf.cosh(r)
# 	# cosh_r = tf.verify_tensor_all_finite(cosh_r, "fail after cosh")
# 	exp_map_p = cosh_r * p

# 	non_zero_norm = tf.gather(r, idx)

# 	z = tf.gather(x, idx)

# 	updates = tf.sinh(non_zero_norm) * z
# 	# updates = tf.verify_tensor_all_finite(updates, "fail after sinh")
# 	dense_shape = tf.shape(p, out_type=tf.int64)
# 	exp_map_x = tf.scatter_nd(indices=idx[:,None],
# 		updates=updates, 
# 		shape=dense_shape)

# 	exp_map = exp_map_p + exp_map_x

# 	###########################################

# 	# exp_map = tf.verify_tensor_all_finite(exp_map, "fail before normal")
# 	exp_map = normalise_to_hyperboloid(exp_map) # account for floating point imprecision
# 	# exp_map = tf.verify_tensor_all_finite(exp_map, "fail after normal")

# 	return exp_map

def exp_map_0(x):
	# x = tf.verify_tensor_all_finite(x, "fail exp 0")
	assert len(x.shape) > 1

	r = norm(x) 
	# unit norm
	x = x / K.maximum(r, K.epsilon())
	# x = tf.keras.backend.l2_normalize(x, axis=-1)

	x = tf.sinh(r) * x
	t = tf.cosh(r)

	# t = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) + 1)
	return K.concatenate([x, t], axis=-1)

	# mu_zero = K.concatenate([K.zeros_like(x[..., :-1]), 
	# 	K.ones_like(x[...,-1:])], axis=-1)
	# return exponential_mapping(mu_zero, x)

def logarithmic_map(p, x):
	assert len(p.shape) == len(x.shape)

	alpha = -minkowski_dot(p, x)

	alpha = K.maximum(alpha, 1 + K.epsilon())

	return tf.acosh(alpha) * (x - alpha * p) / \
		K.maximum(K.sqrt(K.maximum(alpha ** 2 - 1., 0.)),
			K.epsilon())

def log_map_0(x):
	# assert len(x.shape) == 2
	# dim = x.shape[-1]-1
	mu_zero = K.concatenate([K.zeros_like(x[..., :-1]), 
		K.ones_like(x[...,-1:])], axis=-1)
	# print (x.shape, mu_zero.shape)
	# raise SystemExit
	return logarithmic_map(mu_zero, x)

# def bias_add(X, b):
# 	if len(b.shape) != len(X.shape):
# 		assert len(b.shape) == 1
# 		b = K.reshape(b, [1]*(len(X.shape)-1) + [b.shape[0]])
# 	log_b = log_map_0(b)
# 	mu_zero = K.concatenate([K.zeros_like(b[..., :-1]), 
# 		K.ones_like(b[...,-1:])], axis=-1)
# 	P_log_b = parallel_transport(mu_zero, X, log_b)
# 	return exponential_mapping(X, P_log_b)

# def mobius_fx(f, x):
# 	return exp_map_0(f(log_map_0(x)))

# def matrix_vector_multiplication(W, x):
# 	# TODO
# 	pass

# def hyperboloid_initializer(shape, r_max=1e-3):

# 	def poincare_ball_to_hyperboloid(X, append_t=True):
# 		x = 2 * X
# 		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
# 		if append_t:
# 			x = K.concatenate([x, t], axis=-1)
# 		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

# 	w = tf.random_uniform(shape=shape, 
# 		minval=-r_max, maxval=r_max, dtype=K.floatx())
# 	return poincare_ball_to_hyperboloid(w)

# class HyperboloidGaussianEmbeddingLayer(Layer):

# 	def __init__(self,
# 		num_nodes,
# 		embedding_dim,
# 		**kwargs):
# 		super(HyperboloidGaussianEmbeddingLayer, self).__init__(**kwargs)
# 		self.num_nodes = num_nodes
# 		self.embedding_dim = embedding_dim
# 		self.mu_zero = K.constant(np.append(np.zeros((1, 1, self.embedding_dim)), np.ones((1,1,1)), axis=-1))

# 	def build(self, input_shape):
# 		# Create a trainable weight variable for this layer.
# 		self.embedding = self.add_weight(name='hyperbolic_embedding',
# 		  shape=(self.num_nodes, self.embedding_dim),
# 		  initializer=hyperboloid_initializer,
# 		  trainable=True)
# 		assert self.embedding.shape[1] == self.embedding_dim + 1
# 		self.covariance = self.add_weight(name='euclidean_covariance',
# 		  shape=(self.num_nodes, self.embedding_dim),
# 			initializer="zeros",
# 		  trainable=True)
# 		super(HyperboloidGaussianEmbeddingLayer, self).build(input_shape)

# 	def call(self, idx):

# 		source_embedding = tf.gather(self.embedding, idx[:,:1])
# 		target_embedding = tf.gather(self.embedding, idx[:,1:])

# 		to_tangent_space = logarithmic_map(source_embedding,
# 			target_embedding)
# 		to_tangent_space_mu_zero = parallel_transport(source_embedding,
# 			self.mu_zero,
# 			to_tangent_space)

# 		sigmas = tf.gather(self.covariance, idx)

# 		sigmas = K.elu(sigmas, alpha=1.) + 1

# 		kds = kullback_leibler_divergence(\
# 			to_tangent_space_mu_zero,
# 			sigmas=sigmas)

# 		kds = K.squeeze(kds, axis=-1)

# 		return kds

# 	def compute_output_shape(self, input_shape):
# 		return (input_shape[0], input_shape[1] - 1, )

# 	def get_config(self):
# 		base_config = super(HyperboloidGaussianEmbeddingLayer,
# 			self).get_config()
# 		base_config.update({"num_nodes": self.num_nodes,
# 			"embedding_dim": self.embedding_dim})
# 		return base_config

# def mu_zero_init(shape, dtype=K.floatx()):
# 	return K.concatenate([K.zeros(shape[0], dtype=dtype), 
# 					K.ones(1, dtype=dtype )], axis=-1)

# def bias_init(shape, dtype=K.floatx()):
# 	X = tf.random_normal(shape=shape, stddev=1e-3, dtype=dtype)
# 	t = K.sqrt(K.sum(X**2, axis=-1, keepdims=True) + 1)
# 	return K.concatenate([X, t], axis=-1)

# class HyperboloidFeedForwardLayer(Layer):

# 	def __init__(self,
# 		units,
# 		activation=None,
# 		**kwargs):
# 		super(HyperboloidFeedForwardLayer, self).__init__(**kwargs)
# 		self.units = units
# 		self.activation = activation
		

# 	def build(self, input_shape):
# 		# Create a trainable weight variable for this layer.
# 		# self.W = self.add_weight(name='euclidean_weight', 
# 		#   shape=( input_shape[-1], self.units, ),
# 		#   initializer="glorot_uniform",
# 		# #   regularizer=regularizers.l2(reg),
# 		#   trainable=True) # euclidean linear map

# 		self.bias = self.add_weight(name='hyperboloid_bias', 
# 		  shape=(self.units, ),
# 		#   initializer=mu_zero_init,
# 		  initializer=bias_init,
# 		#   regularizer=regularizers.l2(reg),
# 		  trainable=True) # hyperbolic (poincare) weight

# 		super(HyperboloidFeedForwardLayer, self).build(input_shape)

# 	def call(self, x):
# 		# Wx = matrix_vector_multiplication(self.W, x)
# 		Wx_plus_b = bias_add(x, self.bias)
# 		f = activations.get(self.activation)
# 		if f is None:
# 			return Wx_plus_b
# 		else:
# 			return mobius_fx(f, Wx_plus_b)

# 	def compute_output_shape(self, input_shape):
# 		return tuple(list(input_shape)[:-1] + [self.units+1])
	
# 	def get_config(self):
# 		base_config = super(HyperboloidFeedForwardLayer, self).get_config()
# 		base_config.update({"units": self.units, 
# 			"activation": self.activation})
# 		return base_config