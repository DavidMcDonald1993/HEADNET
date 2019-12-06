import keras.backend as K
import tensorflow as tf

from tensorflow.train import AdamOptimizer, RMSPropOptimizer
from tensorflow.train import AdagradOptimizer, GradientDescentOptimizer

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

from headnet.poincare_layers import exp_map_x, lambda_x#, exp_map_0

# def minkowski_dot(x, y):
# 	assert len(x.shape) == len(y.shape)
# 	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

# def normalise_to_hyperboloid(x):
# 	# x = x[:,:-1]
# 	# t = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) + 1)
# 	# return K.concatenate([x, t], axis=-1)
# 	return x / K.maximum( K.sqrt( K.maximum( K.abs(minkowski_dot(x, x)), 0.) ), K.epsilon())

# def exponential_mapping( p, x ):

# 	r = K.sqrt( K.maximum(
# 			minkowski_dot(x, x), 0) )
# 	####################################################
# 	exp_map_p = tf.cosh(r) * p

# 	idx = tf.where(r > 0)[:,0]
# 	non_zero_norm = tf.gather(r, idx)

# 	z = tf.gather(x, idx) / non_zero_norm

# 	updates = tf.sinh(non_zero_norm) * z
# 	# updates = normalise_to_hyperboloid(updates)
# 	dense_shape = tf.cast( tf.shape(p), tf.int64)
# 	# exp_map = tf.scatter_update(ref=p, indices=idx, updates=updates)
# 	exp_map_x = tf.scatter_nd(indices=idx[:,None],
# 		updates=updates, shape=dense_shape)

# 	exp_map = exp_map_p + exp_map_x
# 	#####################################################
# 	# z = x / K.maximum(r, K.epsilon()) # unit norm
# 	# r = K.minimum(r, 1.)
# 	# exp_map = tf.cosh(r) * p + tf.sinh(r) * z
# 	#####################################################
# 	exp_map = normalise_to_hyperboloid(exp_map) # account for floating point imprecision

# 	return exp_map

# def project_onto_tangent_space(
# 	hyperboloid_point,
# 	minkowski_ambient):
# 	return minkowski_ambient + minkowski_dot(hyperboloid_point, minkowski_ambient) * hyperboloid_point

# class ExponentialMappingOptimizer(optimizer.Optimizer):

# 	def __init__(self,
# 		lr=0.1,
# 		use_locking=False,
# 		name="ExponentialMappingOptimizer"):
# 		super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
# 		self.lr = lr
# 		# self.euclidean_optimizer = GradientDescentOptimizer(0.01)
# 		self.euclidean_optimizer = AdamOptimizer(1e-3)
# 		# self.euclidean_optimizer = RMSPropOptimizer(0.001)

# 	def _apply_dense(self, grad, var):
# 		assert False
# 		spacial_grad = grad[:,:-1]
# 		t_grad = -1 * grad[:,-1:]

# 		ambient_grad = tf.concat([spacial_grad, t_grad],
# 			axis=-1)
# 		tangent_grad = project_onto_tangent_space(var,
# 			ambient_grad)

# 		exp_map = exponential_mapping(var,
# 			- self.lr * tangent_grad)

# 		return tf.assign(var, exp_map)

# 	def _apply_sparse(self, grad, var):

# 		if "hyperbolic" in var.name:

# 			indices = grad.indices
# 			values = grad.values

# 			p = tf.gather(var, indices,
# 				name="gather_apply_sparse")

# 			spacial_grad = values[:, :-1]
# 			t_grad = -1 * values[:, -1:]

# 			ambient_grad = tf.concat([spacial_grad, t_grad],
# 				axis=-1, name="optimizer_concat")

# 			tangent_grad = project_onto_tangent_space(p,
# 				ambient_grad)
# 			exp_map = exponential_mapping(p,
# 				- self.lr * tangent_grad)

# 			return tf.scatter_update(ref=var,
# 				indices=indices, updates=exp_map,
# 				name="scatter_update")

# 		else:
# 			# euclidean update using Adam optimizer
# 			return self.euclidean_optimizer.apply_gradients( [(grad, var), ] )

c = 1.

# def norm( x, axis=-1, keepdims=True):
# 	return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))

# def lambda_x(x, axis=-1, ):
# 	norm_x = norm(x, axis=axis, keepdims=True)
# 	return 2 / (1 - c * norm_x ** 2)

class PoincareOptimizer(optimizer.Optimizer):

	def __init__(self, 
		lr=1e-1, 
		use_locking=False,
		name="PoincareOptimizer"):
		super(PoincareOptimizer, self).__init__(use_locking, 
			name)
		self.lr = lr
		self.euclidean_optimizer = AdamOptimizer(1e-3,)

	def _apply_dense(self, grad, var):
		if "poincare" in var.name:
			# assert False
			# poincare update
			# return tf.assign (var, var)
			grad = grad / lambda_x(var) ** 2
			# exp_map = exp_map_0(-self.lr * grad,)
			# exp_map = var + grad
			# exp_map = grad
			exp_map = exp_map_x( -1 * self.lr * grad, 
				var)
			return tf.assign(var, exp_map)
		else:
			# return tf.assign(var, var)
			# euclidean update
			gvs = [(grad, var), ] 
			gvs = [(tf.clip_by_norm(grad, 1e-0), var) 
				for grad, var in gvs]
			return self.euclidean_optimizer.apply_gradients( 
				gvs	
			)

	def _apply_sparse(self, grad, var):
		assert False

		indices = grad.indices
		values = grad.values

		var_ = tf.gather(var, indices, name="gather_apply_sparse")

		if "poincare" in var.name:
			# poincare update
			grad = values / lambda_x(var_) ** 2 
			exp_map = exp_map_x(-self.lr * grad, 
				var_)
			return tf.scatter_update(ref=var, 
				indices=indices, 
				updates=exp_map, 
				name="scatter_update")
		else:
			# euclidean update
			gvs = [(grad, var), ] 
			# gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs]
			return self.euclidean_optimizer.apply_gradients( 
				gvs	
			)

	# def poincare_exponential_map(self, v, x):
	# 	'''
	# 	http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf
	# 	'''
	# 	l_x = lambda_x(x)
	# 	norm_v = norm(v)

	# 	cosh_lambda_norm_v = tf.cosh(l_x * norm_v)
	# 	sinh_lambda_norm_v = tf.sinh(l_x * norm_v)
	# 	x_dot_unit_v = K.batch_dot(x, v / norm_v, axes=1)

	# 	u1 = l_x * (cosh_lambda_norm_v + x_dot_unit_v * sinh_lambda_norm_v)

	# 	u2 = 1 / norm_v * sinh_lambda_norm_v

	# 	d = 1 + (l_x - 1) * cosh_lambda_norm_v + l_x * x_dot_unit_v * sinh_lambda_norm_v

	# 	return u1 / d * x + u2 / d * v