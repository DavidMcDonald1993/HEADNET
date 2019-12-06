import keras.backend as K
import tensorflow as tf

from tensorflow.train import AdamOptimizer#, RMSPropOptimizer
# from tensorflow.train import AdagradOptimizer, GradientDescentOptimizer

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

from .hyperboloid_layers import exponential_mapping

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return (K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) 
		- x[...,-1:] * y[...,-1:])

def minkowski_norm(x):
	return K.sqrt( K.maximum(minkowski_dot(x, x), 0.) )

# def normalise_to_hyperboloid(x):
# 	# x = x[:,:-1]
# 	# t = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) + 1)
# 	# return K.concatenate([x, t], axis=-1)
# 	return x / K.sqrt( - minkowski_dot(x, x) )
# 	# return x / K.maximum( \
# 	# 	K.sqrt( -minkowski_dot(x, x) )  ,
# 	# 	K.epsilon())

# def exponential_mapping( p, x ):

# 	# p = tf.verify_tensor_all_finite(p, "fail in p")

# 	# minkowski unit norm
# 	r = minkowski_norm(x)
# 	# r = tf.verify_tensor_all_finite(r, "fail in r")

# 	x = x / K.maximum(r, K.epsilon())

# 	# x = tf.verify_tensor_all_finite(x, "fail in mink norm")

# 	####################################################

# 	# r = K.minimum(r, 1e-0)

# 	# idx = (r > 1e-7)[:,0]

# 	# updates = tf.cosh(r) * p + tf.sinh(r) * x 
# 	# updates = normalise_to_hyperboloid(updates)

# 	# return tf.where(idx, updates, p)

# 	####################################################

# 	idx = tf.where(r > 0)[:,0]

# 	# clip
# 	r = K.minimum(r, 1e-0)

# 	cosh_r = tf.cosh(r)
# 	# cosh_r = tf.verify_tensor_all_finite(cosh_r, 
# 	# 	"fail in cosh r")
# 	exp_map_p = cosh_r * p

# 	# exp_map_p = tf.verify_tensor_all_finite(exp_map_p, 
# 		# "fail in exp_map_p")

# 	non_zero_norm = tf.gather(r, idx)

# 	z = tf.gather(x, idx)

# 	updates = tf.sinh(non_zero_norm) * z
# 	# updates = tf.verify_tensor_all_finite(updates, 
# 	# 	"fail in updates")
# 	dense_shape = tf.shape(p, out_type=tf.int64)
# 	exp_map_x = tf.scatter_nd(indices=idx[:,None],
# 		updates=updates, 
# 		shape=dense_shape)

# 	exp_map = exp_map_p + exp_map_x

# 	#####################################################
# 	# z = x / K.maximum(r, K.epsilon()) # unit norm
# 	# exp_map = tf.cosh(r) * p + tf.sinh(r) * x
# 	#####################################################
# 	# exp_map = tf.verify_tensor_all_finite(exp_map, "error before")
# 	exp_map = normalise_to_hyperboloid(exp_map) # account for floating point imprecision
# 	# exp_map = tf.verify_tensor_all_finite(exp_map, "error after")

# 	return exp_map

def project_onto_tangent_space(
	hyperboloid_point,
	minkowski_ambient):
	ret = minkowski_ambient + \
		minkowski_dot(hyperboloid_point, minkowski_ambient) * \
		hyperboloid_point
	return ret

class ExponentialMappingOptimizer(optimizer.Optimizer):

	def __init__(self,
		lr=0.1,
		use_locking=False,
		name="ExponentialMappingOptimizer"):
		super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
		self.lr = lr
		# self.euclidean_optimizer = GradientDescentOptimizer(1e-2)
		self.euclidean_optimizer = AdamOptimizer()
		# self.euclidean_optimizer = RMSPropOptimizer(1e-3)

	def _apply_dense(self, grad, var):
		if "hyperboloid" in var.name:
			assert False
			spacial_grad = grad[...,:-1]
			t_grad = -1 * grad[...,-1:]

			ambient_grad = tf.concat([spacial_grad, t_grad],
				axis=-1)
			tangent_grad = project_onto_tangent_space(var,
				ambient_grad)

			exp_map = exponential_mapping(var,
				- self.lr * tangent_grad)

			return tf.assign(var, exp_map)

		else: 

			# euclidean update using Adam optimizer
			return self.euclidean_optimizer.apply_gradients( 
				[(grad, var), ] 
			)

	def _apply_sparse(self, grad, var):

		assert False

		if "hyperbolic" in var.name:

			indices = grad.indices
			values = grad.values

			p = tf.gather(var, indices,
				name="gather_apply_sparse")

			spacial_grad = values[..., :-1]
			t_grad = -1 * values[..., -1:]

			ambient_grad = K.concatenate(\
				[spacial_grad, t_grad],
				axis=-1, )
			tangent_grad = project_onto_tangent_space(p,
				ambient_grad)

			exp_map = exponential_mapping(p,
				- self.lr * tangent_grad)

			return tf.scatter_update(ref=var,
				indices=indices, 
				updates=exp_map,
				name="scatter_update")

		else:
			# euclidean update using Adam optimizer
			return self.euclidean_optimizer.apply_gradients( 
				[(grad, var), ] 
			)

# class MyAdamOptimizer(optimizer.Optimizer):
# 	"""Optimizer that implements the Adam algorithm.
# 	See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
# 	([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
# 	"""

# 	def __init__(self,
# 		learning_rate=1e-3,
# 		beta1=0.9,
# 		beta2=0.999,
# 		epsilon=1e-8,
# 		use_locking=False,
# 		name="Adam"):
# 		r"""Construct a new Adam optimizer.
# 		Initialization:
# 		$$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
# 		$$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
# 		$$t := 0 \text{(Initialize timestep)}$$
# 		The update rule for `variable` with gradient `g` uses an optimization
# 		described at the end of section 2 of the paper:
# 		$$t := t + 1$$
# 		$$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
# 		$$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
# 		$$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
# 		$$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
# 		The default value of 1e-8 for epsilon might not be a good default in
# 		general. For example, when training an Inception network on ImageNet a
# 		current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
# 		formulation just before Section 2.1 of the Kingma and Ba paper rather than
# 		the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
# 		hat" in the paper.
# 		The sparse implementation of this algorithm (used when the gradient is an
# 		IndexedSlices object, typically because of `tf.gather` or an embedding
# 		lookup in the forward pass) does apply momentum to variable slices even if
# 		they were not used in the forward pass (meaning they have a gradient equal
# 		to zero). Momentum decay (beta1) is also applied to the entire momentum
# 		accumulator. This means that the sparse behavior is equivalent to the dense
# 		behavior (in contrast to some momentum implementations which ignore momentum
# 		unless a variable slice was actually used).
# 		Args:
# 			learning_rate: A Tensor or a floating point value.  The learning rate.
# 			beta1: A float value or a constant float tensor. The exponential decay
# 				rate for the 1st moment estimates.
# 			beta2: A float value or a constant float tensor. The exponential decay
# 				rate for the 2nd moment estimates.
# 			epsilon: A small constant for numerical stability. This epsilon is
# 				"epsilon hat" in the Kingma and Ba paper (in the formula just before
# 				Section 2.1), not the epsilon in Algorithm 1 of the paper.
# 			use_locking: If True use locks for update operations.
# 			name: Optional name for the operations created when applying gradients.
# 				Defaults to "Adam".  @compatibility(eager) When eager execution is
# 				enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
# 				callable that takes no arguments and returns the actual value to use.
# 				This can be useful for changing these values across different
# 				invocations of optimizer functions. @end_compatibility
# 		"""
# 		super(MyAdamOptimizer, self).__init__(use_locking, name)
# 		self._lr = learning_rate
# 		self._beta1 = beta1
# 		self._beta2 = beta2
# 		self._epsilon = epsilon

# 		# Tensor versions of the constructor arguments, created in _prepare().
# 		self._lr_t = None
# 		self._beta1_t = None
# 		self._beta2_t = None
# 		self._epsilon_t = None

# 	def _get_beta_accumulators(self):
# 		with ops.init_scope():
# 			if context.executing_eagerly():
# 				graph = None
# 			else:
# 				graph = ops.get_default_graph()
# 			return (self._get_non_slot_variable("beta1_power", graph=graph),
# 							self._get_non_slot_variable("beta2_power", graph=graph))

# 	def _create_slots(self, var_list):
# 		# Create the beta1 and beta2 accumulators on the same device as the first
# 		# variable. Sort the var_list to make sure this device is consistent across
# 		# workers (these need to go on the same PS, otherwise some updates are
# 		# silently ignored).
# 		first_var = min(var_list, key=lambda x: x.name)
# 		self._create_non_slot_variable(
# 				initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
# 		self._create_non_slot_variable(
# 				initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

# 		# Create slots for the first and second moments.
# 		for v in var_list:
# 			self._zeros_slot(v, "m", self._name)
# 			self._zeros_slot(v, "v", self._name)

# 	def _prepare(self):
# 		lr = self._call_if_callable(self._lr)
# 		beta1 = self._call_if_callable(self._beta1)
# 		beta2 = self._call_if_callable(self._beta2)
# 		epsilon = self._call_if_callable(self._epsilon)

# 		self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
# 		self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
# 		self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
# 		self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

# 	def _apply_dense(self, grad, var):
# 		assert False
# 		m = self.get_slot(var, "m")
# 		v = self.get_slot(var, "v")
# 		beta1_power, beta2_power = self._get_beta_accumulators()
# 		return training_ops.apply_adam(
# 				var,
# 				m,
# 				v,
# 				math_ops.cast(beta1_power, var.dtype.base_dtype),
# 				math_ops.cast(beta2_power, var.dtype.base_dtype),
# 				math_ops.cast(self._lr_t, var.dtype.base_dtype),
# 				math_ops.cast(self._beta1_t, var.dtype.base_dtype),
# 				math_ops.cast(self._beta2_t, var.dtype.base_dtype),
# 				math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
# 				grad,
# 				use_locking=self._use_locking).op

# 	def _resource_apply_dense(self, grad, var):
# 		assert False
# 		m = self.get_slot(var, "m")
# 		v = self.get_slot(var, "v")
# 		beta1_power, beta2_power = self._get_beta_accumulators()
# 		return training_ops.resource_apply_adam(
# 				var.handle,
# 				m.handle,
# 				v.handle,
# 				math_ops.cast(beta1_power, grad.dtype.base_dtype),
# 				math_ops.cast(beta2_power, grad.dtype.base_dtype),
# 				math_ops.cast(self._lr_t, grad.dtype.base_dtype),
# 				math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
# 				math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
# 				math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
# 				grad,
# 				use_locking=self._use_locking)

# 	def _apply_sparse_shared(self, grad, var, indices, 
# 		scatter_add):
# 		beta1_power, beta2_power = self._get_beta_accumulators()
# 		beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
# 		beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
# 		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
# 		beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
# 		beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
# 		epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
# 		lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

# 		grad = tf.verify_tensor_all_finite(grad, "fail in grad")

# 		# if "hyperbolic" in var.name:
# 		# 		grad = K.concatenate([grad[:,:-1], -grad[:,-1:]],
# 		# 			axis=-1)

# 		# m_t = beta1 * m + (1 - beta1) * g_t
# 		m = self.get_slot(var, "m")
# 		m_scaled_g_values = grad * (1 - beta1_t)
# 		m_t = state_ops.assign(m, m * beta1_t, 
# 			use_locking=self._use_locking)
# 		with ops.control_dependencies([m_t]):
# 			m_t = scatter_add(m, indices, m_scaled_g_values)

# 		# v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
# 		v = self.get_slot(var, "v")
# 		v_scaled_g_values = (grad * grad) * (1 - beta2_t)
# 		v_t = state_ops.assign(v, v * beta2_t, 
# 			use_locking=self._use_locking)
# 		with ops.control_dependencies([v_t]):
# 			v_t = scatter_add(v, indices, v_scaled_g_values)
# 		v_sqrt = math_ops.sqrt(K.maximum(v_t, 0.))

# 		if "hyperbolic" in var.name:

# 			m_t = tf.verify_tensor_all_finite(m_t, "fail in m_t")
# 			v_sqrt = tf.verify_tensor_all_finite(v_sqrt, 
# 				"fail in v_sqrt")

# 			gr = m_t / (v_sqrt + epsilon_t)

# 			gr = tf.verify_tensor_all_finite(gr, "fail in gr")

# 			gr = K.concatenate(
# 				[gr[...,:-1], -gr[...,-1:]],
# 				axis=-1)
# 			gr_tangent = project_onto_tangent_space(var, gr)

# 			gr_tangent = tf.verify_tensor_all_finite(gr_tangent, 
# 				"fail in tangent")

# 			exp_map = exponential_mapping(var, -lr * gr_tangent)

# 			exp_map = tf.verify_tensor_all_finite(exp_map, 
# 				"fail in exp_map")

# 			var_update = state_ops.assign(
# 				var, 
# 				exp_map, 
# 				use_locking=self._use_locking)
# 		else:
# 			var_update = state_ops.assign_sub(
# 				var, 
# 				lr * m_t / (v_sqrt + epsilon_t), 
# 				use_locking=self._use_locking)
# 		return control_flow_ops.group(*[var_update, m_t, v_t])

# 	def _apply_sparse(self, grad, var):
# 		return self._apply_sparse_shared(
# 				grad.values,
# 				var,
# 				grad.indices,
# 				lambda x, i, v: state_ops.scatter_add(
# 						x,
# 						i,
# 						v,
# 						use_locking=self._use_locking))

# 	def _resource_scatter_add(self, x, i, v):
# 		with ops.control_dependencies(
# 				[resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
# 			return x.value()

# 	def _resource_apply_sparse(self, grad, var, indices):
# 		return self._apply_sparse_shared(grad, var, indices,
# 																		 self._resource_scatter_add)

# 	def _finish(self, update_ops, name_scope):
# 		# Update the power accumulators.
# 		with ops.control_dependencies(update_ops):
# 				beta1_power, beta2_power = self._get_beta_accumulators()
# 		with ops.colocate_with(beta1_power):
# 				update_beta1 = beta1_power.assign(
# 						beta1_power * self._beta1_t, use_locking=self._use_locking)
# 				update_beta2 = beta2_power.assign(
# 						beta2_power * self._beta2_t, use_locking=self._use_locking)
# 		return control_flow_ops.group(
# 				*update_ops + [update_beta1, update_beta2], name=name_scope)