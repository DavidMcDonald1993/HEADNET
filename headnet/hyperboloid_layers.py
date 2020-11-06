import keras.backend as K

import tensorflow as tf
from tensorflow.python.framework import function

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return K.sum(x[...,:-1] * y[...,:-1], 
		axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def parallel_transport(p, q, x):
	# p goes to q
	alpha = -minkowski_dot(p, q)

	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
		K.maximum(alpha + 1, K.epsilon())

def parallel_transport_to_mu_0(p, x):
	# p goes to mu_0
	alpha = p[...,-1:]

	# minkowski_dot(q - alpha * p, x)
	# md = minkowski_dot( - alpha * p + , x)

	raise NotImplementedError

@function.Defun(K.floatx(), K.floatx())
def norm_grad(x, dy):
	return dy*(x/(tf.norm(x, ord="euclidean",
		axis=-1, keepdims=True) + K.epsilon() ))

@function.Defun(K.floatx(), 
	grad_func=norm_grad, 
	shape_func=lambda op: \
		[op.inputs[0].get_shape().as_list()[:-1] + [1]])
def norm(x, ):
	return tf.norm(x, ord="euclidean",
		axis=-1, keepdims=True)


def exp_map_0(x):
	r = norm(x) 
	r = K.maximum(r, K.epsilon())

	# unit norm
	x = x / r

	x = tf.sinh(r) * x
	t = tf.cosh(r)

	return K.concatenate([x, t], axis=-1)

def logarithmic_map(p, x):
	assert len(p.shape) == len(x.shape)

	alpha = -minkowski_dot(p, x)

	alpha = K.maximum(alpha, 1 + K.epsilon())

	return tf.acosh(alpha) * (x - alpha * p) / \
		K.maximum(K.sqrt(K.maximum(alpha ** 2 - 1., 0.)),
			K.epsilon())
