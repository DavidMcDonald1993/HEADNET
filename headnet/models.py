import keras.backend as K
import tensorflow as tf 
from keras.layers import Input, Dense, Activation, Lambda, Concatenate
from keras.models import Model
from keras.initializers  import RandomUniform
from keras import regularizers

from headnet.losses import asym_hyperbolic_loss
from headnet.optimizers import ExponentialMappingOptimizer
from headnet.hyperboloid_layers import logarithmic_map, parallel_transport, exp_map_0
from headnet.hyperboloid_layers import HyperboloidFeedForwardLayer

reg = 1e-3
# m = 1e-3

# def minkowski_dot(x, y):
# 	assert len(x.shape) == len(y.shape)
# 	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

# def parallel_transport_hyperboloid(p, q, x):
# 	alpha = -minkowski_dot(p, q)

# 	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
# 		K.maximum(alpha + 1, K.epsilon())

# def logarithmic_map_hyperboloid(p, x):
# 	assert len(p.shape) == len(x.shape)

# 	alpha = -minkowski_dot(p, x)

# 	alpha = K.maximum(alpha, 1 + K.epsilon())

# 	return tf.acosh(alpha) * (x - alpha * p) / \
# 		K.maximum(K.sqrt(K.maximum(alpha ** 2 - 1., 0.)), K.epsilon())

def map_to_tangent_space_mu_zero(mus):

	source_embedding = mus[:,:1]
	target_embedding = mus[:,1:]

	to_tangent_space = logarithmic_map(source_embedding,
		target_embedding)

	mu_zero = K.concatenate([
		K.zeros_like(source_embedding[..., :-1]), 
		K.ones_like(source_embedding[...,-1:])], axis=-1)
	to_tangent_space_mu_zero = parallel_transport(source_embedding,
		mu_zero,
		to_tangent_space)
	# # ignore 0 t coordinate
	to_tangent_space_mu_zero = to_tangent_space_mu_zero[..., :-1]

	return to_tangent_space_mu_zero
	
def kullback_leibler_divergence(args):

	mus, sigmas = args

	k = K.int_shape(mus)[-1]

	sigmas = K.maximum(sigmas, K.epsilon())

	source_sigma = sigmas[:,:1]
	target_sigma = sigmas[:,1:]

	sigma_ratio = target_sigma / source_sigma

	trace_fac = K.sum(sigma_ratio,
		axis=-1, keepdims=True)

	mu_sq_diff = K.sum(mus ** 2 / \
		source_sigma,
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = K.sum(K.log(sigma_ratio), axis=-1, keepdims=True)

	return K.squeeze(0.5 * (trace_fac + 
			mu_sq_diff - k - log_det), axis=-1)

def build_hyperboloid_asym_model(num_attributes, 
	embedding_dim, 
	num_negative_samples, 
	num_hidden=256,
	lr=1e-1):

	input_transform = Dense(num_hidden,
		# activation=exp_map_0,
		activation="elu",
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		# kernel_initializer=RandomUniform(-m, m),
		trainable=True,
		name="euclidean_transform",
	)

	# to_hyperboloid_1 = Lambda(lambda x: 
	# 	exp_map_0(
	# 		tf.pad(tf.verify_tensor_all_finite(x,"fail in map1"), 
	# 			tf.constant([[0, 0]]*(len(x.shape)-1) + [[0, 1]]))
	# ))

	# hyperboloid_embedding_layer = HyperboloidFeedForwardLayer(
	# 	embedding_dim, 
	# 	# activation="tanh",
	# 	name="hyperbolic_feedforward"
	# )
	
	hyperboloid_embedding_layer = Dense(
		embedding_dim, 
		# activation=exp_map_0,
		# kernel_initializer=RandomUniform(-m, m),
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="dense_to_hyperboloid",
		trainable=True
	)

	to_hyperboloid = Lambda(lambda x: 
		# tf.verify_tensor_all_finite(
			exp_map_0(
			tf.pad(
				# tf.verify_tensor_all_finite(
					x
				# , "fail in map2")
				, 
				tf.constant([[0, 0]]*(len(x.shape)-1) + [[0, 1]]))),
				# ,"fail after exp map")
		name="to_hyperboloid"
	)

	# hyperboloid_shift = HyperboloidFeedForwardLayer(embedding_dim)

	sigma_layer = Dense(embedding_dim, 
		activation=lambda x: K.elu(x) + 1.,
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		# kernel_initializer=RandomUniform(-m, m),
		trainable=True,
		name="dense_to_sigma"
	)

	embedder_input = Input(( num_attributes, ),
		dtype=K.floatx(),
		name="embedder_input")

	embedder_hyperboloid = to_hyperboloid(
		hyperboloid_embedding_layer(
		# to_hyperboloid_1(
			input_transform(embedder_input)
		# )
	))

	embedder_sigmas = sigma_layer(
		input_transform(embedder_input)
	)

	embedder_model = Model(embedder_input, 
		[embedder_hyperboloid, embedder_sigmas],
		name="embedder_model")

	trainable_model_input = Input((1 + 1, 
		num_attributes))

	trainable_hyperboloid = to_hyperboloid(
		hyperboloid_embedding_layer(
		# to_hyperboloid_1(
		input_transform(trainable_model_input)
		# )
	))

	trainable_sigmas = sigma_layer(
		input_transform(trainable_model_input)
	)

	mus = Lambda(map_to_tangent_space_mu_zero,
		name="to_tangent_space_mu_zero")(trainable_hyperboloid)

	kds = Lambda(kullback_leibler_divergence,
		name="kullback_leibler_layer")([mus, trainable_sigmas])

	trainable_model = Model(trainable_model_input, 
		kds,
		name="trainable_model")

	# optimizer = ExponentialMappingOptimizer(lr=lr)
	optimizer = "adam"

	trainable_model.compile(optimizer=optimizer, 
		loss=asym_hyperbolic_loss(num_negative_samples),
		target_tensors=[ tf.placeholder(dtype=tf.int64), ])

	return embedder_model, trainable_model