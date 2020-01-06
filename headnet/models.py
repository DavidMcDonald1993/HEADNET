import keras.backend as K
import tensorflow as tf 
from keras.layers import Input, Dense, Activation, Lambda, Reshape
from keras.models import Model
from keras.initializers  import RandomUniform
from keras import regularizers
from tensorflow.train import AdamOptimizer


from headnet.losses import asym_hyperbolic_loss
from headnet.optimizers import ExponentialMappingOptimizer
from headnet.hyperboloid_layers import logarithmic_map, parallel_transport, exp_map_0
from headnet.hyperboloid_layers import HyperboloidFeedForwardLayer

reg = 0e-3
initializer="glorot_uniform"#RandomUniform(-1e-5, 1e-5)

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
		axis=-1, 
		keepdims=True)

	mu_sq_diff = K.sum(mus ** 2 / \
		source_sigma,
		axis=-1, 
		keepdims=True) # assume sigma inv is diagonal

	log_det = K.sum(K.log(sigma_ratio), 
		axis=-1, 
		keepdims=True)

	return K.squeeze(0.5 * (trace_fac + 
			mu_sq_diff - k - log_det), axis=-1)

def build_hyperboloid_asym_model(num_attributes, 
	embedding_dim, 
	num_negative_samples, 
	num_hidden=128,
	lr=1e-1):

	input_layer = Input((num_attributes,),
		name="input_layer")

	input_transform = Dense(
		num_hidden,
		activation="tanh",
		kernel_initializer=initializer,
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="euclidean_transform",
	)(input_layer)

	hyperboloid_embedding_layer = Dense(
		embedding_dim, 
		kernel_initializer=initializer,
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="dense_to_hyperboloid",
	)(input_transform)

	to_hyperboloid = Lambda(lambda x: 
			exp_map_0(tf.pad(x,
				tf.constant(
					[[0, 0]]*(len(x.shape)-1) + [[0, 1]]))),
		name="to_hyperboloid"
	)(hyperboloid_embedding_layer)

	sigma_layer = Dense(
		embedding_dim, 
		activation=lambda x: K.elu(x) + 1.,
		kernel_initializer=initializer,
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="dense_to_sigma"
	)(input_transform)

	# embedder_input = Input(( num_attributes, ),
	# 	dtype=K.floatx(),
	# 	name="embedder_input")

	# embedder_hyperboloid = to_hyperboloid(
	# 	hyperboloid_embedding_layer(
	# 		input_transform(embedder_input)
	# ))

	# embedder_sigmas = sigma_layer(
	# 	input_transform(embedder_input)
	# )

	embedder_model = Model(input_layer, 
		[to_hyperboloid, sigma_layer],
		name="embedder_model")

	mus, sigmas = embedder_model(input_layer)

	reshape = Lambda(lambda x: 
		K.reshape(x, (-1, 2, K.int_shape(x)[-1])),
		name="reshape")
	
	mus = reshape(mus)
	sigmas = reshape(sigmas) 

	# mus = Reshape((-1, 2, embedding_dim+1))(mus)

	# print (mus.shape)
	# raise SystemExit

	# trainable_model_input = Input((1 + 1 + 0, 
	# 	num_attributes),
	# 	name="trainable_model_input")

	# trainable_hyperboloid = to_hyperboloid(
	# 	hyperboloid_embedding_layer(
	# 	input_transform(trainable_model_input)
	# ))

	# trainable_sigmas = sigma_layer(
	# 	input_transform(trainable_model_input)
	# )

	mus = Lambda(map_to_tangent_space_mu_zero,
		name="to_tangent_space_mu_zero")(mus)

	kds = Lambda(kullback_leibler_divergence,
		name="kullback_leibler_layer")([mus, sigmas])

	trainable_model = Model(input_layer, 
		kds,
		name="trainable_model")

	optimizer = AdamOptimizer()

	trainable_model.compile(optimizer=optimizer, 
		loss=asym_hyperbolic_loss(num_negative_samples),
		target_tensors=[ tf.placeholder(dtype=tf.int64), ])

	return embedder_model, trainable_model