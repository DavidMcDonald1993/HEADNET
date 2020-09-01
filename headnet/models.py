import keras.backend as K
import tensorflow as tf 
from keras.layers import Input, Dense, Lambda, Embedding
from keras.models import Model
from keras.initializers  import RandomUniform
from keras import regularizers
from tensorflow.train import AdamOptimizer, GradientDescentOptimizer, RMSPropOptimizer

from headnet.losses import asym_hyperbolic_loss
from headnet.hyperboloid_layers import logarithmic_map, parallel_transport, exp_map_0

reg = 0e-4
initializer = RandomUniform(-1e-3, 1e-3)

def map_to_tangent_space_mu_zero(mus):

	assert len(mus.shape) == 4

	source_embedding = mus[:,:,:1]
	target_embedding = mus[:,:,1:]

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

	assert len(mus.shape) == len(sigmas.shape) == 4

	k = K.int_shape(mus)[-1]

	sigmas = K.maximum(sigmas, K.epsilon())

	source_sigma = sigmas[:,:,:1]
	target_sigma = sigmas[:,:,1:]

	sigma_ratio = target_sigma / source_sigma
	sigma_ratio = K.maximum(sigma_ratio, K.epsilon())

	trace_fac = K.sum(sigma_ratio,
		axis=-1, 
		keepdims=True)

	mu_sq_diff = K.sum(K.square(mus) / \
		source_sigma,
		axis=-1, 
		keepdims=True) # assume sigma inv is diagonal

	log_det = K.sum(K.log(sigma_ratio), 
		axis=-1, 
		keepdims=True)

	kld = 0.5 * (trace_fac + 
			mu_sq_diff - k - log_det)

	kld = K.squeeze(kld, axis=-1)
	return K.squeeze(kld, axis=-1)

def build_headnet(
	N,
	features, 
	embedding_dim, 
	num_negative_samples, 
	num_hidden=128,
	identity_variance=False):

	if features is not None: # HEADNet with attributes

		print("training using attributes")

		input_layer = Input((features.shape[1],),
			name="attributed_input_layer")

		input_transform = Dense(
			num_hidden,
			activation="relu",
			kernel_initializer=initializer,
			kernel_regularizer=regularizers.l2(reg),
			bias_regularizer=regularizers.l2(reg),
			name="euclidean_transform",
		)(input_layer)

	else:

		print("training without using attributes")

		input_layer = Input((1,), 
			name="unattributed_input_layer")
		input_transform = Embedding(N,
			num_hidden)(input_layer)


	hyperboloid_embedding_layer = Dense(
		embedding_dim, 
		kernel_initializer=initializer,
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="dense_to_hyperboloid",
	)(input_transform)

	to_hyperboloid = Lambda(
		exp_map_0,
		name="to_hyperboloid"
	)(hyperboloid_embedding_layer)

	sigma_layer = Dense(
		embedding_dim, 
		activation=lambda x: K.elu(x) + 1.,
		kernel_initializer="zeros",
		kernel_regularizer=regularizers.l2(reg),
		bias_regularizer=regularizers.l2(reg),
		name="dense_to_sigma",
		trainable=~identity_variance,
	)(input_transform)
	if  identity_variance:
		sigma_layer = Lambda(K.stop_gradient,
			name="variance_stop_gradient")(sigma_layer)

	embedder_model = Model(input_layer, 
		[to_hyperboloid, sigma_layer],
		name="embedder_model")

	if features is not None:

		trainable_input = Input(
			(1 + num_negative_samples, 2, features.shape[1], ),
			name="trainable_input_attributed")
	else:

		trainable_input = Input(
			(1 + num_negative_samples, 2, ),
			name="trainable_input_non_attributed")

	mus, sigmas = embedder_model(trainable_input)

	assert len(mus.shape) == len(sigmas.shape) == 4

	mus = Lambda(map_to_tangent_space_mu_zero,
		name="to_tangent_space_mu_zero")(mus)

	kds = Lambda(kullback_leibler_divergence,
		name="kullback_leibler_layer")([mus, sigmas])

	trainable_model = Model(
		trainable_input,
		kds,
		name="trainable_model")

	optimizer = AdamOptimizer(1e-3)

	trainable_model.compile(optimizer=optimizer, 
		loss=asym_hyperbolic_loss,
		target_tensors=[ tf.placeholder(dtype=tf.int64, 
			shape=(None, 1)),])

	return embedder_model, trainable_model