import tensorflow as tf 
import keras.backend as K

def asym_hyperbolic_loss(y_true, y_pred):
	#num_negative_samples=10):

	# def loss(y_true, y_pred):

	return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
			labels=y_true[:, 0], 
			logits=-y_pred))

	# return loss 
