import tensorflow as tf 
import keras.backend as K

def asym_hyperbolic_loss(num_negative_samples=10):

	def loss(y_true, y_pred):

		y_pred = K.reshape(y_pred, [-1, num_negative_samples+1])

		return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
				labels=y_true[:K.shape(y_pred)[0], 0], 
				logits=-y_pred))

	return loss 
