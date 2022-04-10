import tensorflow as tf 
import keras.backend as K

def negative_sampling_softmax_loss(y_true, y_pred):

	# inverse y_pred since it is a metric (larger distance -> smaller probability)
	y_pred = -y_pred

	return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
			labels=y_true[:, 0], 
			logits=y_pred))
