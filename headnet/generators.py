
import numpy as np

from keras.utils import Sequence

class TrainingDataGenerator(Sequence):

	def __init__(self,
		features,
		positive_samples,
		negative_samples,
		node_map,
		args,
		):
		self.features = features
		assert isinstance(positive_samples, np.ndarray)
		assert isinstance(negative_samples, np.ndarray)
		self.num_positive_samples = positive_samples.shape[0]
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]
		self.negative_samples = negative_samples
		self.node_map = node_map
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples

		print ("Built generator")

	def __len__(self):
		return int(np.ceil(self.num_positive_samples / \
				self.batch_size))

	def __getitem__(self, batch_idx):

		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		node_map = self.node_map

		# select positive samples from list
		batch_positive_samples = positive_samples[
			batch_idx * batch_size : \
			(batch_idx+1) * batch_size
		]
		# update batch size (may be less than self.batch size if at the end of an epoch)
		batch_size = batch_positive_samples.shape[0]

		# select negative samples (batch_size * num_negative_samples pairs)
		batch_negative_samples = np.searchsorted(negative_samples,
			np.random.rand(batch_size, num_negative_samples, 2))		


		if node_map is not None:
			batch_negative_samples = node_map[batch_negative_samples]

		batch_positive_samples = np.expand_dims(
			batch_positive_samples, axis=1)

		training_sample = np.concatenate(
			(batch_positive_samples, batch_negative_samples),
			axis=1)

		# training_sample_shape is (batch_size, 1+num_negative_samples, 2)
		# for a total of batch_size*(1+num_negative_samples) pairs

		# select features using training sample
		if self.features is not None:
			training_sample = training_sample.flatten()
			training_sample = self.features[training_sample].A
			training_sample = training_sample.reshape(
				batch_size, 1 + num_negative_samples, 2, -1)

		# index of positive sample (always 0)
		target = np.zeros((batch_size, 1), dtype=int,)

		# training_sample shape is
		# (batch_size, 1 + num_negative_samples, 2, feature_dim) 

		return training_sample, target

	def on_epoch_end(self):
		# shuffle self.positive_samples
		positive_samples = self.positive_samples
		# get random permutation of indexes [0, self.num_positive_samples-1]
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]