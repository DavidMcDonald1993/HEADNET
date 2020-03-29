from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence

import itertools

from sklearn.preprocessing import StandardScaler

from collections import Counter

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
		self.num_positive_samples = len(positive_samples)
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]
		self.negative_samples = negative_samples
		self.node_map = node_map
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples

		print ("Built generator")

		# nodes = set(positive_samples.flatten())
		# self.missing = set(range(2995)) - nodes

		# for u in self.missing:
		# 	assert negative_samples[u] == 0

		# self.negative_samples = self.negative_samples.cumsum()

	def __len__(self):
		return max(1000, 
			int(np.ceil(self.num_positive_samples / \
				self.batch_size)))

	def __getitem__(self, batch_idx):
	
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		num_positive_samples = self.num_positive_samples
		node_map = self.node_map

		idx = np.random.choice(num_positive_samples, 
			size=batch_size)
		batch_positive_samples = positive_samples[idx]

		batch_negative_samples = np.searchsorted(negative_samples,
			np.random.rand(batch_size * num_negative_samples, 2))

		batch_positive_samples = np.expand_dims(
			batch_positive_samples, axis=1)
		batch_negative_samples = batch_negative_samples.reshape(
			batch_size, num_negative_samples, 2)

		if node_map is not None:
			batch_negative_samples = node_map[batch_negative_samples]

		training_sample = np.concatenate(
			(batch_positive_samples, batch_negative_samples), 
			axis=1)#.reshape(batch_size*(num_negative_samples + 1), 2)

		training_sample = training_sample.flatten()

		# for n in training_sample:
		# 	assert n not in self.missing, n

		training_sample = self.features[training_sample].A

		target = np.zeros((training_sample.shape[0], 1))

		return training_sample, target

	# def on_epoch_end(self):
	# 	positive_samples = self.positive_samples
	# 	idx = np.random.permutation(self.num_positive_samples)
	# 	self.positive_samples = positive_samples[idx]