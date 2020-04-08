from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence
from keras import backend as K

import itertools

from sklearn.preprocessing import StandardScaler

from collections import Counter

import threading

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
		# return max(1000, 
		# 	int(np.ceil(self.num_positive_samples / \
		# 		self.batch_size)))

	def __getitem__(self, batch_idx):

		# print ("{}: {}".format(batch_idx, 
		# 	threading.current_thread()))

		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		num_positive_samples = self.num_positive_samples
		node_map = self.node_map

		# idx = np.random.choice(num_positive_samples, 
		# 	size=batch_size)
		# batch_positive_samples = positive_samples[idx]

		batch_positive_samples = positive_samples[
			batch_idx * batch_size : \
			(batch_idx+1) * batch_size
		]
		batch_size = batch_positive_samples.shape[0]

		batch_negative_samples = np.searchsorted(negative_samples,
			np.random.rand(batch_size, num_negative_samples, 2))		
				
		if node_map is not None:
			batch_negative_samples = node_map[batch_negative_samples]

		batch_positive_samples = np.expand_dims(
			batch_positive_samples, axis=1)

		training_sample = np.concatenate(
			(batch_positive_samples, batch_negative_samples),
			axis=1)

		# shape = list(training_sample.shape)
		training_sample = training_sample.flatten()
		training_sample = self.features[training_sample].A
		training_sample = training_sample.reshape(
			batch_size, 1 + num_negative_samples, 2, -1)

		target = np.zeros((batch_size, 1), dtype=int,)

		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]