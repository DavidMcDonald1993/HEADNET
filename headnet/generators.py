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
		model,
		args,
		graph
		):
		self.features = features
		assert isinstance(positive_samples, np.ndarray)
		assert isinstance(negative_samples, list)
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		self.negative_samples = negative_samples
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.N = len(graph)
		self.context_size = args.context_size
		

		print ("Built generator")

	def __len__(self):
		return 1000
		# return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
	
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		N = self.N

		# idx = np.arange(batch_idx * batch_size, 
		# 	min(len(positive_samples), 
		# 	(batch_idx + 1) * batch_size))
		# batch_positive_samples = positive_samples[
		# 	idx]
		
		# idx = random.choices(range(len(positive_samples)), 
		# 	k=batch_size)
		idx = np.random.choice(num_positive_samples, 
			size=batch_size)
		batch_positive_samples = positive_samples[idx]


		###############################################

		# batch_negative_samples = np.array([
		# 	np.searchsorted(negative_samples[k][u], [
		# 		random.random() for _ in range(num_negative_samples)
		# 	])
		# 	for u, _, k in batch_positive_samples
		# ])

		# training_sample = np.concatenate(
		# 	[batch_positive_samples[:, :-1], batch_negative_samples],
		# 	axis=-1
		# )

		# # for u, v, neg_samples in zip(training_sample[:,0],
		# 	# training_sample[:,1], training_sample[:,2:]):
		# 	# for w in neg_samples:
		# 		# assert self.sps[u, v] < self.sps[u, w ]

		# training_sample = self.features[training_sample]
		# target = np.zeros((len(training_sample), 1))
		# return training_sample, target

		##################

		# idx_ = batch_positive_samples[:,-1].argsort()
		# batch_positive_samples = batch_positive_samples[idx_]

		# batch_negative_samples = np.concatenate([
		# 	np.unravel_index(np.searchsorted(negative_samples[k], 
		# 	[random.random() 
		# 		for _ in range(count * num_negative_samples)]),
		# 		shape=(N, N))
		# 	# np.unravel_index(np.searchsorted(negative_samples[k], 
		# 		# np.random.rand(count*num_negative_samples)),
		# 		# dims=(N, N))
		# 	for k, count in Counter(batch_positive_samples[:,-1]).items()
		# ], axis=1).T


		#######################

		assert self.context_size == 1


		# batch_negative_samples = np.concatenate([
		# 	np.unravel_index(np.searchsorted(
		# 		negative_samples[1], 
		# 		np.random.uniform(
		# 			size=len(batch_positive_samples)*num_negative_samples)),
		# 		# [random.random() 
		# 		# 	for _ in range(batch_size * num_negative_samples)]),
		# 		shape=(N, N))
		# ], axis=1).T

		# training_sample = np.empty(
		# 	(len(batch_positive_samples) + \
		# 		len(batch_negative_samples), 2), dtype=int)
		# training_sample[::num_negative_samples + 1] = \
		# 	batch_positive_samples[:,:-1]

		# for i in range(num_negative_samples):
		# 	training_sample[i+1::num_negative_samples+1] = \
		# 		batch_negative_samples[i::num_negative_samples]


		# batch_negative_samples = np.column_stack(
		# 	np.unravel_index(np.searchsorted(negative_samples[1], 
		# 		np.random.rand(batch_size * num_negative_samples)),
		# 		shape=(N, N)), )

		batch_negative_samples = np.searchsorted(negative_samples[1],
			np.random.rand(batch_size * num_negative_samples, 2))

		batch_positive_samples = np.expand_dims(
			batch_positive_samples[:,:-1], axis=1)
		batch_negative_samples = batch_negative_samples.reshape(
			batch_size, num_negative_samples, 2)

		training_sample = np.concatenate(
			(batch_positive_samples, batch_negative_samples), 
			axis=1).reshape(batch_size*(num_negative_samples + 1), 2)


		# assert np.all(training_sample >= 0)

		# for i in range(batch_size):
		# 	arr = training_sample[
		# 		i*(num_negative_samples+1):(1+i)*(num_negative_samples+1)]
		# 	for row in arr[1:]:
		# 		assert self.sps[arr[0,0], arr[0,1]] < self.sps[row[0], row[1]]
		# 		print (self.sps[arr[0,0], arr[0,1]] , self.sps[row[0], row[1]] )

		# raise SystemExit

		training_sample = training_sample.flatten()

		training_sample = self.features[training_sample].A

		target = np.zeros((training_sample.shape[0], 1))

		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]

		# import matplotlib.pyplot as plt
		# plt.hist(self.counts)
		# plt.show()

		pass