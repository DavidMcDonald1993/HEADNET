from __future__ import print_function

import re
import os
import fcntl
import functools
import numpy as np
import networkx as nx

import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd

import pickle as pkl

from multiprocessing.pool import Pool 

import matplotlib.pyplot as plt

from collections import Counter

from scipy.sparse import identity, csr_matrix, load_npz

import glob


def load_data(args):

	edgelist_filename = args.edgelist
	features_filename = args.features
	labels_filename = args.labels

	graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	zero_weight_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w == 0.]
	print ("removing", len(zero_weight_edges), "edges with 0. weight")
	graph.remove_edges_from(zero_weight_edges)

	print ("ensuring all weights are positive")
	nx.set_edge_attributes(graph, name="weight", values={edge: abs(weight) 
		for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

	print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv") or features_filename.endswith(".csv.gz"):
			features = pd.read_csv(features_filename, index_col=0, sep=",")
			features = [features.reindex(sorted(graph)).values, 
				features.reindex(sorted(features.index)).values]
			print ("no scaling applied")
			# scaler = StandardScaler()
			# scaler.fit(features[0])
			# features = list(map(scaler.transform,
			# 	features))
			features = tuple(map(csr_matrix, features))

		elif features_filename.endswith(".npz"):
			
			features = load_npz(features_filename)
			assert isinstance(features, csr_matrix)

			features = (features[sorted(graph)], features)

		else:
			raise Exception


		print ("training features shape is {}".format(features[0].shape))
		print ("all features shape is {}\n".format(features[1].shape))

	else: 
		features = None

	if labels_filename is not None:

		print ("loading labels from {}".format(labels_filename))

		if labels_filename.endswith(".csv") or labels_filename.endswith(".csv.gz"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(graph.nodes())).values.astype(int)#.flatten()
			assert len(labels.shape) == 2
		elif labels_filename.endswith(".pkl"):
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] 
				for n in sorted(graph.nodes())], dtype=np.int)
		else:
			raise Exception

		print ("labels shape is {}\n".format(labels.shape))

	else:
		labels = None

	graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

	return graph, features, labels

def load_embedding(embedding_filename):
	assert embedding_filename.endswith(".csv.gz")
	embedding_df = pd.read_csv(embedding_filename, index_col=0)
	return embedding_df

def load_weights(model, embedding_directory):

	# previous_models = sorted(glob.glob(
	# 	os.path.join(args.embedding_path, "*.h5")))
	previous_models = sorted(filter(
		re.compile("[0-9]+\_model\.h5").match, 
		os.listdir(embedding_directory)
	))
	if len(previous_models) > 0:
		weight_file = previous_models[-1]
		initial_epoch = int(weight_file.split("_")[0])
		print ("previous models found in directory -- loading from file {} and resuming from epoch {}".format(weight_file, initial_epoch))
		model.load_weights(
			os.path.join(embedding_directory, 
			weight_file))
	else:
		print ("no previous model found in {}".\
			format(embedding_directory))
		initial_epoch = 0

	return model, initial_epoch

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + np.sum(np.square(X), axis=-1, keepdims=True)
	x = np.concatenate([x, t], axis=-1)
	return 1 / (1. - np.sum(np.square(X), axis=-1, keepdims=True)) * x

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return (np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) 
		- x[...,-1:] * y[...,-1:])

def minkowski_norm(x):
	return np.sqrt( np.maximum(minkowski_dot(x, x), 0.) )

def determine_positive_and_negative_samples(graph, args):

	def build_positive_samples(graph, k=3):
		assert k > 0

		def step(X):
			X[X>0] = 1    
			X[X<0] = 0
			return X
		
		N = len(graph)
		A0 = identity(N, dtype=int)
		print ("determined 0 hop neighbours")
		A1 = step(nx.adjacency_matrix(graph, nodelist=sorted(graph)) - A0)
		print ("determined 1 hop neighbours")
		positive_samples = [A0, A1]
		for i in range(2, k+1):
			A_k = step(step(positive_samples[-1].dot(A1)) - step(np.sum(positive_samples, axis=0)))
			print ("determined", i, "hop neighbours")
			positive_samples.append(A_k)
		return positive_samples

	def positive_samples_to_list(positive_samples):
		l = []
		for k, ps in enumerate(positive_samples):
			if k == 0:
				continue
			nzx, nzy = np.nonzero(ps)
			l.append(np.array((nzx, nzy, [k]*len(nzx))))
		return np.concatenate(l, axis=1).T

	def build_negative_samples(positive_samples):
		
		N = positive_samples[0].shape[0]
		negative_samples = []

		for k in range(len(positive_samples)):
			if True or k == args.context_size:
				neg_samples = np.ones((N, N), dtype=bool )
				neg_samples[
					np.sum(positive_samples[:k+1], axis=0).nonzero()
				] = 0
				assert np.allclose(neg_samples.diagonal(), 0)
			else:
				assert False
				neg_samples = np.zeros((N, N))
				neg_samples[np.sum(positive_samples[k+1:], 
					axis=0).nonzero()] = 1
			neg_samples = neg_samples.flatten()
			neg_samples /= neg_samples.sum(axis=-1, keepdims=True)
			neg_samples = neg_samples.cumsum(axis=-1)
			assert np.allclose(neg_samples[..., -1], 1)
			neg_samples[np.abs(neg_samples - neg_samples.max(axis=-1, keepdims=True)) < 1e-15] = 1 
			negative_samples.append(neg_samples)
		
		return negative_samples

	positive_samples = build_positive_samples(graph, 
		k=args.context_size)

	negative_samples = build_negative_samples(positive_samples)

	positive_samples = positive_samples_to_list(positive_samples)

	# assert np.all(positive_samples[:,-1] < args.context_size)

	print ("found {} positive sample pairs".format(
			len(positive_samples)))
	
	return positive_samples, negative_samples

