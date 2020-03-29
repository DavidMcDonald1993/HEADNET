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

	graph_filename = args.graph
	features_filename = args.features
	labels_filename = args.labels

	print ("loading graph from", graph_filename)

	if graph_filename.endswith(".npz"):
		
		graph = load_npz(graph_filename)

	elif graph_filename.endswith(".tsv") or graph_filename.endswith(".tsv.gz"):

		graph = nx.read_weighted_edgelist(graph_filename, 
			delimiter="\t", nodetype=int,
			create_using=nx.DiGraph() 
				if args.directed else nx.Graph())

		zero_weight_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w == 0.]
		print ("removing", len(zero_weight_edges), "edges with 0. weight")
		graph.remove_edges_from(zero_weight_edges)

		print ("ensuring all weights are positive")
		nx.set_edge_attributes(graph, name="weight", 
			values={edge: np.abs(np.int8(weight)) 
			for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

		print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

		# graph = nx.convert_node_labels_to_integers(graph, 
		# 	ordering="sorted")

	else:
		raise NotImplementedError


	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv") or features_filename.endswith(".csv.gz"):
			features = pd.read_csv(features_filename, 
				index_col=0, sep=",")
			feattures = features.reindex(sorted(features.index)).values
			# features = [features.reindex(sorted(graph)).values, 
			# 	features.reindex(sorted(features.index)).values]
			print ("no scaling applied")
			# features = tuple(map(csr_matrix, features))
			features = csr_matrix(features)

		elif features_filename.endswith(".npz"):
			
			features = load_npz(features_filename)
			assert isinstance(features, csr_matrix)
			# features = (features[sorted(graph)], features)

		else:
			raise NotImplementedError

		# print ("training features shape is {}".format(
		# 	features[0].shape))
		# print ("all features shape is {}\n".format(
		# 	features[1].shape))

	else: 
		features = None

	if labels_filename is not None:

		print ("loading labels from {}".format(labels_filename))

		if labels_filename.endswith(".csv") or labels_filename.endswith(".csv.gz"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(labels.index))\
				.values.astype(np.int)
			assert len(labels.shape) == 2
		elif labels_filename.endswith(".pkl"):
			raise Exception
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] 
				for n in sorted(graph)], dtype=np.int)
		else:
			raise Exception

		print ("labels shape is {}\n".format(labels.shape))

	else:
		labels = None


	return graph, features, labels

def load_embedding(embedding_filename):
	assert embedding_filename.endswith(".csv.gz")
	embedding_df = pd.read_csv(embedding_filename, index_col=0)
	return embedding_df

def load_weights(model, embedding_directory):
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

	assert args.context_size == 1

	if isinstance(graph, csr_matrix):
		print ("graph is sparse adj matrix")
		positive_samples = np.array(list(zip(*graph.nonzero())))
		neg_samples = graph.sum(0 ).A.flatten() + graph.sum(1).A.flatten()
		neg_samples = neg_samples ** .75
		node_map = None
	else:
		print ("graph is edgelist")
		sorted_graph = sorted(graph)
		positive_samples = np.array(list(graph.edges()))
		neg_samples = np.array(
			[graph.degree(n) 
				for n in sorted_graph]
			# if n in graph else 0
			# for n in range(2995)]
		) ** .75
		node_map = np.array(sorted_graph, dtype=np.int32)

	# neg_samples = neg_samples.flatten()
	neg_samples /= neg_samples.sum(axis=-1, keepdims=True)
	neg_samples = neg_samples.cumsum(axis=-1)
	assert np.allclose(neg_samples[..., -1], 1)

	return positive_samples, neg_samples, node_map
