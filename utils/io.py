
import os 
import re

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import csr_matrix, load_npz


def write_edgelist_to_file(edgelist, filename, delimiter="\t"):
	# g = nx.DiGraph(edgelist)
	# nx.write_edgelist(g, filename, delimiter="\t")
	with open(filename, "w") as f:
		for u, v in edgelist:
			f.write("{}{}{}\n".format(u, delimiter, v))

def load_data(
	graph_filename,
	features_filename,
	labels_filename,
	directed=True,
	):

	print ("loading graph from", graph_filename)

	if graph_filename.endswith(".npz"):
		
		graph = load_npz(graph_filename)

	elif graph_filename.endswith(".tsv") or graph_filename.endswith(".tsv.gz"):

		graph = nx.read_weighted_edgelist(graph_filename, 
			delimiter="\t", nodetype=int,
			create_using=nx.DiGraph() 
				if directed else nx.Graph())

		zero_weight_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w == 0.]
		print ("removing", len(zero_weight_edges), "edges with 0. weight")
		graph.remove_edges_from(zero_weight_edges)

		print ("ensuring all weights are positive")
		nx.set_edge_attributes(graph, name="weight", 
			values={edge: np.abs(np.int8(weight)) 
			for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

		print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

	else:
		raise NotImplementedError


	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv") or features_filename.endswith(".csv.gz"):
			features = pd.read_csv(features_filename, 
				index_col=0, sep=",")
			features = features.reindex(sorted(features.index)).values
			print ("no scaling applied")
			features = csr_matrix(features)

		elif features_filename.endswith(".npz"):
			
			features = load_npz(features_filename)

		else:
			raise NotImplementedError

		assert isinstance(features, csr_matrix)

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
			raise NotImplementedError
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] 
				for n in sorted(graph)], dtype=np.int)
		else:
			raise NotImplementedError

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
