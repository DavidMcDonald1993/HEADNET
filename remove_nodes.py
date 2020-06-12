import os

import random
import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import csr_matrix, save_npz

import argparse

from headnet.utils import load_data
from remove_utils import write_edgelist_to_file, sample_non_edges

def split_nodes(nodes, 
	seed,
	val_split=0.05, 
	test_split=0.10, ):

	num_val_nodes = int(np.ceil(len(nodes) * val_split))
	num_test_nodes = int(np.ceil(len(nodes) * test_split))

	np.random.seed(seed)

	nodes = np.random.permutation(nodes)

	val_nodes = nodes[:num_val_nodes]
	test_nodes = nodes[num_val_nodes:num_val_nodes+num_test_nodes]
	train_nodes = nodes[num_val_nodes+num_test_nodes:]

	return train_nodes, val_nodes, test_nodes

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Script to remove nodes for link prediction experiments")

	parser.add_argument("--graph", dest="graph", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str, 
		help="path to save training and removed nodes")

	parser.add_argument('--directed', 
		action="store_true", 
		help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()
	return args

def main():

	args = parse_args()
	args.directed = True

	seed= args.seed
	random.seed(seed)

	training_edgelist_dir = os.path.join(args.output, 
		"seed={:03d}".format(seed), "training_edges")
	removed_edges_dir = os.path.join(args.output, 
		"seed={:03d}".format(seed), "removed_edges")

	if not os.path.exists(training_edgelist_dir):
		os.makedirs(training_edgelist_dir, exist_ok=True)
	if not os.path.exists(removed_edges_dir):
		os.makedirs(removed_edges_dir, exist_ok=True)

	# training_edgelist_fn = os.path.join(training_edgelist_dir, 
	# 	"graph.npz")

	val_edgelist_fn = os.path.join(removed_edges_dir, 
		"val_edges.tsv")
	val_non_edgelist_fn = os.path.join(removed_edges_dir, 
		"val_non_edges.tsv")
	test_edgelist_fn = os.path.join(removed_edges_dir,
		"test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_non_edges.tsv")
	
	graph, _, _ = load_data(args)
	print("loaded dataset")

	if isinstance(graph, nx.DiGraph):
		graph = nx.adjacency_matrix(graph, 
			nodelist=sorted(graph),
			weight=None).astype(bool)

	nodes = range(graph.shape[0])
	train_nodes, val_nodes, test_nodes = \
		split_nodes(
		nodes,
		seed,
		val_split=0.0,
		test_split=0.1)

	removed_nodes = np.append(val_nodes, test_nodes)
	for u in removed_nodes:
		assert u not in train_nodes

	print ("num train nodes:", len(train_nodes))
	print ("num val nodes:", len(val_nodes))
	print ("num test nodes:", len(test_nodes))

	edge_set = set(list(zip(*graph.nonzero())))

	if len(val_nodes) > 0:
		val_edges = [(u, v) for u, v in edge_set
			if u in val_nodes or v in val_nodes]
		val_non_edges = sample_non_edges(
			nodes, 
			edge_set, 
			len(val_edges))
	else:
		val_edges = []
		val_non_edges = []

	print ("determined val edges")
	
	if len(test_nodes) > 0:
		test_edges = [(u, v) for u, v in edge_set
			if u in test_nodes or v in test_nodes]
		test_non_edges = sample_non_edges(
			nodes, 
			edge_set.union(val_non_edges), # do not sample val edges
			len(test_edges))
	else:
		test_edges = []
		test_non_edges = []

	print ("determined test edges")

	for edge in val_edges + test_edges:
		graph[edge] = 0
	graph.eliminate_zeros()

	for u, v in val_edges:
		assert not graph[u, v]
	for u, v in test_edges:
		assert not graph[u, v]

	for u in removed_nodes:
		assert not graph[u].A.flatten().any()
		assert not graph.T[u].A.flatten().any()

	print ("removed edges")


	training_sparse_filename = os.path.join(training_edgelist_dir,
		"graph.npz")
	print ("writing adjacency matrix to", 
		training_sparse_filename)
	save_npz(training_sparse_filename, graph)

	training_edgelist_filename = os.path.join(training_edgelist_dir, 
		"edgelist.tsv.gz")
	print ("writing training edgelist to", 
		training_edgelist_filename)
	graph = graph.astype(int)
	nx.write_weighted_edgelist(nx.from_scipy_sparse_matrix(graph, 
		create_using=nx.DiGraph()), training_edgelist_filename, delimiter="\t")

	removed_nodes_filename = os.path.join(removed_edges_dir,
		"removed_nodes.txt")
	print ("writing removed nodes to", removed_nodes_filename)
	with open(removed_nodes_filename, "w") as f:
		f.write("\n".join((str(u) for u in removed_nodes)))

	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")



if __name__ == "__main__":
	main()
