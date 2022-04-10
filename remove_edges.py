import os

import random
import numpy as np
import networkx as nx
from scipy.sparse import save_npz

import argparse

from utils.io import load_data, write_edgelist_to_file
from utils.sampling import sample_non_edges


def split_edges(
	nodes, 
	edges, 
	seed,
	val_split=0.05, 
	test_split=0.10, 
	neg_mul=1,
	cover=True):
	
	assert isinstance(nodes, set)
	assert isinstance(edges, list)

	edge_set = set(edges)

	num_val_edges = int(np.ceil(len(edges) * val_split))
	num_test_edges = int(np.ceil(len(edges) * test_split))

	random.seed(seed)
	print ("shuffling edges using seed", seed)
	random.shuffle(edges)

	# ensure every node appears in edgelist
	cover_edges = []
	if cover:
		node_cover = set()
		for u, v in edges:
			if u not in node_cover or v not in node_cover:
				node_cover = node_cover.union({u, v})
				cover_edges.append((u, v))
			if len(node_cover) == len(nodes):
				break
		
		print ("determined cover", len(cover_edges))
		edges = filter(lambda edge: 
			edge not in cover_edges, edges)
		print ("filtering cover out of edges")

	val_edges = []
	test_edges = []
	train_edges = []
	for edge in edges:
		if len(val_edges) < num_val_edges:
			val_edges.append(edge)
		elif len(test_edges) < num_test_edges:
			test_edges.append(edge)
		else:
			train_edges.append(edge)

	train_edges += cover_edges

	print ("determined edge split")

	val_non_edges = sample_non_edges(
		nodes, 
		edge_set, 
		num_val_edges*neg_mul)
	print ("determined val non edges")
	test_non_edges = sample_non_edges(
		nodes,
		edge_set.union(val_non_edges),
		num_test_edges*neg_mul)
	print ("determined test non edges")

	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Script to remove edges for link prediction experiments")

	parser.add_argument("--graph", dest="graph", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str, 
		help="path to save training and removed edges")

	parser.add_argument('--directed', 
		action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()
	return args

def main():

	args = parse_args()
	args.directed = True

	seed= args.seed
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
	


	graph_filename = args.graph
	features_filename = args.features
	labels_filename = args.labels

	graph, _, _ = load_data(
		graph_filename=graph_filename,
		features_filename=features_filename,
		labels_filename=labels_filename)
	print("loaded dataset")

	if isinstance(graph, nx.DiGraph):
		graph = nx.adjacency_matrix(graph, 
			nodelist=sorted(graph),
			weight=None).astype(bool)

	nodes = set(range(graph.shape[0]))
	edges = list(zip(*graph.nonzero()))
	print ("enumerated edges")
	print ("number of edges", len(edges))

	(_, (val_edges, val_non_edges), (test_edges, test_non_edges)) = split_edges(
		nodes, 
		edges, 
		seed, 
		val_split=0,
		test_split=0.1)

	print ("number of val edges", len(val_edges), 
		"number of val non edges", len(val_edges))
	print ("number of test edges", len(test_edges), 
		"number of test non edges", len(test_edges))

	# remove val and test edges
	for edge in val_edges + test_edges:
		graph[edge] = 0
	graph.eliminate_zeros()

	assert np.all(np.logical_or(graph.A.any(0).flatten(), graph.A.any(1).flatten())) # check at least one connection
	for u, v in val_edges:
		assert not graph[u, v]
	for u, v in test_edges:
		assert not graph[u, v]

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

	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")

if __name__ == "__main__":
	main()
