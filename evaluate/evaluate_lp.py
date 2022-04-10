import sys
import os.path


if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os

import random

import numpy as np
import networkx as nx
import pandas as pd

import argparse

import pickle as pkl

from utils.io import load_data
from evaluate.evaluation_utils import (load_embedding_for_evaluation, evaluate_rank_AUROC_AP, evaluate_mean_average_precision, read_edgelist)


def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate link prediction')
	
	parser.add_argument("--graph", dest="graph", type=str, 
		help="graph to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--removed_edges_dir", dest="removed_edges_dir", type=str, 
		help="path to load removed edges")
	
	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", 
		"kle", "klh", "st", "poincare_hgcn"])

	return parser.parse_args()


def main():

	args = parse_args()

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	test_results_filename = os.path.join(test_results_dir, 
		"{}.pkl".format(args.seed))

	args.directed = True

	graph_filename = args.graph
	features_filename = args.features
	labels_filename = args.labels

	graph, _, _ = load_data(
		graph_filename=graph_filename,
		features_filename=features_filename,
		labels_filename=labels_filename)

	print ("Loaded dataset")
	print ()

	if isinstance(graph, nx.DiGraph):
		graph = nx.adjacency_matrix(graph, 
			nodelist=sorted(graph),
			weight=None).astype(bool)

	N = graph.shape[0]
	print ("network has", N, "nodes")

	graph_edges = list(zip(*graph.nonzero()))
	del graph

	seed = args.seed
	random.seed(seed)

	removed_edges_dir = args.removed_edges_dir

	test_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_non_edges.tsv")

	print ("loading test edges from {}".format(test_edgelist_fn))
	print ("loading test non-edges from {}".format(test_non_edgelist_fn))

	test_edges = read_edgelist(test_edgelist_fn)
	test_non_edges = read_edgelist(test_non_edgelist_fn)

	test_edges = np.array(test_edges)
	test_non_edges = np.array(test_non_edges)

	print ("number of test edges:", len(test_edges))
	print ("number of test non edges:", len(test_non_edges))

	embedding = load_embedding_for_evaluation(args.dist_fn, args.embedding_directory)

	test_results = dict()

	mean_rank_lp, ap_lp, roc_lp = evaluate_rank_AUROC_AP(
			embedding,
			test_edges, 
			test_non_edges,
			args.dist_fn)

	test_results.update(
		{
			"mean_rank_lp": mean_rank_lp, 
			"ap_lp": ap_lp,
			"roc_lp": roc_lp,
		}
	)

	map_lp, precisions_at_k = evaluate_mean_average_precision(
		embedding, 
		test_edges,
		args.dist_fn, 
		graph_edges=graph_edges
	)

	test_results.update(
		{"map_lp": map_lp}
	)

	# print p@k
	for k, pk in precisions_at_k.items():
		print ("precision at", k, pk)
	# update test_results_dict
	test_results.update(
		{
			"p@{}".format(k): pk
			for k, pk in precisions_at_k.items()
		}
	)

	print ("saving test results to {}".format(test_results_filename))

	test_results = pd.Series(test_results)

	with open(test_results_filename, "wb") as f:
		pkl.dump(test_results, f, pkl.HIGHEST_PROTOCOL)

	print ("done")


if __name__ == "__main__":
	main()