import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import pandas as pd
import networkx as nx

import argparse

from headnet.utils import load_data

def split_nodes(nodes, 
	seed,
	val_split=0.05, 
	test_split=0.10, ):

	num_val_edges = int(np.ceil(len(nodes) * val_split))
	num_test_edges = int(np.ceil(len(nodes) * test_split))

	np.random.seed(seed)

	nodes = np.random.permutation(nodes)

	val_nodes = nodes[:num_val_edges]
	test_nodes = nodes[num_val_edges:num_val_edges+num_test_edges]
	train_nodes = nodes[num_val_edges+num_test_edges:]

	return train_nodes, val_nodes, test_nodes

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Script to remove edges for link prediction experiments")

	parser.add_argument("--edgelist", dest="edgelist", type=str, 
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
	output_dir = os.path.join(args.output, 
		"seed={:03d}".format(seed),)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir, exist_ok=True)

	training_nodes_fn = os.path.join(output_dir, 
		"nodes_train.csv")
	val_nodes_fn = os.path.join(output_dir, 
		"nodes_val.csv")
	test_nodes_fn = os.path.join(output_dir, 
		"nodes_test.csv")
	
	graph, _, _ = load_data(args)
	print("loaded dataset")
	assert nx.is_directed(graph)

	train_nodes, val_nodes, test_nodes = \
		split_nodes(list(graph),
		seed,
		val_split=0.0,
		test_split=0.1)
	
	print ("writing train nodes to", training_nodes_fn)
	pd.DataFrame(train_nodes).to_csv(training_nodes_fn)
	print ("writing val nodes to", val_nodes_fn)
	pd.DataFrame(val_nodes).to_csv(val_nodes_fn)
	print ("writing test nodes to", test_nodes_fn)
	pd.DataFrame(test_nodes).to_csv(test_nodes_fn)





if __name__ == "__main__":
	main()
