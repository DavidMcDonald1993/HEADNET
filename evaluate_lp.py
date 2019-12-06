import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import numpy as np
import networkx as nx
import pandas as pd

import argparse

from headnet.utils import load_embedding, load_data

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import average_precision_score, roc_auc_score
import functools
import fcntl

import glob

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(u, v):
	u = np.expand_dims(u, axis=1)
	v = np.expand_dims(v, axis=0)
	mink_dp = -minkowski_dot(u, v)
	mink_dp = np.maximum(mink_dp - 1, 1e-15)
	return np.squeeze(np.arccosh(1 + mink_dp), axis=-1)

def hyperbolic_distance_poincare(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
	return np.arccosh(1 + 2 * uu / dd)

def logarithmic_map(p, x):

	alpha = -minkowski_dot(p, x)

	alpha = np.maximum(alpha, 1+1e-15)

	return np.arccosh(alpha) * (x - alpha * p) / \
		np.sqrt(alpha ** 2 - 1) 
		

def parallel_transport(p, q, x):
	assert len(p.shape) == len(q.shape) == len(x.shape)
	alpha = -minkowski_dot(p, q)
	return x + minkowski_dot(q - alpha * p, x) * (p + q) / \
		(alpha + 1) 

# def log_likelihood(x, mu, sigma_inv, sigma_det):

# 	# ignore zero t coordinate
# 	x = x[...,:-1]

# 	k = x.shape[-1]

# 	x_minus_mu = x - mu

# 	uu = np.sum(x_minus_mu ** 2 * sigma_inv, 
# 		axis=-1, keepdims=True) # assume sigma inv is diagonal

# 	return - 0.5 * (np.log(np.maximum(sigma_det, 1e-15)) +\
# 		uu + k * np.log(2 * np.pi))

# def hyperbolic_log_pdf(mus, sigmas):
	

# 	dim = mus.shape[1] - 1

# 	# project to tangent space
# 	source_mus = np.expand_dims(mus, axis=1)
# 	target_mus = np.expand_dims(mus, axis=0)

# 	to_tangent_space = logarithmic_map(source_mus, 
# 		target_mus)

# 	# parallel transport to mu zero
# 	mu_zero = np.zeros((1, 1, dim + 1))
# 	mu_zero[..., -1] = 1

# 	to_tangent_space_mu_zero = parallel_transport(source_mus,
# 	 mu_zero, to_tangent_space)

# 	# compute euclidean_log_pdf

# 	source_sigmas = np.expand_dims(sigmas, axis=0)
# 	sigma_inv = 1 / source_sigmas
# 	sigma_det = np.prod(source_sigmas, axis=-1, keepdims=True) 

# 	logs = log_likelihood(to_tangent_space_mu_zero, 
# 		np.zeros((1, 1, dim)), 
# 		sigma_inv, 
# 		sigma_det)
# 	logs = np.squeeze(logs, axis=-1)

# 	# compute log det proj v

# 	norm = np.sqrt(np.maximum(0.,
# 		minkowski_dot(to_tangent_space_mu_zero,
# 			to_tangent_space_mu_zero)))
# 	norm = np.squeeze(norm, axis=-1)

# 	log_det_proj = (dim - 0) * (np.log(np.maximum(np.sinh(norm), 1e-15)) -\
# 		np.log(np.maximum(norm, 1e-15)))
	
# 	return logs - log_det_proj

def kullback_leibler_divergence_euclidean(mus, sigmas):

	dim = mus.shape[1] - 1

	# project to tangent space
	source_mus = np.expand_dims(mus, axis=1)
	target_mus = np.expand_dims(mus, axis=0)

	source_sigmas = np.expand_dims(sigmas, axis=1)
	target_sigmas = np.expand_dims(sigmas, axis=0)

	x_minus_mu = target_mus - source_mus

	trace = np.sum(target_sigmas / \
		source_sigmas, 
		axis=-1, keepdims=True)

	uu = np.sum(x_minus_mu ** 2 / \
		source_sigmas, 
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = np.sum(np.log(target_sigmas), 
		axis=-1, keepdims=True) - \
		np.sum(np.log(source_sigmas), 
		axis=-1, keepdims=True)

	return np.squeeze(0.5 * (trace + uu - dim - log_det), axis=-1)


def kullback_leibler_divergence_hyperboloid(mus, sigmas):

	dim = mus.shape[1] - 1

	# project to tangent space
	source_mus = np.expand_dims(mus, axis=1)
	target_mus = np.expand_dims(mus, axis=0)

	to_tangent_space = logarithmic_map(source_mus, 
		target_mus)

	# for x, y in zip(source_mus, to_tangent_space):
		# assert np.allclose(minkowski_dot(x, y), 0, atol=1e-6)

	# parallel transport to mu zero
	mu_zero = np.zeros((1, 1, dim + 1))
	mu_zero[..., -1] = 1
	
	to_tangent_space_mu_zero = parallel_transport(source_mus,
		mu_zero, 
		to_tangent_space)

	# assert np.allclose(to_tangent_space_mu_zero[..., -1], 0, atol=1e-6), np.abs(to_tangent_space_mu_zero[..., -1]).max()

	# mu is zero vector
	# ignore zero t coordinate
	mus = to_tangent_space_mu_zero[...,:-1]

	# dists = hyperbolic_distance_hyperboloid(mus, 
	# 	mus)
	# print (dists.shape)
	# raise SystemExit


	# sigmas = np.maximum(sigmas, 1e-15)

	source_sigmas = np.expand_dims(sigmas, axis=1)
	target_sigmas = np.expand_dims(sigmas, axis=0)

	sigma_ratio = target_sigmas / source_sigmas
	sigma_ratio = np.maximum(sigma_ratio, 1e-15)

	trace_fac = np.sum(sigma_ratio,
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(mus ** 2 / \
		source_sigmas,
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = np.sum(np.log(sigma_ratio),
		axis=-1, keepdims=True)

	return np.squeeze(0.5 * \
		(trace_fac + mu_sq_diff - dim - log_det))

	# trace = np.sum(target_sigmas / \
	# 	source_sigmas, 
	# 	axis=-1, keepdims=True)

	# uu = np.sum(x_minus_mu ** 2 / \
	# 	source_sigmas, 
	# 	axis=-1, keepdims=True) # assume sigma is diagonal

	# log_det = np.sum(np.log(target_sigmas), 
	# 	axis=-1, keepdims=True) - \
	# 	np.sum(np.log(source_sigmas), 
	# 	axis=-1, keepdims=True)

	# return np.squeeze(0.5 * (trace + uu - dim - log_det), axis=-1)

def euclidean_distance(X):
	return euclidean_distances(X)

def evaluate_rank_and_MAP(scores, 
	edgelist, non_edgelist):
	assert not isinstance(edgelist, dict)
	assert (scores <= 0).all()

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	edge_scores = scores[edgelist[:,0], edgelist[:,1]]
	non_edge_scores = scores[non_edgelist[:,0], non_edgelist[:,1]]

	labels = np.append(np.ones_like(edge_scores), 
		np.zeros_like(non_edge_scores))
	scores_ = np.append(edge_scores, non_edge_scores)
	ap_score = average_precision_score(labels, scores_) # macro by default
	auc_score = roc_auc_score(labels, scores_)
		
	# fpr, tpr, thresholds = roc_curve(labels, scores_)
	# import matplotlib.pyplot as plt

	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr, color='darkorange',
	# 		lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic')
	# plt.legend(loc="lower right")
	# plt.show()

	idx = (-non_edge_scores).argsort()
	ranks = np.searchsorted(-non_edge_scores, 
		-edge_scores, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"AUROC =", auc_score)

	return ranks, ap_score, auc_score

def evaluate_mean_average_precision(scores, 
	edgelist, 
	non_edgelist):

	import random
	random.seed(0)

	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict.update({u: []})
		edgelist_dict[u].append(v)

	non_edgelist_dict = {}
	for u, v in non_edgelist:
		if u not in non_edgelist_dict:
			non_edgelist_dict.update({u: []})
		non_edgelist_dict[u].append(v)

	precisions = []
	for u in set(edgelist_dict).\
		intersection(set(non_edgelist_dict)):
		
		true_neighbours = edgelist_dict[u]
		non_neighbours = non_edgelist_dict[u]
		# non_neighbours = random.sample(non_neighbours, len(true_neighbours))
		labels = np.append(np.ones_like(true_neighbours), 
			np.zeros_like(non_neighbours))
		scores_ = scores[u, true_neighbours+non_neighbours]
		s = average_precision_score(labels, scores_)
		precisions.append(s)

	return np.mean(precisions)

def touch(path):
	with open(path, 'a'):
		os.utime(path, None)

def read_edgelist(fn):
	edges = []
	with open(fn, "r") as f:
		for line in (l.rstrip() for l in f.readlines()):
			edge = tuple(int(i) for i in line.split("\t"))
			edges.append(edge)
	return edges

def lock_method(lock_filename):
	''' Use an OS lock such that a method can only be called once at a time. '''

	def decorator(func):

		@functools.wraps(func)
		def lock_and_run_method(*args, **kwargs):

			# Hold program if it is already running 
			# Snippet based on
			# http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
			fp = open(lock_filename, 'r+')
			done = False
			while not done:
				try:
					fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
					done = True
				except IOError:
					pass
			return func(*args, **kwargs)

		return lock_and_run_method

	return decorator 

def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
	lock_method(lock_filename)(fn)(*args, **kwargs)

def save_test_results(filename, seed, data, ):
	d = pd.DataFrame(index=[seed], data=data)
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)

def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate link prediction')
	

	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str, 
		help="path to load training and removed edges")
	
	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", "kle", "klh"])

	return parser.parse_args()

def elu(x, alpha=1.):
	x = x.copy()
	mask = x <= 0
	x[mask] = alpha * (np.exp(x[mask]) - 1)
	return x

def main():

	args = parse_args()

	args.directed = True

	graph, _, _ = load_data(args)
	assert nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	dist_fn = args.dist_fn
	# assert dist_fn in ["kle", "klh"]

	print ("distance function is", dist_fn)

	seed= args.seed
	removed_edges_dir = os.path.join(args.output, "seed={:03d}".format(seed), "removed_edges")

	test_edgelist_fn = os.path.join(removed_edges_dir, "test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, "test_non_edges.tsv")

	files = sorted(glob.iglob(os.path.join(args.embedding_directory, 
		"*.csv")))

	if dist_fn == "klh":
		embedding_filename = os.path.join(args.embedding_directory, 
			"final_embedding.csv.gz")
		variance_filename = os.path.join(args.embedding_directory,
			"final_variance.csv.gz")
	elif dist_fn == "kle":
		embedding_filename = os.path.join(args.embedding_directory, 
			"mu.csv.gz")
		variance_filename = os.path.join(args.embedding_directory,
			"sigma.csv.gz")
	else:

		files = sorted(glob.iglob(os.path.join(args.embedding_directory, 
			"*_embedding.csv.gz")))
		embedding_filename = files[-1]

	print ("loading embedding from", embedding_filename)
	embedding_df = load_embedding(embedding_filename)
	
	# row 0 is embedding for node 0
	# row 1 is embedding for node 1 etc...
	embedding_df = embedding_df.reindex(sorted(embedding_df.index))
	embedding = embedding_df.values

	if  dist_fn in ["kle", "klh"]:
		print ("loading variance from", variance_filename)
		variance_df = load_embedding(variance_filename)
		variance_df = variance_df.reindex(sorted(variance_df.index))
		variance = variance_df.values

	if dist_fn == "poincare":
		dists = hyperbolic_distance_poincare(embedding)
	elif dist_fn == "hyperboloid":
		dists = hyperbolic_distance_hyperboloid(embedding, embedding)
	elif dist_fn == "klh":
		dists = kullback_leibler_divergence_hyperboloid(embedding, variance)
	elif dist_fn == "kle":
		dists = kullback_leibler_divergence_euclidean(embedding, variance)
	else: 
		dists = euclidean_distance(embedding)

	print ("loading test edges from {}".format(test_edgelist_fn))
	print ("loading test non-edges from {}".format(test_non_edgelist_fn))

	test_edges = read_edgelist(test_edgelist_fn)
	test_non_edges = read_edgelist(test_non_edgelist_fn)

	print ("number of test edges:", len(test_edges))
	print ("number of test non edges:", len(test_non_edges))

	test_results = dict()

	(mean_rank_lp, ap_lp, 
	roc_lp) = evaluate_rank_and_MAP(-dists, 
	test_edges, 
	test_non_edges)

	test_results.update({"mean_rank_lp": mean_rank_lp, 
		"ap_lp": ap_lp,
		"roc_lp": roc_lp})

	map_lp = evaluate_mean_average_precision(-dists, 
		test_edges, 
		nx.non_edges(graph))

	print ("MAP lp", map_lp)

	test_results.update({"map_lp": map_lp})

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	test_results_filename = os.path.join(test_results_dir, 
		"test_results.csv")
	test_results_lock_filename = os.path.join(test_results_dir, 
		"test_results.lock")
	touch(test_results_lock_filename)

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, 
		test_results_filename, seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()