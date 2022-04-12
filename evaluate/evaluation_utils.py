import os

import numpy as np
import pandas as pd

import glob


import functools
import fcntl

import random

import types

from sklearn.metrics import average_precision_score, roc_auc_score#, roc_curve

def euclidean_distance(u, v):
	return np.linalg.norm(u - v, axis=-1)

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(u, v):
	mink_dp = -minkowski_dot(u, v)
	mink_dp = np.maximum(mink_dp - 1, 1e-15)
	return np.squeeze(np.arccosh(1 + mink_dp), axis=-1)

def hyperbolic_distance_poincare(u, v):
	assert len(u.shape) == len(v.shape)
	norm_u_sq = np.linalg.norm(u, keepdims=False, axis=-1) ** 2
	# norm_u_sq = np.minimum(norm_u_sq, np.nextafter(1,0, ))
	norm_u_sq = np.minimum(norm_u_sq, 1-1e-1)
	norm_v_sq = np.linalg.norm(v, keepdims=False, axis=-1)
	# norm_v_sq = np.minimum(norm_v_sq, np.nextafter(1,0, ))
	norm_v_sq = np.minimum(norm_v_sq, 1-1e-7)
	uu = np.linalg.norm(u - v, keepdims=False, axis=-1, ) ** 2
	dd = (1 - norm_u_sq) * (1 - norm_v_sq)

	return np.arccosh(1 + 2 * uu / dd + 1e-7)

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


def kullback_leibler_divergence_euclidean(
	source_mus,
	source_sigmas,
	target_mus,
	target_sigmas):

	dim = source_mus.shape[1] 

	# project to tangent space

	sigma_ratio = target_sigmas / source_sigmas

	x_minus_mu = target_mus - source_mus

	trace = np.sum(sigma_ratio, 
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(x_minus_mu ** 2 / \
		source_sigmas, 
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = np.sum(np.log(sigma_ratio), 
		axis=-1, keepdims=True)

	return np.squeeze(
		0.5 * (trace + mu_sq_diff - dim - log_det), 
		axis=-1)

def kullback_leibler_divergence_hyperboloid(
	source_mus,
	source_sigmas,
	target_mus,
	target_sigmas):

	dim = source_mus.shape[1] - 1

	to_tangent_space = logarithmic_map(source_mus, 
		target_mus)

	# parallel transport to mu zero
	mu_zero = np.zeros((1, dim + 1))
	mu_zero[..., -1] = 1
	
	to_tangent_space_mu_zero = parallel_transport(source_mus,
		mu_zero, 
		to_tangent_space)

	# mu is zero vector
	# ignore zero t coordinate
	x_minus_mu = to_tangent_space_mu_zero[...,:-1]

	sigma_ratio = target_sigmas / source_sigmas

	trace_fac = np.sum(sigma_ratio,
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(x_minus_mu ** 2 / \
		source_sigmas,
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = np.sum(np.log(sigma_ratio), 
		axis=-1, keepdims=True)

	return np.squeeze(
		0.5 * (trace_fac + mu_sq_diff - dim - log_det), 
		axis=-1
	)

def load_file(filename, header="infer", sep=","):
	print ("reading from", filename)
	df = pd.read_csv(filename, index_col=0, 
		header=header, sep=sep)
	idx = sorted(df.index)
	df = df.reindex(idx)
	return df.values

def load_hyperboloid(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*_embedding.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename)

	return embedding

def load_poincare(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"embedding.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename)

	return embedding

def load_poincare_hgcn(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, "*.npy")))
	embedding_filename = files[-1]
	return np.load(embedding_filename)

def load_euclidean(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename, header=None, sep=" ")
	return embedding

def load_klh(embedding_directory):
	embedding_filename = os.path.join(embedding_directory, 
			"final_embedding.csv.gz")
	variance_filename = os.path.join(embedding_directory,
		"final_variance.csv.gz")

	embedding = load_file(embedding_filename)
	variance = load_file(variance_filename)

	return embedding, variance

def load_kle(embedding_directory):

	embedding_filename = os.path.join(
		embedding_directory, 
		"mu.csv.gz")
	if not os.path.exists(embedding_filename):
		# HEADNET euclidean
		embedding_filename = os.path.join(
			embedding_directory, 
			"final_embedding.csv.gz")
	
	variance_filename = os.path.join(
		embedding_directory,
		"sigma.csv.gz")
	if not os.path.exists(variance_filename):
		# HEADNET euclidean
		variance_filename = os.path.join(
			embedding_directory,
			"final_variance.csv.gz",
		)
	embedding = load_file(embedding_filename)
	variance = load_file(variance_filename)


	return embedding, variance

def load_st(embedding_directory):
	source_filename = os.path.join(embedding_directory, 
		"source.csv.gz")
	target_filename = os.path.join(embedding_directory,
			"target.csv.gz")

	source = load_file(source_filename)
	target = load_file(target_filename)
	return source, target

def load_embedding_for_evaluation(
	dist_fn, 
	embedding_directory,
	):
	print ("loading embedding from", embedding_directory)

	if dist_fn == "hyperboloid":
		embedding = load_hyperboloid(embedding_directory)
		return embedding
	elif dist_fn == "poincare":
		embedding = load_poincare(embedding_directory)
		return embedding
	elif dist_fn == "poincare_hgcn":
		embedding = load_poincare_hgcn(embedding_directory)
		return embedding 
	elif dist_fn == "euclidean":
		embedding = load_euclidean(embedding_directory)
		return embedding
	elif dist_fn == "klh":
		embedding, variance = load_klh(embedding_directory)
		return embedding, variance
	elif dist_fn == "kle":
		embedding, variance = load_kle(embedding_directory)
		return embedding, variance
	elif dist_fn == "st":
		source, target = load_st(embedding_directory)
		return source, target

def compute_scores(U, V, dist_fn):
	assert isinstance(U, types.GeneratorType)
	assert isinstance(V, types.GeneratorType)


	if dist_fn == "hyperboloid":
		scores = -np.concatenate([
			hyperbolic_distance_hyperboloid(u, v)
			for u, v in zip(U, V)])
	elif dist_fn in {"poincare", "poincare_hgcn"}:
		scores = -np.concatenate([
			hyperbolic_distance_poincare(u, v)
			for u, v in zip(U, V)])
	elif dist_fn == "euclidean":
		scores = -np.concatenate([
			euclidean_distance(u, v)
			for u, v in zip(U, V)])
	elif dist_fn == "klh":
		scores = -np.concatenate([
			kullback_leibler_divergence_hyperboloid(u[0], u[1], v[0], v[1])
			for u, v in zip(U, V)])
	elif dist_fn == "kle":
		scores = -np.concatenate([
			kullback_leibler_divergence_euclidean(u[0], u[1], v[0], v[1])
			for u, v in zip(U, V)])
	elif dist_fn == "st":
		scores = -np.concatenate([euclidean_distance(u, v)
			for u, v in zip(U, V)])

	return scores

def evaluate_mean_average_precision_and_p_at_k(
	embedding, 
	edgelist, 
	dist_fn,
	edges_to_skip=None,
	ks=(1,3,5,10), # p@k ks
	max_non_neighbours=1000,
	chunk_size=10000,
	):
	print ("evaluating mAP and p@k")

	if isinstance(embedding, tuple):
		N, _  = embedding[0].shape
	else:
		N, _  = embedding.shape

	all_nodes = set(range(N))

	# convert list of edge tuples to dictinoary
	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict[u] = set()
		edgelist_dict[u].add(v)


	# optional set of edges to skip in evaluation (i.e. the training edges in link prediction)
	if edges_to_skip:
		graph_edgelist_dict = {}
		for u, v in edges_to_skip:
			if u not in graph_edgelist_dict:
				graph_edgelist_dict[u] = set()
			if u in edgelist_dict and v not in edgelist_dict[u]:
				graph_edgelist_dict[u].add(v)

	all_node_average_precision_scores = []
	precisions_at_k = {
		k: [] 
		for k in ks
	}

	for i, u in enumerate(edgelist_dict):

		node_true_neighbours = edgelist_dict[u]
		# get all non-neighbours for node u
		node_non_neighbours = all_nodes - {u} - node_true_neighbours
		# remove optional edges
		if edges_to_skip and u in graph_edgelist_dict:
			node_non_neighbours -= graph_edgelist_dict[u]
		
		node_true_neighbours = list(node_true_neighbours)
		node_non_neighbours = list(node_non_neighbours)

		# take random sample of non-neighours
		if len(node_non_neighbours) > max_non_neighbours:
			node_non_neighbours = random.sample(
				node_non_neighbours, 
				k=max_non_neighbours,)

		all_node_neighbours = node_true_neighbours + node_non_neighbours
		num_chunks = int(np.ceil(len(all_node_neighbours) / chunk_size))

		if isinstance(embedding, tuple):
			if dist_fn in ("klh", "kle"):
				means, variances = embedding
				scores = compute_scores(
					((means[u:u+1], variances[u:u+1]) 
						for _ in range(num_chunks)),
					((means[all_node_neighbours[chunk_num*chunk_size:(chunk_num+1)*chunk_size]], 
						variances[all_node_neighbours[chunk_num*chunk_size:(chunk_num+1)*chunk_size]]) 
						for chunk_num in range(num_chunks)),
					dist_fn)
			else:
				assert dist_fn == "st"
				source, target = embedding
				scores = compute_scores(
					(source[u:u+1] for _ in range(num_chunks)),
					(target[all_node_neighbours[chunk_num*chunk_size:(chunk_num+1)*chunk_size]]
						for chunk_num in range(num_chunks)),
					dist_fn)
		else:
			scores = compute_scores(
				(embedding[u:u+1] for _ in range(num_chunks)), 
				(embedding[all_node_neighbours[chunk_num*chunk_size:(chunk_num+1)*chunk_size]]
					for chunk_num in range(num_chunks)),
				dist_fn)
		assert len(scores.shape) == 1

		# initialise labels
		# true neighbours are listed first
		labels = np.append(
			np.ones_like(node_true_neighbours),
			np.zeros_like(node_non_neighbours),
		)
		
		node_average_precision_score = average_precision_score(labels, scores)
		all_node_average_precision_scores.append(node_average_precision_score)

		# smallest score to largest
		nodes_sorted = scores.argsort()

		# convert true neighbours list to set for efficient `in` operation
		# since order of true_neighbours is not important now
		node_true_neighbours = set(node_true_neighbours)

		# compute p@k for all k
		for k in ks:
			top_k_nodes = nodes_sorted[-k:]
			# mean of top_k_nodes in true_neighbours
			node_precision_score_k = np.mean(
				[all_node_neighbours[u] in node_true_neighbours 
				for u in top_k_nodes]
			)
			# add precision score for current k to precisions_at_k dict
			precisions_at_k[k].append(node_precision_score_k)

		if i % 1000 == 0:
			print ("completed", i, "/", len(edgelist_dict))

	mAP = np.mean(all_node_average_precision_scores)
	print ("MAP", mAP)

	precisions_at_k = {
		k: (np.mean(v) if len(v) > 0 else 0)
			for k, v in precisions_at_k.items()
		}

	return mAP, precisions_at_k

def evaluate_rank_AUROC_AP(
	embedding,
	test_edges, 
	test_non_edges, 
	dist_fn,
	):
	print ("evaluating rank, AUROC and AP")

	edge_scores = get_scores(
		embedding,
		test_edges, 
		dist_fn)

	non_edge_scores = get_scores(
		embedding,
		test_non_edges, 
		dist_fn)

	assert len(edge_scores.shape) == 1
	assert len(non_edge_scores.shape) == 1

	labels = np.append(np.ones_like(edge_scores), 
		np.zeros_like(non_edge_scores))
	scores_ = np.append(edge_scores, non_edge_scores)
	ap_score = average_precision_score(labels, scores_) # macro by default
	auc_score = roc_auc_score(labels, scores_)

	idx = (-non_edge_scores).argsort()
	ranks = np.searchsorted(-non_edge_scores, 
		-edge_scores, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"AUROC =", auc_score)

	return ranks, ap_score, auc_score

def get_scores(embedding, 
	edges, 
	dist_fn, 
	chunk_size=10000):
	print ("computing scores")

	num_chunks = int(np.ceil(edges.shape[0]  / chunk_size))

	if dist_fn in ("kle", "klh"):
		# split embedding into means and variances
		means, variances = embedding

		embedding_u = (
			(means[edges[i*chunk_size:(i+1)*chunk_size, 0]], 
				variances[edges[i*chunk_size:(i+1)*chunk_size, 0]])
			for i in range(num_chunks)
		)
		embedding_v = (
			(means[edges[i*chunk_size:(i+1)*chunk_size, 1]], 
				variances[edges[i*chunk_size:(i+1)*chunk_size, 1]])
			for i in range(num_chunks)
		)

	elif dist_fn == "st":
		source, target = embedding
		embedding_u = (source[edges[i*chunk_size:(i+1)*chunk_size,0]]
			for i in range(num_chunks))
		embedding_v = (target[edges[i*chunk_size:(i+1)*chunk_size,1]]
			for i in range(num_chunks))

	else:
		embedding_u = (embedding[edges[i*chunk_size:(i+1)*chunk_size,0]]
			for i in range(num_chunks))
		embedding_v = (embedding[edges[i*chunk_size:(i+1)*chunk_size,1]]
			for i in range(num_chunks))


	return compute_scores(embedding_u, embedding_v, dist_fn)

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

def check_complete(test_results_filename, seed):
	if os.path.exists(test_results_filename):
		existing_results = pd.read_csv(test_results_filename, index_col=0)
		if seed in existing_results.index:
			print (test_results_filename, ": seed=", seed, "complete --terminating")
			return True 
	return False
	