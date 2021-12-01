import os 
import random

import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity


def determine_positive_and_negative_samples(graph, args):

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
		) ** .75
		node_map = np.array(sorted_graph, dtype=np.int32)

	neg_samples /= neg_samples.sum(axis=-1, keepdims=True)
	neg_samples = neg_samples.cumsum(axis=-1)
	assert np.allclose(neg_samples[..., -1], 1)

	print ("found", positive_samples.shape[0], "positive samples")

	return positive_samples, neg_samples, node_map

def sample_non_edges(nodes, edges, sample_size):
	assert isinstance(edges, set)
	nodes = list(nodes)
	# edges = set(edges)
	print ("sampling", sample_size, "non edges")
	non_edges = set()
	while len(non_edges) < sample_size:
		non_edges_= {tuple(random.sample(nodes, k=2))
			for _ in range(sample_size - len(non_edges))}
		non_edges_ -= edges 
		non_edges = non_edges.union(non_edges_)
		# if edge not in edges + non_edges:
		# 	non_edges.append(edge)
	return list(non_edges)

def determine_positive_samples_and_probs(graph, features, args):

	N = len(graph)
	negative_samples = np.ones((N, N), dtype=bool)
	np.fill_diagonal(negative_samples, 0)

	nodes = sorted(graph)

	if args.no_walks:

		positive_samples = list(graph.edges())
		positive_samples += [(v, u) # undirected graph
			for u, v in positive_samples]

		counts = np.array([graph.degree(u)
			for u in sorted(graph)])

		if not args.all_negs:
			for n in nodes:
				negative_samples[n, list(graph.neighbors(n))] = 0

	else:
		positive_samples = []
		# positive_samples = dict()
		counts = np.zeros(N)

		print ("determining positive and negative samples", 
			"using random walks")

		walks = perform_walks(graph, features, args)

		if not args.visualise:
			del graph
		del features

		context_size = args.context_size

		for num_walk, walk in enumerate(walks):
			for i in range(len(walk)):
				u = walk[i]
				counts[u] += 1
				for j in range(context_size):

					if i+j+1 >= len(walk):
						break
					v = walk[i+j+1]
					if u == v:
						continue

					positive_samples.append((u, v))
					positive_samples.append((v, u))
					# if (u, v) not in positive_samples:
					# 	positive_samples[(u, v)] = 0
					# if (v, u) not in positive_samples:
					# 	positive_samples[(v, u)] = 0
					# positive_samples[(u, v)] += 1
					# positive_samples[(v, u)] += 1

					if not args.all_negs:
						negative_samples[u, v] = 0
						negative_samples[v, u] = 0

			if num_walk % 1000 == 0:  
				print ("processed walk {:04d}/{}".format(
					num_walk, 
					len(walks)
					))

	print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
	print ("found {} positive sample pairs".format(
		len(positive_samples)))

	counts = counts ** 0.75
	probs = counts[None, :] 
	probs = probs * negative_samples
	assert (probs > 0).any(axis=-1).all(), \
		"a node in the network does not have any negative samples"
	probs /= probs.sum(axis=-1, keepdims=True)
	probs = probs.cumsum(axis=-1)

	assert np.allclose(probs[:,-1], 1)

	print ("PREPROCESSED NEGATIVE SAMPLE PROBABILTIES")

	positive_samples = np.array(positive_samples)

	if not args.use_generator:
		print ("SORTING POSITIVE SAMPLES")
		idx = positive_samples[:,0].argsort()
		positive_samples = positive_samples[idx]
		print ("SORTED POSITIVE SAMPLES")

	return positive_samples, probs

def select_negative_samples(positive_samples, probs, num_negative_samples):

	negative_samples = (choose_negative_samples(x, num_negative_samples) 
		for x in ((u, count, probs[u]) 
		for u, count in sorted(Counter(positive_samples[:,0]).items(), key=lambda x: x[0])))
	negative_samples = np.concatenate([arr for _, arr in 
		sorted(negative_samples, key=lambda x: x[0])], axis=0,)

	print ("selected negative samples")

	return positive_samples, negative_samples

def determine_positive_and_negative_samples_using_random_walks(graph, features, args):

	graph = graph.to_undirected() # we perform walks on undirected matrix

	nodes = graph.nodes()

	if not isinstance(nodes, set):
		nodes = set(nodes)

	positive_samples, probs = \
		determine_positive_samples_and_probs(
			graph, features, args)

	if not args.use_generator:
		print("Training without generator -- selecting negative samples before training")
		positive_samples, negative_samples = select_negative_samples(
			positive_samples, probs, args.num_negative_samples)
		probs = None
	else:
		print ("Training using data generator -- skipping selection of negative samples")
		negative_samples = None 

	return positive_samples, negative_samples, probs

def choose_negative_samples(x, num_negative_samples):
		u, count, probs = x
		return u, np.searchsorted(probs, np.random.rand(count, num_negative_samples)).astype(np.int32)


class Graph():
	def __init__(self, 
		graph, 
		is_directed, 
		p, 
		q, 
		alpha=0, 
		feature_sim=None, 
		seed=0):
		assert not nx.is_directed(graph)
		self.graph = graph
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.alpha = alpha
		self.feature_sim = feature_sim 
		if self.feature_sim is not None:
			self.feature_sim = self.feature_sim.cumsum(-1)

		np.random.seed(seed)
		random.seed(seed)


	def node2vec_walk(self,  start_node, walk_length,):
		'''
		Simulate a random walk starting from start node.
		'''
		graph = self.graph
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		feature_sim = self.feature_sim

		jump = False
		preprocessed_edges = alias_edges is not None

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			# node2vec style random walk 
			cur_nbrs = sorted(graph.neighbors(cur))

			if (feature_sim is not None 
				and self.alpha > 0 
				and not (feature_sim[cur]<1e-15).all() 
				and (np.random.rand() < self.alpha or len(cur_nbrs) == 0)):
				# random jump based on attribute similarity
				next_ = np.searchsorted(feature_sim[cur],
					np.random.rand())
				walk.append(next_)
				jump = True

			elif len(cur_nbrs) > 0:
				if len(walk) == 1 or jump or not preprocessed_edges:
					walk.append(cur_nbrs[alias_draw_original(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next_ = cur_nbrs[alias_draw_original(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next_)
				jump = False
			else:
				break

		return walk

	
	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		graph = self.graph
		walks = []
		nodes = sorted(graph.nodes())
		i = 0

		print ("PERFORMING WALKS")

		# with Pool(processes=2) as p:
		# 	nodes *= num_walks
		# 	walks = p.map(functools.partial(self.node2vec_walk, walk_length=walk_length), nodes)

		for _ in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(node, 
					walk_length=walk_length, ))

				if i % 1000 == 0:
					print ("performed walk {:04d}/{}".format(i, num_walks*len(graph)))
				i += 1

				# yield self.node2vec_walk(node, 
				# 	walk_length=walk_length, )

		return walks

	def get_alias_node(self, node):

		graph = self.graph

		unnormalized_probs = [abs(graph[node][nbr]['weight']) for nbr in sorted(graph.neighbors(node))]
		norm_const = sum(unnormalized_probs) + 1e-7
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return node, alias_setup_original(normalized_probs)

	def get_alias_edge(self, edge):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		src, dst = edge

		graph = self.graph
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(graph.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight'])/p)
			elif graph.has_edge(dst_nbr, src):
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight']))
			else:
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight'])/q)
		norm_const = sum(unnormalized_probs) + 1e-7
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return edge, alias_setup_original(normalized_probs)


	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		print ("preprocessing transition probs")
		graph = self.graph
		is_directed = self.is_directed

		print ("preprocessing nodes")

		# with Pool(processes=None) as p:
		# 	alias_nodes = p.map(self.get_alias_node, graph.nodes())
		alias_nodes = (self.get_alias_node(node) for node in graph.nodes())
		alias_nodes = {node: alias_node for node, alias_node in alias_nodes}

		print ("preprocessed all nodes")
		self.alias_nodes = alias_nodes

		edges = list(graph.edges())
		if not is_directed:
			edges += [(v, u) for u, v in edges]

		if self.p != 1 or self.q != 1:
			print ("preprocessing edges")

			# with Pool(processes=None) as p:
				# alias_edges = p.map(self.get_alias_edge, edges)
			alias_edges = (self.get_alias_edge(edge) for edge in edges)
			alias_edges = {edge: alias_edge for edge, alias_edge in alias_edges}

			print ("preprocessed all edges")
		else:
			print ("p and q are both set to 1, skipping preprocessing edges")
			alias_edges = None
		self.alias_edges = alias_edges

def save_walks_to_file(walks, walk_file):
	with open(walk_file, "w") as f:
		for walk in walks:
			f.write(",".join([str(n) for n in walk]) + "\n")

def load_walks_from_file(walk_file, ):

	walks = []

	with open(walk_file, "r") as f:
		for line in (line.rstrip() for line in f.readlines()):
			walks.append([int(n) for n in line.split(",")])
	return walks

def make_feature_sim(features):

	if features is not None:
		feature_sim = cosine_similarity(features)
		np.fill_diagonal(feature_sim, 0) # remove diagonal
		feature_sim[feature_sim < 1e-15] = 0
		feature_sim /= np.maximum(
			feature_sim.sum(axis=-1, keepdims=True), 1e-15) # row normalize
	else:
		feature_sim = None

	return feature_sim


def perform_walks(graph, features, args):

	walk_file = args.walk_filename

	if not os.path.exists(walk_file):

		feature_sim = make_feature_sim(features)

		if args.alpha > 0:
			assert features is not None

		node2vec_graph = Graph(
			graph=graph, 
			is_directed=False,
			p=args.p, 
			q=args.q,
			alpha=args.alpha, 
			feature_sim=feature_sim, 
			seed=args.seed)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(
			num_walks=args.num_walks, 
			walk_length=args.walk_length)
		
		if args.save_walks: 
			walks = list(walks)
			save_walks_to_file(walks, walk_file)
			print ("saved walks to {}".format(walk_file))

	else:
		print ("loading walks from {}".format(walk_file))
		walks = load_walks_from_file(walk_file, )

	return walks


def alias_setup(probs):

	n, probs = probs

	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return n, (J, q)

def alias_setup_original(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q, size=1):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
	r = np.random.uniform(size=size)
	idx = r >= q[kk]
	kk[idx] = J[kk[idx]]
	return kk

def alias_draw_original(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]