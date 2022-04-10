
import sys
import os.path


if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os

import numpy as np
import pandas as pd

import argparse

import pickle as pkl

from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
	roc_auc_score, 
	f1_score, 
	precision_score, 
	recall_score)
from sklearn.preprocessing import LabelBinarizer

from skmultilearn.model_selection import IterativeStratification

from utils.hyperbolic_functions import (
	# hyperboloid_to_klein, 
	# poincare_ball_to_hyperboloid, 
	hyperboloid_to_poincare_ball,
	hyperboloid_to_poincare_ball, 
	poincare_ball_to_klein,
	)
from utils.io import load_data
from evaluate.evaluation_utils import load_embedding_for_evaluation

def compute_measures( 
    labels, 
    probs,
	threshold=.5,
	average="micro"):

	if len(labels.shape) == 1:
		labels = LabelBinarizer().fit_transform(labels)

	roc = roc_auc_score(labels, probs, average=average)
	pred = probs > threshold
	f1 = f1_score(labels, pred, average=average)
	precision = precision_score(labels, pred, average=average)
	recall = recall_score(labels, pred, average=average )

	return roc, f1, precision, recall

def evaluate_kfold_label_classification(
	embedding, 
	labels, 
	k=10,
	model="SVC",
	):
	assert len(labels.shape) == 2
	print (f"Evaluatung {k}-fold cross validation")
	
	print (f"using classifier: {model}")
	if model == "SVC":
		model = SVC(probability=True)
	else:
		raise NotImplementedError


	if labels.shape[1] == 1:
		print ("single label clasification")
		labels = labels.flatten()
		sss = StratifiedKFold(n_splits=k, 
			shuffle=True, 
			random_state=0)

	else:
		print ("multi-label classification")
		sss = IterativeStratification(
			n_splits=k, 
			order=1, # consider single labels only 
			)
		model = OneVsRestClassifier(model, )
			
	k_fold_rocs = np.zeros(k)
	k_fold_f1s = np.zeros(k)
	k_fold_precisions = np.zeros(k)
	k_fold_recalls = np.zeros(k)

	for i, (split_train, split_test) in enumerate(sss.split(embedding, labels, )):
		print ("Fold", i+1, "fitting model...")

		model.fit(embedding[split_train], labels[split_train])	
		probs = model.predict_proba(embedding[split_test])

		(k_fold_rocs[i], 
			k_fold_f1s[i], 
			k_fold_precisions[i], 
			k_fold_recalls[i]) = compute_measures(
				labels=labels[split_test],
				probs=probs,)

		print ("Completed {}/{} folds".format(i+1, k))

	return (np.mean(k_fold_rocs), np.mean(k_fold_f1s),
		np.mean(k_fold_precisions), np.mean(k_fold_recalls))

def evaluate_node_classification_with_label_fraction(
	embedding, 
	labels,
	label_fractions=np.arange(0.02, 0.11, 0.01), 
	n_repeats=30,
	model="SVC",
	):

	print ("Evaluating node classification")

	f1_micros = np.zeros((n_repeats, len(label_fractions)))
	f1_macros = np.zeros((n_repeats, len(label_fractions)))
	
	print (f"using classifier: {model}")
	if model == "SVC":
		model = SVC(probability=True)
	else:
		raise NotImplementedError


	if labels.shape[1] == 1:
		print ("single label clasification")
		labels = labels.flatten()

		split = StratifiedShuffleSplit
		for seed in range(n_repeats):
		
			for i, label_percentage in enumerate(label_fractions):
				print ("processing label percentage", i, f": {label_percentage:.02f}")
				sss = split(
					n_splits=1, 
					test_size=1-label_percentage, 
					random_state=seed)
				split_train, split_test = next(sss.split(embedding, labels))
				
				model.fit(embedding[split_train], labels[split_train])
				predictions = model.predict(embedding[split_test])

				f1_micro = f1_score(labels[split_test], predictions, 
					average="micro")
				f1_macro = f1_score(labels[split_test], predictions, 
					average="macro")
				# print (f"label_percentage {label_percentage:.02f}", "F1 micro", f1_micro, "F1 macro", f1_macro)

				f1_micros[seed, i] = f1_micro
				f1_macros[seed, i] = f1_macro
			print (f"completed repeat {seed+1}")

	else: # multilabel classification
		print ("multilabel classification")
		model = OneVsRestClassifier(model)
		split = IterativeStratification

		for seed in range(n_repeats):
		
			for i, label_percentage in enumerate(label_fractions):
				print ("processing label percentage", i, f": {label_percentage:.02f}")
				sss = split(n_splits=2, order=1, #random_state=seed,
					sample_distribution_per_fold=[1.0-label_percentage, label_percentage])
				split_train, split_test = next(sss.split(embedding, labels))
				model.fit(embedding[split_train], labels[split_train])
				predictions = model.predict(embedding[split_test])
				f1_micro = f1_score(labels[split_test], predictions, 
					average="micro")
				f1_macro = f1_score(labels[split_test], predictions, 
					average="macro")
				f1_micros[seed,i] = f1_micro
				f1_macros[seed,i] = f1_macro
			print (f"completed repeat {seed+1}")

	return label_fractions, f1_micros.mean(axis=0), f1_macros.mean(axis=0)

def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate node classification')
	
	parser.add_argument("--graph", dest="graph", type=str, 
		help="graph to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--embedding", dest="embedding_directory",  
		help="path of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", "kle", "klh", "poincare_hgcn"])

	return parser.parse_args()

def main():

	args = parse_args()

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	
	test_results_filename = os.path.join(
		test_results_dir, 
		f"{args.seed}.pkl")

	# print results if they exist and terminate
	if os.path.exists(test_results_filename):
		print (f"{test_results_filename} ALREADY EXISTS")
		with open(test_results_filename, "rb") as f:
			test_results = pkl.load(f)

		for k, v in test_results.items():
			if "macro" in k:
				continue # skip macro-average
			print (k, v)
		return 


	graph_filename = args.graph
	features_filename = args.features
	labels_filename = args.labels

	_, _, node_labels = load_data(
		graph_filename=graph_filename,
		features_filename=features_filename,
		labels_filename=labels_filename)

	print ("Loaded dataset")

	embedding = load_embedding_for_evaluation(
		dist_fn=args.dist_fn, 
		embedding_directory=args.embedding_directory)
	if isinstance(embedding, tuple):
		embedding, variance = embedding

	min_count = 10
	if node_labels.shape[1] == 1: # remove any node belonging to an under-represented class
		label_counts = Counter(node_labels.flatten())
		mask = np.array([label_counts[l] >= min_count
			for l in node_labels.flatten()])
		embedding = embedding[mask]
		node_labels = node_labels[mask]

	else:
		assert node_labels.shape[1] > 1
		idx = node_labels.sum(0) >= min_count
		node_labels = node_labels[:, idx]
		idx = node_labels.any(-1)
		embedding = embedding[idx]
		node_labels = node_labels[idx]

	if args.dist_fn in {"hyperboloid", "klh"}:
		print ("loaded a hyperboloid embedding")
		print ("projecting from hyperboloid to poincare")
		embedding = hyperboloid_to_poincare_ball(embedding)
		print ("projecting from poincare to klein")
		embedding = poincare_ball_to_klein(embedding)

	elif args.dist_fn == "poincare":
		print ("loaded a poincare embedding")
		print ("projecting from poincare to klein")
		embedding = poincare_ball_to_klein(embedding)

	test_results = {}
	
	label_percentages, f1_micros, f1_macros = \
		evaluate_node_classification_with_label_fraction(
			embedding=embedding, 
			labels=node_labels,
			n_repeats=5)

	for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
		print ("{:.2f}".format(label_percentage), 
			"micro = {:.5f}".format(f1_micro), 
			"macro = {:.5f}".format(f1_macro) )
		test_results.update({"{:.5f}_micro".format(label_percentage): f1_micro})
		test_results.update({"{:.5f}_macro".format(label_percentage): f1_macro})
	


	k = 10
	k_fold_roc, k_fold_f1, k_fold_precision, k_fold_recall = \
		evaluate_kfold_label_classification(embedding, node_labels, k=k)

	test_results.update(
		{
		f"{k}-fold-roc": k_fold_roc, 
		f"{k}-fold-f1": k_fold_f1,
		f"{k}-fold-precision": k_fold_precision,
		f"{k}-fold-recall": k_fold_recall,
		}
	)

	print (k, "-fold F1:", k_fold_f1)

	print (f"saving test results to {test_results_filename}")

	test_results = pd.Series(test_results)
	with open(test_results_filename, "wb") as f:
		pkl.dump(test_results, f, pkl.HIGHEST_PROTOCOL)

	print ("done")
	
if __name__ == "__main__":
	main()