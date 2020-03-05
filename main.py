from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import random
import numpy as np
import pandas as pd

from keras import backend as K

K.set_floatx("float64")
K.set_epsilon(np.float64(1e-15))

from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping

import tensorflow as tf

from headnet.utils import hyperboloid_to_poincare_ball, load_data
from headnet.utils import  determine_positive_and_negative_samples, load_weights
from headnet.generators import TrainingDataGenerator
from headnet.visualise import draw_graph, plot_degree_dist
from headnet.callbacks import Checkpointer
from headnet.models import build_hyperboloid_asym_model

from evaluation_utils import hyperbolic_distance_hyperboloid, hyperbolic_distance_poincare

np.set_printoptions(suppress=True)

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

config.log_device_placement=False
config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))



def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="HEADNET algorithm for feature learning on complex networks")

	parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, default=None,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	parser.add_argument("--lr", dest="lr", type=np.float64, default=.1,
		help="Learning rate (default is .1).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50,
		help="The number of epochs to train for (default is 50).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
		help="Batch size for training (default is 512).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=1,
		help="Context size for generating positive samples (default is 1).")
	parser.add_argument("--patience", dest="patience", type=int, default=3,
		help="The number of epochs of no improvement in loss before training is stopped. (Default is 3)")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 10).", default=10)

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	parser.add_argument("--embedding", dest="embedding_path", default=None, 
		help="path to save embedings.")

	parser.add_argument('--visualise', action="store_true", 
		help='flag to visualise embedding (embedding_dim must be 2)')

	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''
	if not os.path.exists(args.embedding_path):
		os.makedirs(args.embedding_path)
		print ("making {}".format(args.embedding_path))
	print ("saving embedding to {}".format(args.embedding_path))

def main():


	args = parse_args()

	args.directed = True

	assert args.context_size == 1

	assert not (args.visualise and args.embedding_dim > 2), "Can only visualise two dimensions"
	assert args.embedding_path is not None, "you must specify a path to save embedding"

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	graph, (features_train, features_all), node_labels = load_data(args)
	if not args.visualise and node_labels is not None:
		node_labels = None
	assert features_train is not None
	print ("Loaded dataset")

	if False:
		plot_degree_dist(graph, "degree distribution")

	configure_paths(args)

	print ("Configured paths")

	nodes = sorted(graph.nodes())

	# build model
	num_features = features_train.shape[1]
	
	embedder, model = build_hyperboloid_asym_model(num_features, 
		args.embedding_dim, 
		args.num_negative_samples, 
		lr=args.lr)
	model, initial_epoch = load_weights(model, args.embedding_path)

	model.summary()

	best_model_path = os.path.join(args.embedding_path, 
			"best_model.h5")

	callbacks = [
		TerminateOnNaN(),
		EarlyStopping(monitor="loss", 
			patience=args.patience, 
			mode="min",
			verbose=True),
		ModelCheckpoint(best_model_path,
			save_best_only=True,
			save_weights_only=True,
			monitor="loss",
			mode="min"),
		Checkpointer(epoch=initial_epoch, 
			nodes=nodes, 
			embedding_directory=args.embedding_path,
			model=model,
			embedder=embedder,
			features=features_train,
		)
	]			

	positive_samples, negative_samples = \
		determine_positive_and_negative_samples(graph, args)

	print ("Training with data generator with {} worker threads".format(args.workers))
	training_generator = TrainingDataGenerator(
		features_train,
		positive_samples,  
		negative_samples,
		model,
		args,
		graph
	)

	model.fit_generator(training_generator, 
		workers=args.workers,
		max_queue_size=10, 
		# use_multiprocessing=args.workers>0, 
		use_multiprocessing=False,
		epochs=args.num_epochs, 
		steps_per_epoch=len(training_generator),
		initial_epoch=initial_epoch, 
		verbose=args.verbose,
		callbacks=callbacks
	)

	print ("Training complete")
	if os.path.exists(best_model_path):
		print ("Loading best model from", best_model_path)
		model.load_weights(best_model_path)

	print ("saving final embedding")

	embedding, sigmas = embedder.predict(features_all)
	poincare_embedding = hyperboloid_to_poincare_ball(embedding)
	print ("embedding", np.linalg.norm(poincare_embedding.mean(0)))

	norms = np.linalg.norm(poincare_embedding, axis=-1)
	print ("norm min", norms.min(), "norm max", norms.max())
	print ("sigma min", sigmas.min(), "sigma max", sigmas.max())
		
	embedding_filename = os.path.join(args.embedding_path,
		"final_embedding.csv")
	print ("saving embedding to", embedding_filename)
	embedding_df = pd.DataFrame(embedding)
	embedding_df.to_csv(embedding_filename)

	variance_filename = os.path.join(args.embedding_path,
		"final_variance.csv")
	print ("saving variance to", variance_filename)
	variance_df = pd.DataFrame(sigmas)
	variance_df.to_csv(variance_filename)

	if args.visualise:
		draw_graph(graph,
			poincare_embedding, 
			node_labels, 
			path="2d-poincare-disk-visualisation.png")


if __name__ == "__main__":
	main()