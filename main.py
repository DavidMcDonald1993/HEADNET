import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import random
import numpy as np
import pandas as pd

from keras import backend as K

K.set_floatx("float64")
K.set_epsilon(np.float64(1e-15))

from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping

import tensorflow as tf

# configuration = tf.compat.v1.ConfigProto()
# configuration.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=configuration)


import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

KTF.set_session(sess)


from headnet.generators import TrainingDataGenerator
from headnet.callbacks import HEADNetCheckpointer
from headnet.models import build_headnet

from utils.io import load_data, load_weights
from utils.hyperbolic_functions import hyperboloid_to_poincare_ball
from utils.sampling import determine_positive_and_negative_samples
from utils.visualise import draw_graph


def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="HEADNET algorithm for feature learning on complex networks")

	parser.add_argument("--graph", dest="graph", type=str, default=None,
		help="path to graph to load.")
	parser.add_argument("--features", dest="features", type=str, default=None,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50,
		help="The number of epochs to train for (default is 50).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
		help="Batch size for training (default is 512).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--patience", dest="patience", type=int, default=5,
		help="The number of epochs of no improvement in loss before training is stopped. (Default is 5)")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 10).", default=10)

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	parser.add_argument("--embedding", dest="embedding_path", default=None, 
		help="path to save embedings.")

	parser.add_argument("--visualise", action="store_true", 
		help="flag to visualise embedding (embedding_dim must be 2)")

	parser.add_argument("--identity_variance", action="store_true",
		help="flag to fix the variance matrix to the identity matrix.")

	parser.add_argument("--euclidean_distance", action="store_true",
		help="flag to use euclidean distance in model building and training.")

	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''
	os.makedirs(args.embedding_path, exist_ok=True)
	print (f"saving embedding to {args.embedding_path}")

def main():

	args = parse_args()

	args.directed = True

	assert not (args.visualise and args.embedding_dim > 2), "Can only visualise two dimensions"
	assert args.embedding_path is not None, "you must specify a path to save embedding"

	if args.identity_variance:
		print ("using fixed identity variance")

	if args.euclidean_distance:
		print ("using euclidean distance")

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)


	graph_filename = args.graph
	features_filename = args.features
	labels_filename = args.labels

	graph, features, node_labels = load_data(
		graph_filename=graph_filename,
		features_filename=features_filename,
		labels_filename=labels_filename)
	if not args.visualise and node_labels is not None:
		node_labels = None
	print ("Loaded dataset")

	configure_paths(args)

	print ("Configured paths")

	positive_samples, negative_samples, node_map = \
		determine_positive_and_negative_samples(graph, args)

	# number of nodes
	N = graph.shape[0]

	if not args.visualise:
		del graph 

	# build model
	embedder, model = build_headnet(
		N,
		features, 
		args.embedding_dim, 
		args.num_negative_samples, 
		identity_variance=args.identity_variance,
		euclidean_distance=args.euclidean_distance,
		)
	model, initial_epoch = load_weights(
		model, 
		args.embedding_path)

	# print Keras summary to console
	model.summary()

	best_model_path = os.path.join(
		args.embedding_path, 
		"best_model.h5")

	callbacks = [
		TerminateOnNaN(),
		EarlyStopping(
			monitor="loss", 
			patience=args.patience, 
			mode="min",
			verbose=True),
		ModelCheckpoint(
			best_model_path,
			save_best_only=True,
			save_weights_only=True,
			monitor="loss",
			mode="min"),
		HEADNetCheckpointer(
			epoch=initial_epoch, 
			embedding_directory=args.embedding_path,
			model=model,
			embedder=embedder,
			features=features if features is not None else np.arange(N),)#.reshape(N, 1),)
	]			

	print (f"Training with data generator with {args.workers} worker threads")
	training_generator = TrainingDataGenerator(
		features,
		positive_samples,  
		negative_samples,
		node_map,
		args,
	)

	print ("Beginning model training")

	model.fit_generator(
		training_generator, 
		workers=args.workers,
		use_multiprocessing=False,
		steps_per_epoch=len(training_generator),
		epochs=args.num_epochs, 
		initial_epoch=initial_epoch, 
		verbose=args.verbose,
		callbacks=callbacks,
	)


	print ("Training complete")
	if os.path.exists(best_model_path):
		print ("Loading best model from", best_model_path)
		model.load_weights(best_model_path)

	print ("saving final embedding")

	if features is not None:
		embedding, sigmas = embedder.predict(features)
	else:
		embedding, sigmas = embedder.predict(np.arange(N))

		embedding = np.squeeze(embedding, 1)
		sigmas = np.squeeze(sigmas, 1)

	assert np.isfinite(embedding).all()
	assert np.isfinite(sigmas).all()

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

	if args.embedding_dim==2 and args.visualise:
		poincare_embedding = hyperboloid_to_poincare_ball(embedding)
		draw_graph(
			graph,
			poincare_embedding, 
			node_labels, 
			path="2d-poincare-disk-visualisation.png")


if __name__ == "__main__":
	main()