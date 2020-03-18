from __future__ import print_function

import re
import os
import glob
import numpy as np
import pandas as pd

from keras.callbacks import Callback

from headnet.utils import hyperboloid_to_poincare_ball

class Checkpointer(Callback):

	def __init__(self, 
		epoch,
		nodes,
		embedding_directory,
		model,
		embedder,
		features,
		history=1
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory
		self.model = model
		self.embedder = embedder
		self.features = features
		self.history = history

	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		if self.epoch % 1 != 0:
			return
		print ("Epoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		embedding_directory = self.embedding_directory
		history = self.history
		# for old_model_path in sorted(
		# 	glob.iglob(os.path.join(self.embedding_directory, 
		# 		"[0-9]+_model.h5"))):
		old_model_paths = sorted(filter(
				re.compile("[0-9]+\_model\.h5").match, 
				os.listdir(embedding_directory)))
		if history > 0:
			old_model_paths = old_model_paths[:-history]
		for old_model_path in old_model_paths:
			print ("removing model: {}".format(old_model_path))
			os.remove(os.path.join(embedding_directory, 
				old_model_path))
				
	def save_model(self):

		weights_filename = os.path.join(self.embedding_directory, 
			"{:05d}_model.h5".format(self.epoch))
		self.model.save_weights(weights_filename)
		print ("saving weights to", weights_filename)
		
		embedding, variance = self.embedder.predict(self.features)

		embedding = hyperboloid_to_poincare_ball(embedding)
		print ("embedding", np.linalg.norm(embedding.mean(0)))
		ranks = np.linalg.norm(embedding, axis=-1)
		print ("ranks", ranks.min(), ranks.mean(),
			ranks.max() )
		# assert not np.any(np.isnan(embedding))
		# assert not np.any(np.isinf(embedding))
		# assert (embedding[:,-1] > 0).all(), embedding[:,-1]
		# assert np.allclose(minkowski_dot(embedding, embedding), -1)

		# print ("saving current embedding to {}".\
		# 	format(embedding_filename))
		# embedding_df = pd.DataFrame(embedding, index=self.nodes)
		# embedding_df.to_csv(embedding_filename, compression="gzip")

		# variance_filename = os.path.join(self.embedding_directory, 
		# 	"{:05d}_variance.csv.gz".format(self.epoch))
		print ("variance", variance.min(),
			variance.mean(), variance.max())

		# print ("saving current variance to {}".\
		# 	format(variance_filename))
		# variance_df = pd.DataFrame(variance, index=self.nodes)
		# variance_df.to_csv(variance_filename, compression="gzip")
		# print()
