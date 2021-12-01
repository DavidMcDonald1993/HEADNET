import numpy as np

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + np.sum(np.square(X), axis=-1, keepdims=True)
	x = np.concatenate([x, t], axis=-1)
	return 1 / (1. - np.sum(np.square(X), axis=-1, keepdims=True)) * x

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return (np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) 
		- x[...,-1:] * y[...,-1:])

def minkowski_norm(x):
	return np.sqrt( np.maximum(minkowski_dot(x, x), 0.) )

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1:]

def poincare_ball_to_klein(X):
	norm = np.linalg.norm(X, axis=-1, keepdims=True)
	return 2 * norm / (1 + norm**2) * X / norm