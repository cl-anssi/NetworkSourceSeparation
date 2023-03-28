import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, find

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

try:
	import torch
	from torch_sparse import transpose, spmm
except ModuleNotFoundError:
	print('Torch backend unavailable (missing dependencies)')




MARGIN = 1e-10

def safediv(x, y):
	'''
	Safe division (avoids division by zero).


	Arguments
	---------
	X : scalar, array or tensor
		Dividend.
	Y : scalar, array or tensor
		Divisor.

	Returns
	-------
	ratio : scalar, array or tensor
		Result of the division.

	'''

	return x / (y + MARGIN)


class BipartiteSuperposedNMF(BaseEstimator):
	'''
	Estimator for the bipartite Superposed Nonnegative Matrix
	Factorization model with NumPy backend.


	Parameters
	----------
	n_estimators : int, default=2
		Number of activity sources to infer.
	dimension : int, default=10
		Embedding dimension of the NMF model for each activity source.
	epsilon : float, default=1e-4
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the loss function after each iteration is greater than
		epsilon.
	max_iter : int, default=200
		Maximum number of iterations in the inference procedure.
	l2_coeff_UV : float, default=0
		L2 regularization coefficient for the embedding matrices.
	lasso_coeff_W : float, default=0
		L1 regularization coefficient for the mixing matrix.
	rescale : str, default='binary'
		Method to use to rescale the occurrence counts of the edges.
		Possible values:
		- none: keep the occurrence counts unchanged
		- binary: ignore the counts and use binary values only
		- log: use a logarithmic scale
		- degree: divide the count by the square root of the product
		  of the mean degrees of the endpoints of the edge over the
		  training set
	chunk_size : int, default=64
		Size of the chunks of data used for computing the
		multiplicative updates for the mixing coefficients.
		Large values lead to greater parallelization but higher memory
		usage.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of the inference procedure.
		If verbose >= 2, a message is displayed after each iteration of
		the inference procedure.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	nodelist : tuple
		Two lists containing the identifiers of the top and bottom
		nodes, respectively.
	top_enc : LabelEncoder
		Label encoder for the top nodes.
		This attribute is set when fitting the model.
	bottom_enc : LabelEncoder
		Label encoder for the bottom nodes.
		This attribute is set when fitting the model.
	timestamps : list
		Timestamps for which historical mixing coefficients are
		available.
		This attribute is set when fitting the model.
	scores : list
		Value of the loss function at the end of each iteration of the
		inference procedure.
		This attribute is set when fitting the model.
	N : int
		Number of top nodes.
		This attribute is set when fitting the model.
	M : int
		Number of bottom nodes.
		This attribute is set when fitting the model.
	U : array of shape (n_estimators, n_top_nodes, embedding_dim)
		Embedding matrices for the top nodes.
		This attribute is set when fitting the model.
	V : array of shape (n_estimators, n_bottom_nodes, embedding_dim)
		Embedding matrices for the bottom nodes.
		This attribute is set when fitting the model.
	W : array of shape (n_timestamps, n_estimators)
		Array containing the mixing coefficients for each timestamp.
		This attribute is set when fitting the model.

	'''

	def __init__(
			self,
			n_estimators=2,
			dimension=10,
			epsilon=1e-4,
			max_iter=200,
			l2_coeff_UV=0,
			lasso_coeff_W=0,
			rescale='binary',
			chunk_size=64,
			verbose=0,
			random_state=None
		):

		self.n_estimators = n_estimators
		self.dimension = dimension
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.l2_coeff_UV = l2_coeff_UV
		self.lasso_coeff_W = lasso_coeff_W
		self.rescale = rescale
		self.chunk_size = chunk_size
		self.verbose = verbose
		self.random_state = random_state


	def fit(self, X, y=None, nodelist=None):
		'''
		Fit the estimator.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.
			The nodes can be identified by strings or integers.
			They are encoded using a LabelEncoder.
		y : not used, included for consistency with the scikit-learn
			API.
		nodelist : tuple or None, default=None
			Two lists containing the identifiers of the top and bottom
			nodes, respectively.
			If None, the list of top and bottom nodes is inferred from
			the dataset.

		Returns
		-------
		self : object
			Fitted estimator.

		'''

		self.rnd = check_random_state(self.random_state)
		adj_mat = self._make_dataset(X, nodelist=nodelist)
		self.scores = self._fit_params(adj_mat)
		return self


	def fit_and_append_weights(self, X):
		'''
		Computes the optimal mixing coefficients for each time step in
		the input data, then appends them to the model's sequence of
		historical mixing coefficients.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.

		'''

		adj_mat = self._build_adjacency_matrices(X)
		W = self._fit_weights(adj_mat)
		for t in sorted(list(set(X[:, 0].astype(int)))):
			self.timestamps.append(t)
		self._append_weights(W)


	def score_samples(self, X, order=1, period=None):
		'''
		Returns the opposite of the reconstruction error for each
		temporal edge in the input data.
		High reconstruction error (i.e. low score) means anomalous
		temporal edge.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.
		order : int, default=1
			Order of the reconstruction error, i.e., the error is
			(true value - expected value) ** order.
		period : int or None, default=None
			If None, the optimal mixing coefficients for each time step
			are computed and used.
			Otherwise, the mixing coefficients are predicted using a
			seasonal model with the given period.

		Returns
		-------
		y : array of shape (n_edges,)
			Opposite of the reconstruction error for each edge.

		'''

		adj_mat = self._build_adjacency_matrices(X)
		timestamps = dict(zip(
			sorted(list(set(X[:, 0].astype(int)))),
			range(len(set(X[:, 0])))
		))
		if period is None:
			W = self._fit_weights(adj_mat)
		else:
			W = self._periodic_weights(timestamps, period)
		R = self._reconstruction_error(adj_mat, W, order=order)
		inputs = self._encode_inputs(X)
		y = np.array([
			R[timestamps[t]][x, y]
			for t, x, y in inputs
		])
		return -y


	def _append_weights(self, W):
		'''
		Appends the given array of mixing coefficients to the model's
		sequence of historical mixing coefficients.


		Arguments
		---------
		W : array of shape (n_timestamps, n_estimators)
			Array of mixing coefficients.

		'''

		self.W = np.concatenate(
			[self.W, W],
			axis=0
		)


	def _build_adjacency_matrices(self, X):
		'''
		Converts an array of temporal edges into a list of sparse
		adjacency matrices.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.

		Returns
		-------
		adj_mat : list
			Sorted list of CSR matrices.
			The i-th matrix is the adjacency matrix of the i-th graph
			in the input data, with coefficients corresponding to
			occurrence counts rescaled using the method specified by
			the self.rescale attribute.

		'''

		df = pd.DataFrame({
			'timestamp': X[:, 0].astype(int),
			'src': X[:, 1],
			'dst': X[:, 2],
			'val': X[:, 3].astype(float)
		})
		idx = df.groupby(['timestamp']).groups
		timestamps = sorted(list(set(df['timestamp'])))
		adj_mat = []
		for t in timestamps:
			sub = df.loc[idx[t], :]
			A = self._make_adjacency_matrix(
				sub['src'],
				sub['dst'],
				sub['val']
			)
			adj_mat.append(A)
		return adj_mat


	def _encode_inputs(self, X):
		'''
		Encodes an array of temporal edges by replacing node
		identifiers with their integer IDs.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.

		Returns
		-------
		X_encoded : array of shape (n_edges, 3)
			Array of temporal edges.
			Each row contains three values: timestamp, top node ID,
			and bottom node ID.

		'''

		inputs = np.stack([
				X[:, 0].astype(int),
				self.top_enc.transform(X[:, 1]),
				self.bottom_enc.transform(X[:, 2])
			],
			axis=1
		)
		return inputs


	def _fit_params(self, adj_mat):
		'''
		Implements the inference procedure.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			to use for training.

		Returns
		-------
		scores : list
			Successive values of the training objective at each
			iteration of the inference procedure.

		'''

		self._initialize(adj_mat)
		old_score, diff, n_iter = -1e10, 10, 0
		scores = []
		self._log(
			'Starting training; '
			f'K={self.U.shape[2]}, L={self.U.shape[0]}, '
			f'lambda1={self.lasso_coeff_W}, lambda2={self.l2_coeff_UV}',
			1
		)
		while diff > self.epsilon and n_iter < self.max_iter:
			self.W *= self._multiplicative_update_W(adj_mat, self.W)
			self.U *= self._multiplicative_update_U(adj_mat)
			self.V *= self._multiplicative_update_V(adj_mat)
			score = self._validation_score(adj_mat, self.W)
			scores.append(score)
			diff = np.abs(1 - score / old_score)
			old_score = score
			n_iter += 1
			self._log(f'Iteration {n_iter}; score: {score}', 2)
		self._log(
			f'Training stopped after {n_iter} iterations; '
			f'final score: {score}',
			1
		)
		return scores


	def _fit_weights(self, adj_mat):
		'''
		Computes the optimal mixing coefficients for each time step in
		the input data.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			for which mixing coefficients should be inferred.

		Returns
		-------
		W : array of shape (n_timestamps, n_estimators)
			Inferred mixing coefficients.

		'''

		W = self._initialize_weights(len(adj_mat))
		old_score, diff, n_iter = -1e10, 10, 0
		while diff > self.epsilon and n_iter < self.max_iter:
			W *= self._multiplicative_update_W(adj_mat, W)
			score = self._validation_score(adj_mat, W)
			diff, old_score = np.abs(1 - score / old_score), score
			n_iter += 1
			self._log(f'Iteration {n_iter}; score: {score}', 2)
		return W


	def _get_degrees(self, edges):
		'''
		Returns the degrees of the endpoints of each edge in the input
		data.


		Arguments
		---------
		edges : array of shape (n_edges, 2)
			Sequence of edges, where each edge is defined by the
			integer IDs of its endpoints.

		Returns
		-------
		top_deg : array of shape (n_edges,)
			Degrees of the top nodes.
		bottom_deg : array of shape (n_edges,)
			Degrees of the bottom nodes.

		'''

		top_deg = self.top_degrees[edges[:, 0]]
		bottom_deg = self.bottom_degrees[edges[:, 1]]
		return top_deg, bottom_deg


	def _initialize(self, adj_mat):
		'''
		Randomly initializes model parameters.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			for which parameters are initialized.

		'''

		t = len(adj_mat)
		self.W = self._initialize_weights(t)
		self.U = self.rnd.rand(
			self.n_estimators,
			self.N,
			self.dimension
		)
		self.V = self.rnd.rand(
			self.n_estimators,
			self.M,
			self.dimension
		)


	def _initialize_weights(self, num_timestamps):
		'''
		Generates random mixing coefficients.


		Arguments
		---------
		num_timestamps : int
			Number of time steps for which mixing coefficients should
			be generated.

		Returns
		-------
		W : array of shape (num_timestamps, n_estimators)
			Mixing coefficients.

		'''

		W = self.rnd.rand(num_timestamps, self.n_estimators)
		return W


	def _log(self, message, level=0):
		'''
		Helper function.
		Prints given message if the verbosity level is high enough.


		Arguments
		---------
		message : str
			Message to print.
		level : int, default=0
			The message is printed only if self.verbose >= level.

		'''

		if self.verbose >= level:
			print(message)


	def _make_adjacency_matrix(self, top_nodes, bottom_nodes, values):
		'''
		Build a sparse adjacency matrix from lists of edge endpoints
		and occurrence counts.


		Arguments
		---------
		top_nodes : array of shape (n_edges,) or list
			Top nodes of the edges.
		bottom_nodes : array of shape (n_edges,) or list
			Bottom nodes of the edges.
		values : array of shape (n_edges,)
			Occurrence counts of the edges.

		Return
		------
		A : sparse matrix of shape (n_top_nodes, n_bottom_nodes)
			Adjacency matrix in CSR format.

		'''

		A = csr_matrix(
			(
				self._rescale(
					values,
					np.stack([
							self.top_enc.transform(top_nodes),
							self.bottom_enc.transform(bottom_nodes)
						], axis=1
					)
				),
				(
					self.top_enc.transform(top_nodes),
					self.bottom_enc.transform(bottom_nodes)
				)
			),
			shape=(self.N, self.M)
		)
		return A


	def _make_dataset(self, X, nodelist=None):
		'''
		Converts an array of temporal edges into a list of sparse
		adjacency matrices and extracts some properties of the dataset.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, top node, bottom
			node, and number of occurrences.
		nodelist : tuple or None, default=None
			Two lists containing the identifiers of the top and bottom
			nodes, respectively.
			If None, the list of top and bottom nodes is inferred from
			the dataset.

		Returns
		-------
		adj_mat : list
			Sorted list of CSR matrices.
			The i-th matrix is the adjacency matrix of the i-th graph
			in the input data, with coefficients corresponding to
			occurrence counts rescaled using the method specified by
			the self.rescale attribute.

		'''

		if nodelist is None:
			self.nodelist = (
				list(set(X[:, 1])),
				list(set(X[:, 2]))
			)
		else:
			self.nodelist = nodelist
		top, bottom = self.nodelist
		self.top_enc = LabelEncoder().fit(top)
		self.bottom_enc = LabelEncoder().fit(bottom)
		self.N = len(top)
		self.M = len(bottom)
		self.timestamps = sorted(list(set(X[:, 0].astype(int))))
		if self.rescale == 'degree':
			X_enc = np.stack([
					X[:, 0],
					self.top_enc.transform(X[:, 1]),
					self.bottom_enc.transform(X[:, 2]),
					X[:, 3]
				], axis=1
			)
			df = pd.DataFrame(
				X_enc,
				columns=('ts', 'top', 'bot', 'cnt')
			)
			cnt = df.groupby(['top'])['cnt'].sum().to_dict()
			self.top_degrees = np.array([
				cnt[u] / len(self.timestamps)
				if u in cnt else 1
				for u in self.top_enc.classes_
			])
			cnt = df.groupby(['bot'])['cnt'].sum().to_dict()
			self.bottom_degrees = np.array([
				cnt[v] / len(self.timestamps)
				if v in cnt else 1
				for v in self.bottom_enc.classes_
			])
		return self._build_adjacency_matrices(X)


	def _multiplicative_update_U(self, adj_mat):
		'''
		Computes the multiplicative update for the embeddings of the
		top nodes.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_U : array of shape
		                (n_estimators, n_top_nodes, dimension)
		    Multiplicative update to apply to the embeddings of the
		    top nodes.

		'''

		U1 = np.zeros_like(self.U)
		for t, Y in enumerate(adj_mat):
			for l in range(self.n_estimators):
				U1[l, :, :] += self.W[t, l] * Y.dot(self.V[l, :, :])
		U2 = np.zeros_like(U1)
		for l in range(self.n_estimators):
			prod_mat = [
				np.linalg.multi_dot([
					self.U[i, :, :],
					self.V[i, :, :].T,
					self.V[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				U2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
		T, M, K = self.W.shape[0], self.V.shape[1], self.V.shape[2]
		U2 += self.l2_coeff_UV * T * M / (K * 2) * self.U
		return safediv(U1, U2)


	def _multiplicative_update_V(self, adj_mat):
		'''
		Computes the multiplicative update for the embeddings of the
		bottom nodes.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_V : array of shape
		                (n_estimators, n_bottom_nodes, dimension)
		    Multiplicative update to apply to the embeddings of the
		    bottom nodes.

		'''

		V1 = np.zeros_like(self.V)
		for t, Y in enumerate(adj_mat):
			for l in range(self.n_estimators):
				V1[l, :, :] += self.W[t, l] * Y.T.dot(self.U[l, :, :])
		V2 = np.zeros_like(V1)
		for l in range(self.n_estimators):
			prod_mat = [
				np.linalg.multi_dot([
					self.V[i, :, :],
					self.U[i, :, :].T,
					self.U[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				V2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		V2 += self.l2_coeff_UV * T * N / (K * 2) * self.V
		return safediv(V1, V2)


	def _multiplicative_update_W(self, adj_mat, W):
		'''
		Computes the multiplicative update for the mixing coefficients.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.
		W : array of shape (n_timestamps, n_estimators)
			Current mixing coefficients.

		Returns
		-------
		mult_update_W : array of shape (n_timestamps, n_estimators)
		    Multiplicative update to apply to the mixing coefficients.

		'''

		W1 = np.stack([
			np.array([
				(self.U[l, :, :] * Y.dot(self.V[l, :, :])).sum()
				for l in range(self.n_estimators)
			])
			for Y in adj_mat
		], axis=0)
		prod_mat = np.zeros((self.n_estimators, self.n_estimators))
		for j in range(0, self.V.shape[1], self.chunk_size):
			tmp = np.matmul(
				self.U,
				self.V[:, j:j + self.chunk_size, :].transpose((0, 2, 1))
			)
			prod_mat += np.stack([
					np.einsum('lij,ij->l', tmp, tmp[l, :, :])
					for l in range(self.n_estimators)
				],
				axis=0
			)
		W2 = np.matmul(W, prod_mat)
		N, M = self.U.shape[1], self.V.shape[1]
		W_reg = self.lasso_coeff_W
		W2 += N * M / 2 * W_reg
		return safediv(W1, W2)


	def _periodic_weights(self, timestamps, period):
		'''
		Predicts the mixing coefficients for the given timestamps using
		a simple seasonal model.


		Arguments
		---------
		timestamps : dict
			Timestamps for which the mixing coefficients should be
			predicted.
			The keys are the timestamps, and the values are the
			corresponding indices in the array of mixing coefficients
			to be built.
		period : int
			Period of the seasonal model.

		Return
		------
		W : array of shape (n_timestamps, n_estimators)
			Predicted mixing coefficients.

		'''

		W = [None for t in timestamps]
		for t in timestamps:
			W[timestamps[t]] = np.stack(
				[
					self.W[i, :]
					for i, s in enumerate(self.timestamps)
					if (t - s) % period == 0
				],
				axis=0
			).mean(0)
		return np.stack(W, axis=0)


	def _reconstruction_error(self, adj_mat, weights, order=2):
		'''
		Computes the reconstruction error of the model for the given
		adjacency matrices.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices.
		weights : array of shape (len(adj_mat), n_estimators)
			Mixing coefficients to use for prediction.
		order : int, default=2
			Order of the reconstruction error, i.e., the error is
			(true value - expected value) ** order.

		Returns
		-------
		R : list
			List of CSR matrices containing the reconstruction error
			for each observed edge.

		'''

		R = []
		for A, w in zip(adj_mat, weights):
			xs, ys, val = find(A)
			rec = sum(
				w[l] * np.einsum(
					'ij,ij->i',
					self.U[l, xs, :],
					self.V[l, ys, :]
				)
				for l in range(self.n_estimators)
			)
			R.append(
				csr_matrix(
					(
						np.power(val - rec, order),
						(xs, ys)
					)
				)
			)
		return R


	def _rescale(self, values, edges):
		'''
		Rescales edge occurrence counts using the method specified by
		self.rescale.


		Arguments
		---------
		values : array of shape (n_edges,)
			Occurrence counts of the edges.
		edges : array of shape (n_edges, 2)
			Integer IDs of the endpoints of the edges.

		Returns
		-------
		rescaled : array of shape (n_edges,)
			Rescaled occurrence counts.

		'''

		if self.rescale == 'log':
			return np.log(1 + values)
		elif self.rescale == 'none':
			return values
		elif self.rescale == 'binary':
			return np.ones_like(values)
		elif self.rescale == 'degree':
			top_deg, bottom_deg = self._get_degrees(edges)
			denom = np.sqrt(top_deg * bottom_deg)
			return safediv(values, denom)
		else:
			raise ValueError(f'Unknown rescaling method: {self.rescale}')


	def _validation_score(self, adj_mat, weights):
		'''
		Computes the mean squared error of the model for the given
		adjacency matrices.

		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices.
		weights : array of shape (len(adj_mat), n_estimators)
			Mixing coefficients to use for prediction.

		Returns
		-------
		score : float
			Mean squared error for the given adjacency matrices.

		'''

		R = self._reconstruction_error(adj_mat, weights, order=2)
		return sum(find(r)[2].mean() for r in R) / len(R)


class SuperposedNMF(BipartiteSuperposedNMF):
	'''
	Estimator for the Superposed Nonnegative Matrix Factorization
	model with NumPy backend.


	Parameters
	----------
	n_estimators : int, default=2
		Number of activity sources to infer.
	dimension : int, default=10
		Embedding dimension of the NMF model for each activity source.
	epsilon : float, default=1e-4
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the loss function after each iteration is greater than
		epsilon.
	max_iter : int, default=200
		Maximum number of iterations in the inference procedure.
	l2_coeff_UV : float, default=0
		L2 regularization coefficient for the embedding matrices.
	lasso_coeff_W : float, default=0
		L1 regularization coefficient for the mixing matrix.
	rescale : str, default='binary'
		Method to use to rescale the occurrence counts of the edges.
		Possible values:
		- none: keep the occurrence counts unchanged
		- binary: ignore the counts and use binary values only
		- log: use a logarithmic scale
		- degree: divide the count by the square root of the product
		  of the mean degrees of the endpoints of the edge over the
		  training set
	chunk_size : int, default=64
		Size of the chunks of data used for computing the
		multiplicative updates for the mixing coefficients.
		Large values lead to greater parallelization but higher memory
		usage.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of the inference procedure.
		If verbose >= 2, a message is displayed after each iteration of
		the inference procedure.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	nodelist : list
		List containing the identifiers of the nodes.
	node_enc : LabelEncoder
		Label encoder for the nodes.
		This attribute is set when fitting the model.
	timestamps : list
		Timestamps for which historical mixing coefficients are
		available.
	scores : list
		Value of the loss function at the end of each iteration of the
		inference procedure.
		This attribute is set when fitting the model.
	N : int
		Number of nodes.
		This attribute is set when fitting the model.
	M : int
		Alias for self.N.
		This attribute is set when fitting the model.
	U : array of shape (n_estimators, n_nodes, embedding_dim)
		Origin embedding matrices.
		This attribute is set when fitting the model.
	V : array of shape (n_estimators, n_nodes, embedding_dim)
		Destination embedding matrices.
		This attribute is set when fitting the model.
	W : array of shape (n_timestamps, n_estimators)
		Array containing the mixing coefficients for each timestamp.
		This attribute is set when fitting the model.

	'''

	def _encode_inputs(self, X):
		'''
		Encodes an array of temporal edges by replacing node
		identifiers with their integer IDs.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, origin node,
			destination node, and number of occurrences.

		Returns
		-------
		X_encoded : array of shape (n_edges, 3)
			Array of temporal edges.
			Each row contains three values: timestamp, origin node ID,
			and destination node ID.

		'''

		inputs = np.stack([
				X[:, 0].astype(int),
				self.node_enc.transform(X[:, 1]),
				self.node_enc.transform(X[:, 2])
			],
			axis=1
		)
		return inputs


	def _get_degrees(self, edges):
		'''
		Returns the degrees of the endpoints of each edge in the input
		data.


		Arguments
		---------
		edges : array of shape (n_edges, 2)
			Sequence of edges, where each edge is defined by the
			integer IDs of its endpoints.

		Returns
		-------
		out_deg : array of shape (n_edges,)
			Out degrees of the origin nodes.
		in_deg : array of shape (n_edges,)
			In degrees of the destination nodes.

		'''

		out_deg = self.out_degrees[edges[:, 0]]
		in_deg = self.in_degrees[edges[:, 1]]
		return out_deg, in_deg


	def _make_adjacency_matrix(self, src_nodes, dst_nodes, values):
		'''
		Build a sparse adjacency matrix from lists of edge endpoints
		and occurrence counts.


		Arguments
		---------
		src_nodes : array of shape (n_edges,) or list
			Origin nodes of the edges.
		dst_nodes : array of shape (n_edges,) or list
			Destination nodes of the edges.
		values : array of shape (n_edges,)
			Occurrence counts of the edges.

		Return
		------
		A : sparse matrix of shape (n_nodes, n_nodes)
			Adjacency matrix in CSR format.

		'''

		A = csr_matrix(
			(
				self._rescale(
					values,
					np.stack([
							self.node_enc.transform(src_nodes),
							self.node_enc.transform(dst_nodes)
						], axis=1
					)
				),
				(
					self.node_enc.transform(src_nodes),
					self.node_enc.transform(dst_nodes)
				)
			),
			shape=(self.N, self.N)
		)
		return A


	def _make_dataset(self, X, nodelist=None):
		'''
		Converts an array of temporal edges into a list of sparse
		adjacency matrices and extracts some properties of the dataset.


		Arguments
		---------
		X : array of shape (n_edges, 4)
			Array of temporal edges.
			Each row contains four values: timestamp, origin node,
			destination, and number of occurrences.
		nodelist : list or None, default=None
			List containing the identifiers of the nodes.
			If None, the node list is inferred from the dataset.

		Returns
		-------
		adj_mat : list
			Sorted list of CSR matrices.
			The i-th matrix is the adjacency matrix of the i-th graph
			in the input data, with coefficients corresponding to
			occurrence counts rescaled using the method specified by
			the self.rescale attribute.

		'''

		if nodelist is None:
			self.nodelist = list(
				set(X[:, 1]).union(set(X[:, 2]))
			)
		else:
			self.nodelist = nodelist
		self.node_enc = LabelEncoder().fit(self.nodelist)
		self.N = len(self.nodelist)
		self.M = self.N
		self.timestamps = sorted(list(set(X[:, 0].astype(int))))
		if self.rescale == 'degree':
			X_enc = np.stack([
					X[:, 0],
					self.node_enc.transform(X[:, 1]),
					self.node_enc.transform(X[:, 2]),
					X[:, 3]
				], axis=1
			)
			df = pd.DataFrame(
				X_enc,
				columns=('ts', 'src', 'dst', 'cnt')
			)
			cnt = df.groupby(['src'])['cnt'].sum().to_dict()
			self.out_degrees = np.array([
				cnt[u] / len(self.timestamps)
				if u in cnt else 1
				for u in self.node_enc.classes_
			])
			cnt = df.groupby(['dst'])['cnt'].sum().to_dict()
			self.in_degrees = np.array([
				cnt[u] / len(self.timestamps)
				if u in cnt else 1
				for u in self.node_enc.classes_
			])
		return self._build_adjacency_matrices(X)


	def _multiplicative_update_U(self, adj_mat):
		'''
		Computes the multiplicative update for the origin embedding
		matrices.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_U : array of shape
		                (n_estimators, n_nodes, dimension)
		    Multiplicative update to apply to the origin embedding
		    matrices.

		'''

		U1 = np.zeros_like(self.U)
		for t, Y in enumerate(adj_mat):
			for l in range(self.n_estimators):
				U1[l, :, :] += self.W[t, l] * Y.dot(self.V[l, :, :])
		U2 = np.zeros_like(U1)
		diag_prod = [
			np.einsum(
				'ik,ik->i',
				self.U[l, :, :],
				self.V[l, :, :]
			)
			for l in range(self.n_estimators)
		]
		for l in range(self.n_estimators):
			prod_mat = [
				np.linalg.multi_dot([
					self.U[i, :, :],
					self.V[i, :, :].T,
					self.V[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				U2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
				corr = sum(
					self.W[t, i] * mat[:, np.newaxis]
					for i, mat in enumerate(diag_prod)
				)
				U2[l, :, :] -= self.W[t, l] * self.V[l, :, :] * corr
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		U2 += self.l2_coeff_UV * T * (N - 1) / (K * 2) * self.U
		return safediv(U1, U2)


	def _multiplicative_update_V(self, adj_mat):
		'''
		Computes the multiplicative update for the destination
		embedding matrices.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_V : array of shape
		                (n_estimators, n_nodes, dimension)
		    Multiplicative update to apply to the destination embedding
		    matrices.

		'''

		V1 = np.zeros_like(self.V)
		for t, Y in enumerate(adj_mat):
			for l in range(self.n_estimators):
				V1[l, :, :] += self.W[t, l] * Y.T.dot(self.U[l, :, :])
		V2 = np.zeros_like(V1)
		diag_prod = [
			np.einsum(
				'ik,ik->i',
				self.U[l, :, :],
				self.V[l, :, :]
			)
			for l in range(self.n_estimators)
		]
		for l in range(self.n_estimators):
			prod_mat = [
				np.linalg.multi_dot([
					self.V[i, :, :],
					self.U[i, :, :].T,
					self.U[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				V2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
				corr = sum(
					self.W[t, i] * mat[:, np.newaxis]
					for i, mat in enumerate(diag_prod)
				)
				V2[l, :, :] -= self.W[t, l] * self.U[l, :, :] * corr
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		V2 += self.l2_coeff_UV * T * (N - 1) / (K * 2) * self.V
		return safediv(V1, V2)


	def _multiplicative_update_W(self, adj_mat, W):
		'''
		Computes the multiplicative update for the mixing coefficients.


		Arguments
		---------
		adj_mat : list
			List of CSR matrices representing the adjacency matrices
			used for training.
		W : array of shape (n_timestamps, n_estimators)
			Current mixing coefficients.

		Returns
		-------
		mult_update_W : array of shape (n_timestamps, n_estimators)
		    Multiplicative update to apply to the mixing coefficients.

		'''

		W1 = np.stack([
			np.array([
				(self.U[l, :, :] * Y.dot(self.V[l, :, :])).sum()
				for l in range(self.n_estimators)
			])
			for Y in adj_mat
		], axis=0)
		prod_mat = np.zeros((self.n_estimators, self.n_estimators))
		for j in range(0, self.V.shape[1], self.chunk_size):
			tmp = np.matmul(
				self.U,
				self.V[:, j:j + self.chunk_size, :].transpose((0, 2, 1))
			)
			tmp[
					:,
					range(j, j + tmp.shape[2]),
					range(tmp.shape[2])
				] = 0
			prod_mat += np.stack([
					np.einsum('lij,ij->l', tmp, tmp[l, :, :])
					for l in range(self.n_estimators)
				],
				axis=0
			)
		W2 = np.matmul(W, prod_mat)
		N = self.U.shape[1]
		W_reg = self.lasso_coeff_W
		W2 += N * (N - 1) / 2 * W_reg
		return safediv(W1, W2)


class TorchBipartiteSuperposedNMF(BipartiteSuperposedNMF):
	'''
	Estimator for the bipartite Superposed Nonnegative Matrix
	Factorization model with PyTorch backend.


	Parameters
	----------
	n_estimators : int, default=2
		Number of activity sources to infer.
	dimension : int, default=10
		Embedding dimension of the NMF model for each activity source.
	epsilon : float, default=1e-4
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the loss function after each iteration is greater than
		epsilon.
	max_iter : int, default=200
		Maximum number of iterations in the inference procedure.
	l2_coeff_UV : float, default=0
		L2 regularization coefficient for the embedding matrices.
	lasso_coeff_W : float, default=0
		L1 regularization coefficient for the mixing matrix.
	rescale : str, default='binary'
		Method to use to rescale the occurrence counts of the edges.
		Possible values:
		- none: keep the occurrence counts unchanged
		- binary: ignore the counts and use binary values only
		- log: use a logarithmic scale
		- degree: divide the count by the square root of the product
		  of the mean degrees of the endpoints of the edge over the
		  training set
	chunk_size : int, default=64
		Size of the chunks of data used for computing the
		multiplicative updates for the mixing coefficients.
		Large values lead to greater parallelization but higher memory
		usage.
	device : str, default='cpu'
		Name of the device used for PyTorch tensor operations.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of the inference procedure.
		If verbose >= 2, a message is displayed after each iteration of
		the inference procedure.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	nodelist : tuple
		Two lists containing the identifiers of the top and bottom
		nodes, respectively.
	top_enc : LabelEncoder
		Label encoder for the top nodes.
		This attribute is set when fitting the model.
	bottom_enc : LabelEncoder
		Label encoder for the bottom nodes.
		This attribute is set when fitting the model.
	timestamps : list
		Timestamps for which historical mixing coefficients are
		available.
		This attribute is set when fitting the model.
	scores : list
		Value of the loss function at the end of each iteration of the
		inference procedure.
		This attribute is set when fitting the model.
	N : int
		Number of top nodes.
		This attribute is set when fitting the model.
	M : int
		Number of bottom nodes.
		This attribute is set when fitting the model.
	U : tensor of shape (n_estimators, n_top_nodes, embedding_dim)
		Embedding matrices for the top nodes.
		This attribute is set when fitting the model.
	V : tensor of shape (n_estimators, n_bottom_nodes, embedding_dim)
		Embedding matrices for the bottom nodes.
		This attribute is set when fitting the model.
	W : tensor of shape (n_timestamps, n_estimators)
		Tensor containing the mixing coefficients for each timestamp.
		This attribute is set when fitting the model.

	'''

	def __init__(
			self,
			n_estimators=2,
			dimension=2,
			epsilon=1e-4,
			max_iter=200,
			l2_coeff_UV=0,
			lasso_coeff_W=0,
			rescale='binary',
			chunk_size=64,
			device='cpu',
			verbose=0,
			random_state=None
		):

		self.n_estimators = n_estimators
		self.dimension = dimension
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.l2_coeff_UV = l2_coeff_UV
		self.lasso_coeff_W = lasso_coeff_W
		self.rescale = rescale
		self.chunk_size = chunk_size
		self.device = torch.device(device)
		self.verbose = verbose
		self.random_state = random_state


	def _append_weights(self, W):
		'''
		Appends the given tensor of mixing coefficients to the model's
		sequence of historical mixing coefficients.


		Arguments
		---------
		W : tensor of shape (n_timestamps, n_estimators)
			Tensor of mixing coefficients.

		'''

		self.W = torch.cat(
			[self.W, W],
			axis=0
		)


	def _initialize(self, adj_mat):
		'''
		Randomly initializes model parameters.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors matrices representing the adjacency
			matrices for which parameters are initialized.

		'''

		t = len(adj_mat)
		self.W = self._initialize_weights(t)
		self.U = torch.from_numpy(
				self.rnd.rand(
					self.n_estimators,
					self.N,
					self.dimension
				).astype(np.float32)
			).to(self.device)
		self.V = torch.from_numpy(
				self.rnd.rand(
					self.n_estimators,
					self.M,
					self.dimension
				).astype(np.float32)
			).to(self.device)


	def _initialize_weights(self, num_timestamps):
		'''
		Generates random mixing coefficients.


		Arguments
		---------
		num_timestamps : int
			Number of time steps for which mixing coefficients should
			be generated.

		Returns
		-------
		W : tensor of shape (num_timestamps, n_estimators)
			Mixing coefficients.

		'''

		W = self.rnd.rand(
				num_timestamps,
				self.n_estimators
			).astype(np.float32)
		return torch.from_numpy(W).to(self.device)


	def _make_adjacency_matrix(self, top_nodes, bottom_nodes, values):
		'''
		Build a sparse adjacency matrix from lists of edge endpoints
		and occurrence counts.


		Arguments
		---------
		top_nodes : array of shape (n_edges,) or list
			Top nodes of the edges.
		bottom_nodes : array of shape (n_edges,) or list
			Bottom nodes of the edges.
		values : array of shape (n_edges,)
			Occurrence counts of the edges.

		Return
		------
		coords : tensor of shape (2, n_edges)
			Endpoints of the edges.
		values : tensor of shape (n_edges,)
			Weights of the edges.

		'''

		edges = np.stack(
				[
					self.top_enc.transform(top_nodes),
					self.bottom_enc.transform(bottom_nodes)
				], axis=1
			).astype(int)
		rescaled = self._rescale(values, edges)
		return (
			torch.from_numpy(edges.T).to(self.device),
			torch.from_numpy(rescaled).to(self.device)
		)


	def _multiplicative_update_U(self, adj_mat):
		'''
		Computes the multiplicative update for the embeddings of the
		top nodes.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_U : tensor of shape
		                (n_estimators, n_top_nodes, dimension)
		    Multiplicative update to apply to the embeddings of the
		    top nodes.

		'''

		U1 = torch.zeros_like(self.U)
		for t, (coord, val) in enumerate(adj_mat):
			for l in range(self.n_estimators):
				U1[l, :, :] += self.W[t, l] * spmm(
					coord,
					val,
					self.N,
					self.M,
					self.V[l, :, :]
				)
		U2 = torch.zeros_like(U1)
		for l in range(self.n_estimators):
			prod_mat = [
				torch.linalg.multi_dot([
					self.U[i, :, :],
					self.V[i, :, :].T,
					self.V[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				U2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
		T, M, K = self.W.shape[0], self.V.shape[1], self.V.shape[2]
		U2 += self.l2_coeff_UV * T * M / (K * 2) * self.U
		return safediv(U1, U2)


	def _multiplicative_update_V(self, adj_mat):
		'''
		Computes the multiplicative update for the embeddings of the
		bottom nodes.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_V : tensor of shape
		                (n_estimators, n_bottom_nodes, dimension)
		    Multiplicative update to apply to the embeddings of the
		    bottom nodes.

		'''

		V1 = torch.zeros_like(self.V)
		for t, (coord, val) in enumerate(adj_mat):
			for l in range(self.n_estimators):
				V1[l, :, :] += self.W[t, l] * spmm(
					*transpose(
						coord,
						val,
						self.N,
						self.M
					),
					self.M,
					self.N,
					self.U[l, :, :]
				)
		V2 = torch.zeros_like(V1)
		for l in range(self.n_estimators):
			prod_mat = [
				torch.linalg.multi_dot([
					self.V[i, :, :],
					self.U[i, :, :].T,
					self.U[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				V2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		V2 += self.l2_coeff_UV * T * N / (K * 2) * self.V
		return safediv(V1, V2)


	def _multiplicative_update_W(self, adj_mat, W):
		'''
		Computes the multiplicative update for the mixing coefficients.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.
		W : tensor of shape (n_timestamps, n_estimators)
			Current mixing coefficients.

		Returns
		-------
		mult_update_W : tensor of shape (n_timestamps, n_estimators)
		    Multiplicative update to apply to the mixing coefficients.

		'''

		W1 = torch.tensor(
			[
				[
					(
						self.U[l, :, :] * spmm(
							coord,
							val,
							self.N,
							self.M,
							self.V[l, :, :]
						)
					).sum()
					for l in range(self.n_estimators)
				]
				for coord, val in adj_mat
			],
			device=self.device
		)
		prod_mat = torch.zeros(
				(self.n_estimators, self.n_estimators)
			).to(self.device)
		for j in range(0, self.V.shape[1], self.chunk_size):
			tmp = torch.matmul(
				self.U,
				self.V[:, j:j + self.chunk_size, :].transpose(1, 2)
			)
			prod_mat += torch.stack([
					torch.einsum('lij,ij->l', tmp, tmp[l, :, :])
					for l in range(self.n_estimators)
				],
				axis=0
			)
		W2 = torch.matmul(W, prod_mat)
		N, M = self.U.shape[1], self.V.shape[1]
		W_reg = self.lasso_coeff_W
		W2 += N * M / 2 * W_reg
		return safediv(W1, W2)


	def _periodic_weights(self, timestamps, period):
		'''
		Predicts the mixing coefficients for the given timestamps using
		a simple seasonal model.


		Arguments
		---------
		timestamps : dict
			Timestamps for which the mixing coefficients should be
			predicted.
			The keys are the timestamps, and the values are the
			corresponding indices in the array of mixing coefficients
			to be built.
		period : int
			Period of the seasonal model.

		Return
		------
		W : tensor of shape (n_timestamps, n_estimators)
			Predicted mixing coefficients.

		'''

		W = [None for t in timestamps]
		for t in timestamps:
			W[timestamps[t]] = torch.stack(
				[
					self.W[i, :]
					for i, s in enumerate(self.timestamps)
					if (t - s) % period == 0
				],
				axis=0
			).mean(0)
		return torch.stack(W, axis=0)


	def _reconstruction_error(self, adj_mat, weights, order=2):
		'''
		Computes the reconstruction error of the model for the given
		adjacency matrices.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices.
		weights : tensor of shape (len(adj_mat), n_estimators)
			Mixing coefficients to use for prediction.
		order : int, default=2
			Order of the reconstruction error, i.e., the error is
			(true value - expected value) ** order.

		Returns
		-------
		R : list
			List of CSR matrices containing the reconstruction error
			for each observed edge.

		'''

		R = []
		for (coord, val), w in zip(adj_mat, weights):
			rec = sum(
				w[l] * torch.einsum(
					'ij,ij->i',
					self.U[l, coord[0, :], :],
					self.V[l, coord[1, :], :]
				)
				for l in range(self.n_estimators)
			)
			xs = coord[0, :].cpu().numpy()
			ys = coord[1, :].cpu().numpy()
			R.append(
				csr_matrix(
					(
						torch.pow(val - rec, order).cpu().numpy(),
						(xs, ys)
					)
				)
			)
		return R


class TorchSuperposedNMF(SuperposedNMF):
	'''
	Estimator for the Superposed Nonnegative Matrix Factorization
	model with PyTorch backend.


	Parameters
	----------
	n_estimators : int, default=2
		Number of activity sources to infer.
	dimension : int, default=10
		Embedding dimension of the NMF model for each activity source.
	epsilon : float, default=1e-4
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the loss function after each iteration is greater than
		epsilon.
	max_iter : int, default=200
		Maximum number of iterations in the inference procedure.
	l2_coeff_UV : float, default=0
		L2 regularization coefficient for the embedding matrices.
	lasso_coeff_W : float, default=0
		L1 regularization coefficient for the mixing matrix.
	rescale : str, default='binary'
		Method to use to rescale the occurrence counts of the edges.
		Possible values:
		- none: keep the occurrence counts unchanged
		- binary: ignore the counts and use binary values only
		- log: use a logarithmic scale
		- degree: divide the count by the square root of the product
		  of the mean degrees of the endpoints of the edge over the
		  training set
	chunk_size : int, default=64
		Size of the chunks of data used for computing the
		multiplicative updates for the mixing coefficients.
		Large values lead to greater parallelization but higher memory
		usage.
	device : str, default='cpu'
		Name of the device used for PyTorch tensor operations.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of the inference procedure.
		If verbose >= 2, a message is displayed after each iteration of
		the inference procedure.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	nodelist : list
		List containing the identifiers of the nodes.
	node_enc : LabelEncoder
		Label encoder for the nodes.
		This attribute is set when fitting the model.
	timestamps : list
		Timestamps for which historical mixing coefficients are
		available.
	scores : list
		Value of the loss function at the end of each iteration of the
		inference procedure.
		This attribute is set when fitting the model.
	N : int
		Number of nodes.
		This attribute is set when fitting the model.
	M : int
		Alias for self.N.
		This attribute is set when fitting the model.
	U : tensor of shape (n_estimators, n_nodes, embedding_dim)
		Origin embedding matrices.
		This attribute is set when fitting the model.
	V : tensor of shape (n_estimators, n_nodes, embedding_dim)
		Destination embedding matrices.
		This attribute is set when fitting the model.
	W : tensor of shape (n_timestamps, n_estimators)
		Tensor containing the mixing coefficients for each timestamp.
		This attribute is set when fitting the model.

	'''

	def __init__(
			self,
			n_estimators=2,
			dimension=2,
			epsilon=1e-4,
			max_iter=200,
			l2_coeff_UV=0,
			lasso_coeff_W=0,
			rescale='binary',
			chunk_size=64,
			device='cpu',
			verbose=0,
			random_state=None
		):

		self.n_estimators = n_estimators
		self.dimension = dimension
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.l2_coeff_UV = l2_coeff_UV
		self.lasso_coeff_W = lasso_coeff_W
		self.rescale = rescale
		self.chunk_size = chunk_size
		self.device = torch.device(device)
		self.verbose = verbose
		self.random_state = random_state


	def _append_weights(self, W):
		'''
		Appends the given tensor of mixing coefficients to the model's
		sequence of historical mixing coefficients.


		Arguments
		---------
		W : tensor of shape (n_timestamps, n_estimators)
			Tensor of mixing coefficients.

		'''

		self.W = torch.cat(
			[self.W, W],
			axis=0
		)


	def _initialize(self, adj_mat):
		'''
		Randomly initializes model parameters.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors matrices representing the adjacency
			matrices for which parameters are initialized.

		'''

		t = len(adj_mat)
		self.W = self._initialize_weights(t)
		self.U = torch.from_numpy(
				self.rnd.rand(
					self.n_estimators,
					self.N,
					self.dimension
				).astype(np.float32)
			).to(self.device)
		self.V = torch.from_numpy(
				self.rnd.rand(
					self.n_estimators,
					self.N,
					self.dimension
				).astype(np.float32)
			).to(self.device)


	def _initialize_weights(self, num_timestamps):
		'''
		Generates random mixing coefficients.


		Arguments
		---------
		num_timestamps : int
			Number of time steps for which mixing coefficients should
			be generated.

		Returns
		-------
		W : tensor of shape (num_timestamps, n_estimators)
			Mixing coefficients.

		'''

		W = self.rnd.rand(
				num_timestamps,
				self.n_estimators
			).astype(np.float32)
		return torch.from_numpy(W).to(self.device)


	def _make_adjacency_matrix(self, src_nodes, dst_nodes, values):
		'''
		Build a sparse adjacency matrix from lists of edge endpoints
		and occurrence counts.


		Arguments
		---------
		src_nodes : array of shape (n_edges,) or list
			Origin nodes of the edges.
		dst_nodes : array of shape (n_edges,) or list
			Destination nodes of the edges.
		values : array of shape (n_edges,)
			Occurrence counts of the edges.

		Return
		------
		coords : tensor of shape (2, n_edges)
			Endpoints of the edges.
		values : tensor of shape (n_edges,)
			Weights of the edges.

		'''

		edges = np.stack(
				[
					self.node_enc.transform(src_nodes),
					self.node_enc.transform(dst_nodes)
				], axis=1
			).astype(int)
		rescaled = self._rescale(values, edges)
		return (
			torch.from_numpy(edges.T).to(self.device),
			torch.from_numpy(rescaled).to(self.device)
		)


	def _multiplicative_update_U(self, adj_mat):
		'''
		Computes the multiplicative update for the origin embedding
		matrices.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_U : tensor of shape
		                (n_estimators, n_nodes, dimension)
		    Multiplicative update to apply to the origin embedding
		    matrices.

		'''

		U1 = torch.zeros_like(self.U)
		for t, (coord, val) in enumerate(adj_mat):
			for l in range(self.n_estimators):
				U1[l, :, :] += self.W[t, l] * spmm(
					coord,
					val,
					self.N,
					self.N,
					self.V[l, :, :]
				)
		U2 = torch.zeros_like(U1)
		diag_prod = [
			torch.einsum(
				'ik,ik->i',
				self.U[l, :, :],
				self.V[l, :, :]
			)
			for l in range(self.n_estimators)
		]
		for l in range(self.n_estimators):
			prod_mat = [
				torch.linalg.multi_dot([
					self.U[i, :, :],
					self.V[i, :, :].T,
					self.V[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				U2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
				corr = sum(
					self.W[t, i] * mat[:, np.newaxis]
					for i, mat in enumerate(diag_prod)
				)
				U2[l, :, :] -= self.W[t, l] * self.V[l, :, :] * corr
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		U2 += self.l2_coeff_UV * T * (N - 1) / (K * 2) * self.U
		return safediv(U1, U2)


	def _multiplicative_update_V(self, adj_mat):
		'''
		Computes the multiplicative update for the destination
		embedding matrices.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.

		Returns
		-------
		mult_update_V : tensor of shape
		                (n_estimators, n_nodes, dimension)
		    Multiplicative update to apply to the destination embedding
		    matrices.

		'''

		V1 = torch.zeros_like(self.V)
		for t, (coord, val) in enumerate(adj_mat):
			for l in range(self.n_estimators):
				V1[l, :, :] += self.W[t, l] * spmm(
					*transpose(
						coord,
						val,
						self.N,
						self.N
					),
					self.N,
					self.N,
					self.U[l, :, :]
				)
		V2 = torch.zeros_like(V1)
		diag_prod = [
			torch.einsum(
				'ik,ik->i',
				self.U[l, :, :],
				self.V[l, :, :]
			)
			for l in range(self.n_estimators)
		]
		for l in range(self.n_estimators):
			prod_mat = [
				torch.linalg.multi_dot([
					self.V[i, :, :],
					self.U[i, :, :].T,
					self.U[l, :, :]
				])
				for i in range(self.n_estimators)
			]
			for t in range(self.W.shape[0]):
				V2[l, :, :] += self.W[t, l] * sum(
					self.W[t, i] * mat
					for i, mat in enumerate(prod_mat)
				)
				corr = sum(
					self.W[t, i] * mat[:, np.newaxis]
					for i, mat in enumerate(diag_prod)
				)
				V2[l, :, :] -= self.W[t, l] * self.U[l, :, :] * corr
		T, N, K = self.W.shape[0], self.U.shape[1], self.U.shape[2]
		V2 += self.l2_coeff_UV * T * (N - 1) / (K * 2) * self.V
		return safediv(V1, V2)


	def _multiplicative_update_W(self, adj_mat, W):
		'''
		Computes the multiplicative update for the mixing coefficients.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices
			used for training.
		W : tensor of shape (n_timestamps, n_estimators)
			Current mixing coefficients.

		Returns
		-------
		mult_update_W : tensor of shape (n_timestamps, n_estimators)
		    Multiplicative update to apply to the mixing coefficients.

		'''

		W1 = torch.tensor(
			[
				[
					(
						self.U[l, :, :] * spmm(
							coord,
							val,
							self.N,
							self.N,
							self.V[l, :, :]
						)
					).sum()
					for l in range(self.n_estimators)
				]
				for coord, val in adj_mat
			],
			device=self.device
		)
		prod_mat = torch.zeros(
				(self.n_estimators, self.n_estimators)
			).to(self.device)
		for j in range(0, self.V.shape[1], self.chunk_size):
			tmp = torch.matmul(
				self.U,
				self.V[:, j:j + self.chunk_size, :].transpose(1, 2)
			)
			tmp[
					:,
					range(j, j + tmp.shape[2]),
					range(tmp.shape[2])
				] = 0
			prod_mat += torch.stack([
					torch.einsum('lij,ij->l', tmp, tmp[l, :, :])
					for l in range(self.n_estimators)
				],
				axis=0
			)
		W2 = torch.matmul(W, prod_mat)
		N = self.U.shape[1]
		W_reg = self.lasso_coeff_W
		W2 += N * (N - 1) / 2 * W_reg
		return safediv(W1, W2)


	def _periodic_weights(self, timestamps, period):
		'''
		Predicts the mixing coefficients for the given timestamps using
		a simple seasonal model.


		Arguments
		---------
		timestamps : dict
			Timestamps for which the mixing coefficients should be
			predicted.
			The keys are the timestamps, and the values are the
			corresponding indices in the array of mixing coefficients
			to be built.
		period : int
			Period of the seasonal model.

		Return
		------
		W : tensor of shape (n_timestamps, n_estimators)
			Predicted mixing coefficients.

		'''

		W = [None for t in timestamps]
		for t in timestamps:
			W[timestamps[t]] = torch.stack(
				[
					self.W[i, :]
					for i, s in enumerate(self.timestamps)
					if (t - s) % period == 0
				],
				axis=0
			).mean(0)
		return torch.stack(W, axis=0)


	def _reconstruction_error(self, adj_mat, weights, order=2):
		'''
		Computes the reconstruction error of the model for the given
		adjacency matrices.


		Arguments
		---------
		adj_mat : list
			List of sparse tensors representing the adjacency matrices.
		weights : tensor of shape (len(adj_mat), n_estimators)
			Mixing coefficients to use for prediction.
		order : int, default=2
			Order of the reconstruction error, i.e., the error is
			(true value - expected value) ** order.

		Returns
		-------
		R : list
			List of CSR matrices containing the reconstruction error
			for each observed edge.

		'''

		R = []
		for (coord, val), w in zip(adj_mat, weights):
			rec = sum(
				w[l] * torch.einsum(
					'ij,ij->i',
					self.U[l, coord[0, :], :],
					self.V[l, coord[1, :], :]
				)
				for l in range(self.n_estimators)
			)
			xs = coord[0, :].cpu().numpy()
			ys = coord[1, :].cpu().numpy()
			R.append(
				csr_matrix(
					(
						torch.pow(val - rec, order).cpu().numpy(),
						(xs, ys)
					)
				)
			)
		return R

