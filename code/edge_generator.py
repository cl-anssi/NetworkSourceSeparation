import numpy as np
import pandas as pd

from sklearn.utils import check_random_state




class EdgeGenerator:
	'''
	Implements the generation of the three types of negative edges
	defined in [1].


	Parameters
	----------
	X_train : array of shape (n_training_edges, 4)
		Array of temporal edges representing the training set.
		Each row contains four values: timestamp, origin node,
		destination node, and number of occurrences.
	X_test : array of shape (n_test_edges, 5)
		Array of temporal edges representing the test set.
		Each row contains five values: timestamp, origin node,
		destination node, number of occurrences, and label
		(normal=1, anomalous=-1).
	neg_sample_ratio : float, default=1
		Proportion of negative samples to generate w.r.t. the number
		of positive edges.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	G_train : set
		Set of edges (represented as tuples of length two) occurring
		in the training set.
	G_test : dict
		Dictionary mapping the timestamps from the test set to
		adjacency lists representing the corresponding graphs.
	inductive_edges : set
		Set of edges occurring in the test set and not in the training
		set.
	nodes : list
		List of nodes observed in the training and test sets.
	rnd : object
		Random number generator.

	References
	----------
	[1] F. Poursafaei et al., Towards Better Evaluation for Dynamic
	    Link Prediction, NeurIPS, 2022.
	'''

	def __init__(
			self,
			X_train,
			X_test,
			neg_sample_ratio=1,
			random_state=None
		):

		self.G_train = set([
			(u, v)
			for u, v in zip(X_train[:, 1], X_train[:, 2])
		])
		timestamps = sorted(list(set(X_test[:, 0])))
		df_test = pd.DataFrame(
			X_test,
			columns=('timestamp', 'src', 'dst', 'cnt', 'lab')
		)
		idx = df_test.groupby(['timestamp']).groups
		self.G_test = {
			t: {
				(u, v): w
				for _, _, u, v, w, _ in df_test.loc[idx[t], :].itertuples()
			}
			for t in timestamps
		}
		self.inductive_edges = set().union(*[
				set(g.keys())
				for g in self.G_test.values()
			]) - self.G_train
		self.nodes = sorted(list(
			set(
				X_train[:, 1:3].flatten().tolist()
			).union(
				X_test[:, 1:3].flatten().tolist()
			)
		))
		self.rnd = check_random_state(random_state)
		self.neg_sample_ratio = neg_sample_ratio


	def generate_dataset(self, neg_type='easy'):
		'''
		Returns a generator object whose elements are the graphs
		from the test set with injected negative edges.


		Arguments
		---------
		neg_type : str, default='easy'
			Type of negative edges to generate.
			Possible values:
			- 'easy': random rewiring of positive edges
			- 'historical': edges observed in the training set, but not
			  at the current time step
			- 'inductive': edges observed in the test set, but not in
			  the training set and not at the current time step

		Returns
		-------
		dataset : generator
			Generator whose elements are arrays of shape (n_edges, 5).
			Each edge is described by the following elements:
			timestamp, origin, destination, occurrence count, label
			(positive=1, negative=0).

		'''

		for t, G in self.G_test.items():
			g = dict(G)
			num_true_edges = len(g)
			num_neg_edges = int(
				self.neg_sample_ratio * num_true_edges
			)
			if neg_type == 'easy':
				pass
			elif neg_type in ('historical', 'inductive'):
				edges = (
					self.generate_historical_edges(g, num_neg_edges)
					if neg_type == 'historical'
					else self.generate_inductive_edges(g, num_neg_edges)
				)
				if edges is not None:
					g.update({e: 0 for e in edges})
			else:
				raise ValueError(
					f'Unknown negative edge type: {neg_type}'
				)
			if len(g) < num_true_edges + num_neg_edges:
				edges = self.generate_negative_edges(
					self.G_test[t],
					num_true_edges + num_neg_edges - len(g)
				)
				g.update({e: 0 for e in edges})
			yield np.array([
					(
						t,
						u,
						v,
						g[(u, v)],
						int(g[(u, v)] > 0)
					)
					for u, v in g
				])


	def generate_negative_edges(self, graph, num_edges, max_tries=10):
		'''
		Generates random negative edges by rewiring positive edges.


		Arguments
		---------
		graph : dict
			Positive edges.
			Dictionary whose keys are edges (represented as tuples of
			length 2) and values are occurrence counts.
		num_edges : int
			Number of edges to generate.
		max_tries : int, default=10
			When some of the generated edges collide with positive
			edges or other negative edges, a new batch of edges is
			generated.
			This procedure stops when enough distinct edges have been
			generated or the number of iterations is greater than
			max_tries.

		Returns
		-------
		edges : list
			List of tuples representing negative edges.

		'''

		edges = self.get_random_edges(graph, num_edges)
		nodes = self.get_random_nodes(num_edges)
		neg = set([
				(u, v) for (u, _), v in zip(edges, nodes)
			]) - set(graph.keys())
		tries = 1
		while len(neg) < num_edges and tries < max_tries:
			edges = self.get_random_edges(
				graph,
				num_edges - len(neg)
			)
			nodes = self.get_random_nodes(
				num_edges - len(neg)
			)
			neg = neg.union(
				set([
					(u, v) for (u, _), v in zip(edges, nodes)
				]) - set(graph.keys())
			)
			tries += 1
		return list(neg)


	def generate_historical_edges(self, graph, num_edges):
		'''
		Generates historical negative edges.


		Arguments
		---------
		graph : dict
			Positive edges.
			Dictionary whose keys are edges (represented as tuples of
			length 2) and values are occurrence counts.
		num_edges : int
			Number of edges to generate.

		Returns
		-------
		edges : list
			List of tuples representing negative edges.

		'''

		edges = sorted(list(
			self.G_train - set(graph.keys())
		))
		return self.get_samples(edges, num_edges, False)


	def generate_inductive_edges(self, graph, num_edges):
		'''
		Generates inductive negative edges.


		Arguments
		---------
		graph : dict
			Positive edges.
			Dictionary whose keys are edges (represented as tuples of
			length 2) and values are occurrence counts.
		num_edges : int
			Number of edges to generate.

		Returns
		-------
		edges : list
			List of tuples representing negative edges.

		'''

		edges = sorted(list(
			self.inductive_edges - set(graph.keys())
		))
		return self.get_samples(edges, num_edges, False)


	def get_random_edges(self, graph, num_edges):
		'''
		Randomly samples edges from a graph.


		Arguments
		---------
		graph : dict
			Positive edges.
			Dictionary whose keys are edges (represented as tuples of
			length 2) and values are occurrence counts.
		num_edges : int
			Number of edges to sample.

		Returns
		-------
		edges : list
			List of edges represented as tuples of length 2.

		'''

		edges = list(graph.keys())
		return self.get_samples(edges, num_edges)


	def get_random_nodes(self, num_nodes):
		'''
		Randomly samples nodes from the set of nodes observed in the
		training and test sets.


		Arguments
		---------
		num_nodes : int
			Number of nodes to sample.

		Returns
		-------
		nodes : list
			List of nodes.

		'''

		return self.get_samples(self.nodes, num_nodes)


	def get_samples(self, samples, num_samples, allow_duplicates=True):
		'''
		Randomly samples from the given list.


		Arguments
		---------
		samples : list
			List of values to sample from.
		num_samples : int
			Number of elements to sample.
		allow_duplicates : bool, default=True
			If False, the returned list contains no duplicate elements
			but may be shorter than num_samples.

		Returns
		-------
		samples : list
			Sampled elements.

		'''

		if len(samples) == 0:
			return None
		elif len(samples) < num_samples and not allow_duplicates:
			idx = np.arange(len(samples))
			self.rnd.shuffle(idx)
		else:
			idx = self.rnd.randint(0, len(samples), num_samples)
		return [samples[i] for i in idx]


	def make_dataset(self, neg_type='easy'):
		'''
		Returns a list whose elements are the graphs
		from the test set with injected negative edges.


		Arguments
		---------
		neg_type : str, default='easy'
			Type of negative edges to generate.
			Possible values:
			- 'easy': random rewiring of positive edges
			- 'historical': edges observed in the training set, but not
			  at the current time step
			- 'inductive': edges observed in the test set, but not in
			  the training set and not at the current time step

		Returns
		-------
		dataset : list
			List whose elements are arrays of shape (n_edges, 5).
			Each edge is described by the following elements:
			timestamp, origin, destination, occurrence count, label
			(positive=1, negative=0).

		'''

		dataset = [
			X for X in self.generate_dataset(neg_type)
		]
		return dataset