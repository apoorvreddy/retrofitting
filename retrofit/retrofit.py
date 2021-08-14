import numpy as np
from retrofit.graph import Graph
from retrofit.vectors import Vectors
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set
from scipy import sparse
from tqdm import tqdm
	
class RetroFit(object):
	def __init__(self, iterations=10, alpha=1.0, beta=None):
		self.iterations = iterations
		self.alpha = alpha
		self.beta = beta
	def fit(self, vectors, graph):
		# initialize new vectors with a copy of the given vectors
		newVectors = Vectors(vectors)
		# number of vertices in the graph
		numV = max(graph.nodes) + 1
		# initialize Beta matrix
		if self.beta is None:
			self.beta = np.zeros((numV, numV))
			for u in graph.vertices():
				indegree = graph.getInDegree(u)
				nbrs = graph.neighbors(u)
				self.beta[u][nbrs] = 1.0/indegree
		# cache the denominator - 1 time computation
		denominatorCache = {}
		for u in range(numV):
			denominatorCache[u] = self.beta[u].sum() + self.alpha*graph.getInDegree(u)
		# sparsify beta
		betasp = sparse.csr_matrix(self.beta)
		for it in tqdm(range(self.iterations)):
			for u in tqdm(range(numV), leave=False):
				if graph.getInDegree(u) > 0:
					numerator = sparse.csr_matrix.dot(betasp.getrow(u), newVectors.vectors) + self.alpha * vectors.vectors[u]
					denominator = denominatorCache[u]
					newVectors.vectors[u] = numerator/denominator
		return newVectors
