import numpy as np
from retrofit.graph import Graph
from retrofit.vectors import Vectors
from scipy import sparse
from tqdm import tqdm
	
class RetroFit(object):
	def __init__(self, iterations=2, alpha=1.0, beta=None):
		self.iterations = iterations
		self.alpha = alpha
		self.beta = beta

	def fit(self, vectors, graph):
		# initialize new vectors with a copy of the given vectors
		newVectors = Vectors(vectors)
		# number of vertices in the graph
		numV = max(graph.nodes) + 1
		# denominatorCache stores the denominator of the update equation for each node in the graph
		# cache the denominator - 1 time computation
		denominatorCache = {}
		if self.beta is None:
			# if beta is not specified, set beta_ij to 1/indegree(i)
			# update equation is faster too as this removes need for matrix multiplies
			# sum of beta_ij over j is 1
			for u in range(numV):
				denominatorCache[u] = 1.0 + self.alpha*graph.getInDegree(u)
			for it in tqdm(range(self.iterations)):
				for u in tqdm(range(numV), leave=False):
					if graph.getInDegree(u) > 0:
						nbrs = graph.neighbors(u)
						# online update
						numerator = newVectors.vectors[nbrs].sum(axis=0)/(1.0 * graph.getInDegree(u)) + self.alpha * vectors.vectors[u]
						denominator = denominatorCache[u]
						newVectors.vectors[u] = numerator/denominator
		else:
			# handle the general case where beta is specified
			for u in range(numV):
				denominatorCache[u] = self.beta[u].sum() + self.alpha*graph.getInDegree(u)
			# sparsify beta
			betasp = sparse.csr_matrix(self.beta)
			for it in tqdm(range(self.iterations)):
				for u in tqdm(range(numV), leave=False):
					if graph.getInDegree(u) > 0:
						# online update
						numerator = sparse.csr_matrix.dot(betasp.getrow(u), newVectors.vectors) + self.alpha * vectors.vectors[u]
						denominator = denominatorCache[u]
						newVectors.vectors[u] = numerator/denominator
		return newVectors
