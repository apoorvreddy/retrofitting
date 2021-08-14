from collections import defaultdict
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, Any


class Graph(object):
	def __init__(self, edges):
		self.adjlist = defaultdict(list)
		self.indegree = defaultdict(int)
		self.nodes = set()
		self.__make_graph(edges)

	def __make_graph(self, edges):
		for u, v in edges:
			self.adjlist[u].append(v)
			self.indegree[u] += 1
			self.nodes.add(u)
			self.nodes.add(v)

	def neighbors(self, u) -> List[int]:
		return self.adjlist[u]

	def getInDegree(self, u) -> int:
		return self.indegree[u]

	def vertices(self) -> List[int]:
		return list(self.adjlist.keys())

	def numVertices(self) -> int:
		return len(self.nodes)