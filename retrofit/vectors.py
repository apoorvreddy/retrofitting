import numpy as np
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, Any

class Vectors(object):
	def __init__(self, *args):
		if len(args) == 2:
			self.object2idx = args[0]
			self.vectors = args[1]
		elif len(args) == 1:
			otherVector = args[0]
			self.object2idx = otherVector.object2idx.copy()
			self.vectors = np.copy(otherVector.vectors, order='K')

	def getVector(obj):
		return self.vectors[self.object2idx[obj]]

	def getIdx(obj):
		return self.object2idx[obj]