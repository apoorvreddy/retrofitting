import nltk
import spacy
import numpy as np
from nltk.corpus import wordnet as wn
from itertools import chain

from retrofit.vectors import Vectors
from retrofit.graph import Graph


nlp = spacy.load('en_core_web_md')

# take unigram and word-only tokens from wordnet
wordnetVocab = [word for word in wn.words() if '_' not in word and word.isalpha() and '-' not in word]

print("len1", len(wordnetVocab))
#
doc = ' '.join(wordnetVocab)
XV = nlp(doc)
print("lenXV", len(XV))

# take intersection of words in wordnetVocab and spacy's wordvector vocab
finalVocab = []
finalVectors = []
for i, xv in enumerate(XV):
	if (XV[i].vector == np.zeros(300)).sum() != 300:
		finalVocab.append(str(XV[i]))
		finalVectors.append(XV[i].vector)

# cast final vectors to a 2d numpy array
finalVectors = np.array(finalVectors)

print("Vectors Shape", finalVectors.shape)
print("Vocabulary Size", len(finalVocab))

word2idx = dict(zip(finalVocab, range(len(finalVocab))))
idx2word = {v:k for k, v in word2idx.items()}
V = Vectors(word2idx, finalVectors)


# construct the graph
finalVocab = set(finalVocab)

edges = []
for word in finalVocab:
	synonyms = wn.synsets(word)
	lemmas = set(chain.from_iterable([w.lemma_names() for w in synonyms]))
	for lemma in lemmas:
		if lemma in finalVocab:
			edges.append((word2idx[word], word2idx[lemma]))

graph = Graph(edges)


# ## RetroFit
# from retrofit.retrofit import RetroFit
# retro = RetroFit(V, graph)
# newVectors = retro.fit(verbose=True)