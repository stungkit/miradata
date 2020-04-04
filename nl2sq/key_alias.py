import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import io
import json
import random
import math
from . import sentence_similarity


def assign_word2cluster(word_list, cluster_labels):
    '''
    RETURNS: dict {"cluster":[words  assigend to cluster]}
    '''
    cluster_to_words = defaultdict(list)
    for index, cluster in enumerate(cluster_labels):
        cluster_to_words[cluster].append(word_list[index])
    return cluster_to_words


def array_gen(aliases, nwords):
    np_arrays = np.zeros((len(aliases), 300))
    wordlist = []
    for index, alias in enumerate(aliases):
        wordlist.append(alias)
        vec = sentence_similarity.PhraseVector(alias).vector
        if np.isnan(vec).any():
            np_arrays[index] = float("nan")
        else:
            np_arrays[index] = vec
    return np_arrays, wordlist
