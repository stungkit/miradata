import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import io
import json
import random
import math
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from .sentence_similarity import PhraseVector
from .kmedoid import kMedoids


def __assign_word2cluster(word_list, cluster_labels):
    '''
    RETURNS: dict {"cluster":[words  assigend to cluster]}
    '''
    cluster_to_words = defaultdict(list)
    for index, cluster in enumerate(cluster_labels):
        cluster_to_words[cluster].append(word_list[index])
    return cluster_to_words


def array_gen(aliases, embedding, nwords, embedding_dim):
    np_arrays = np.zeros((len(aliases), embedding_dim))
    wordlist = []
    for index, alias in enumerate(aliases):
        wordlist.append(alias)
        vec = PhraseVector(alias, embedding).vector
        if np.isnan(vec).any():
            np_arrays[index] = float("nan")
        else:
            np_arrays[index] = vec
    return np_arrays, wordlist


def get_rep_aliases(schema_meta, reduction_factor, embedding, embedding_dim, min_num_aliases, max_num_cluster):
    col_alias = defaultdict()
    # this is a list of all attributes.
    account = schema_meta['Entities'][0]['Attributes']
    opportunity = schema_meta['Entities'][1]['Attributes']

    for idx1 in range(0, len(account)):
        aliases1 = account[idx1]['Aliases']
        col_name1 = account[idx1]['Column']
        natural_words1 = account[idx1]['NaturalWords']
        # Number of words to analyse according to memory availability
        n_words = len(aliases1)
        # Do not cluster if the # of aliases are less than min_num_aliases. 
        if n_words <= min_num_aliases:  
            col_alias[col_name1] = aliases1
        else: 
            reduced_num = max(int(n_words * reduction_factor), 1)
            n_clusters = min(reduced_num, max_num_cluster)
            cluster_data, wordlist = array_gen(aliases1, embedding, n_words, embedding_dim)
            cluster_data = np.nan_to_num(cluster_data)
            D = pairwise_distances(cluster_data)
            reps, _ = kMedoids(D, n_clusters)
            col_alias[col_name1] = [aliases1[i] for i in reps]
    with open('data/processed/representative_aliases.txt', 'w') as outfile:
        json.dump(col_alias, outfile)


def confusion_analysis(schema_meta, reduction_factor, embedding, embedding_dim):
    account = schema_meta['Entities'][0]['Attributes']
    opportunity = schema_meta['Entities'][1]['Attributes']
    
    with io.open("data/processed/column_confusion_analysis.txt", mode='w+', encoding="UTF-8") as file:
        for idx1 in range(0, len(account)):
            aliases1 = account[idx1]['Aliases']
            col_name1 = account[idx1]['Column']
            natural_words1 = account[idx1]['NaturalWords']
            # Number of words to analyse according to memory availability
            n_words = len(aliases1)
            n_clusters = max(int(n_words * reduction_factor), 1)
            cluster_data, wordlist = array_gen(aliases1, embedding, n_words, embedding_dim)
            cluster_data = np.nan_to_num(cluster_data)

            # K means
            model = KMeans(init='k-means++', n_clusters=n_clusters,
                        n_init=15, random_state=1, max_iter=500, verbose=1)
            model.fit(cluster_data)

            cluster_labels = model.labels_  # returns all cluster number assigned to each word respectively
            cluster_to_words = __assign_word2cluster(wordlist, cluster_labels)

            # saving output in outut.text file
            print("\n########" + col_name1 + "########", file=file)
            for key in sorted(cluster_to_words.keys()):
                print("Cluster " + str(key), " :: ",
                    "|".join(k for k in cluster_to_words[key]), file=file)
        file.close()


def similar_column_analysis(schema_meta, embedding):
    account = schema_meta['Entities'][0]['Attributes']
    df = pd.DataFrame(columns=['alias1', 'alias2',
                            'SimilarityScore', 'key1', 'key2'])
    max_score = 0.85

    with open('data/processed/out_alias.txt', 'w') as f:
        for idx1 in range(0, len(account)):
            for idx2 in range(idx1 + 1, len(account)):
                q1_list = account[idx1]['Aliases']
                col_name1 = account[idx1]['Column']
                q1_list.append(col_name1)
                q2_list = account[idx2]['Aliases']
                col_name2 = account[idx2]['Column']
                q2_list.append(col_name2)
                print ('calculating ', col_name1, col_name2)
                print(idx1, idx2, file=f)
                random.shuffle(q1_list)
                random.shuffle(q2_list)
                for q1 in q1_list:
                    for q2 in q2_list:
                        phraseVector1 = PhraseVector(q1, embedding)
                        phraseVector2 = PhraseVector(q2, embedding)
                        similarityScore = phraseVector1.CosineSimilarity(
                            phraseVector2.vector)
                        #print(q1, q2, similarityScore)
                        if similarityScore > max_score:
                            print(similarityScore, file=f)
                            print(q1, "++++++++++", q2,
                                col_name1, col_name2, file=f)
                            print("##########################################", file=f)
                            df = df.append({"alias1": q1, "alias2": q2, 'SimilarityScore': similarityScore,
                                            'key1': col_name1, 'key2': col_name2}, ignore_index=True)
    
    # TODO: Cut the top n. Maybe this step to be left for users. 
    df.to_csv('similar_column_analysis.csv')
