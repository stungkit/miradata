import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import pandas as pd
import io
import json
import random
import math
from nl2sq.key_alias import *
from nl2sq.utils import *
from nl2sq.sentence_similarity import *
from nl2sq.kmedoid import kMedoids

cluster_data_file = "data/raw/offlineSchemaMetadata.json"
data = read_json(cluster_data_file)
# this is a list of all attributes.
account = data['Entities'][0]['Attributes']
opportunity = data['Entities'][1]['Attributes']


# For alias clustering analysis
with io.open("cluster_output_hierarchical.txt", mode='w+', encoding="UTF-8") as file:
    for idx1 in range(0, len(account)):
        # TODO: make reduction factor as hyper parameter
        reduction_factor = .3333  # Amount of dimension reduction {0,1}

        aliases1 = account[idx1]['Aliases']
        col_name1 = account[idx1]['Column']
        natural_words1 = account[idx1]['NaturalWords']
        # Number of words to analyse according to memory availability
        n_words = len(aliases1)
        n_clusters = max(int(n_words * reduction_factor), 1)
        cluster_data, wordlist = array_gen(aliases1, nwords=n_words)
        cluster_data = np.nan_to_num(cluster_data)

        # K means
        model = KMeans(init='k-means++', n_clusters=n_clusters,
                       n_init=15, random_state=1, max_iter=500, verbose=1)
        model.fit(cluster_data)

        """
        #Hierarchical clustering
        model = AgglomerativeClustering()
        model.fit(cluster_data)
        """
        cluster_labels = model.labels_  # returns all cluster number assigned to each word respectively
        cluster_to_words = assign_word2cluster(wordlist, cluster_labels)

        # saving output in outut.text file
        print("\n########" + col_name1 + "########", file=file)

        for key in sorted(cluster_to_words.keys()):
            #print("Cluster "+str(key) +" :: "+ "|".join( k for k in cluster_to_words[key])+"\n")
            #file.writelines("Cluster "+str(key) +" :: "+ "|".join( k for k in cluster_to_words[key])+"\n")
            print("Cluster " + str(key), " :: ",
                  "|".join(k for k in cluster_to_words[key]), file=file)
    file.close()


# Create a file for representative columns (k-medoid methods)
col_alias = defaultdict()
for idx1 in range(0, len(account)):
    reduction_factor = .3333
    aliases1 = account[idx1]['Aliases']
    col_name1 = account[idx1]['Column']
    natural_words1 = account[idx1]['NaturalWords']
    # Number of words to analyse according to memory availability
    n_words = len(aliases1)
    n_clusters = max(int(n_words * reduction_factor), 1)
    cluster_data, wordlist = array_gen(aliases1, nwords=n_words)
    cluster_data = np.nan_to_num(cluster_data)

    D = pairwise_distances(cluster_data)
    reps, _ = kMedoids(D, n_clusters)
    col_alias[col_name1] = [aliases1[i] for i in reps]

with open('representative_aliases.txt', 'w') as outfile:
    json.dump(col_alias, outfile)


# Similar Columns Analysis
df = pd.DataFrame(columns=['alias1', 'alias2',
                           'SimilarityScore', 'key1', 'key2'])
max_score = 0.85

with open('out_alias.txt', 'w') as f:
    for idx1 in range(0, len(account)):
        for idx2 in range(idx1 + 1, len(account)):
            q1_list = account[idx1]['Aliases']
            col_name1 = account[idx1]['Column']
            q1_list.append(col_name1)
            q2_list = account[idx2]['Aliases']
            col_name2 = account[idx2]['Column']
            q2_list.append(col_name2)
            print(idx1, idx2, file=f)
            random.shuffle(q1_list)
            random.shuffle(q2_list)
            for q1 in q1_list:
                for q2 in q2_list:
                    phraseVector1 = PhraseVector(q1)
                    phraseVector2 = PhraseVector(q2)
                    similarityScore = phraseVector1.CosineSimilarity(
                        phraseVector2.vector)
                    # print(similarityScore)
                    if similarityScore > max_score:
                        #bestMatch = pbix_query
                        print(similarityScore, file=f)
                        print(q1, "++++++++++", q2,
                              col_name1, col_name2, file=f)
                        print("##########################################", file=f)
                        df = df.append({"alias1": q1, "alias2": q2, 'SimilarityScore': similarityScore,
                                        'key1': col_name1, 'key2': col_name2}, ignore_index=True)
df.to_csv('final_alias.csv')
