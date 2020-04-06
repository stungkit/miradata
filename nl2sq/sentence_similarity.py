import gensim
import numpy as np
import math
from scipy.spatial import distance
from random import sample
import sys
from nltk.corpus import stopwords

class PhraseVector:
    """
    <Usage>
    phraseVector1 = PhraseVector(userInput1)
    phraseVector2 = PhraseVector(userInput2)
    similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
    """
    def __init__(self, phrase, embedding):
        self.embedding = embedding
        self.vector = self.PhraseToVec(phrase)

    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore=[]):
        if len(ignore) == 0:
            return np.mean(vectorSet, axis=0)
        else:
            return np.dot(np.transpose(vectorSet), ignore) / sum(ignore)

    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [
            word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector = self.embedding[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / \
            (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity
