from textblob import TextBlob
from collections import defaultdict
from random import randint
from time import sleep
import json
from . import utils
import re

same_cnt = 0


def paraphrase(original):
    """
    Args:
            original (str): The original sentence.
    Returns:
            set: The set of paraphrased sentences.

    For additional language codes: https://cloud.google.com/translate/docs/languages        
    """
    paraphrased_set = set()
    global same_cnt
    for language in ['de', 'fr', 'ja', 'nl', 'it']:
        try:
            intermediate_language = TextBlob(original).translate(to=language)
            sleep(randint(0, 1))
            paraphrased = intermediate_language.translate(to="en")
            if original != paraphrased:
                paraphrased_set.add(str(paraphrased))
                sleep(randint(0, 1))
            else:
                same_cnt += 1
        except Exception as e:
            #print (e)
            pass
    return paraphrased_set  # paraphrase


if __name__ == "__main__":

    # Replace the input file with your own.
    original_sentences = utils.preprocess('intent.txt')
    paraphrased_sentences = defaultdict()

    for original_sentence in original_sentences:
        paraphrased_list = list(paraphrase(original_sentence))
        paraphrased_sentences[original_sentence] = paraphrased_list

    with open('paraphrased.json', 'w') as outfile:
        json.dump(paraphrased_sentences, outfile)

    #print ('total ', same_cnt, ' sentences are the same as original.')
