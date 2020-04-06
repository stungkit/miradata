from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import re
import json
import random
import time
import math
from collections import defaultdict
import itertools
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import gensim
plt.switch_backend('agg')

from . import encoder_rnn
from . import decoder_rnn_attn

SOS_token = 0
EOS_token = 1


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def readLangs(lang1, lang2, datafile, reverse=False):
    print("Reading language pairs...")

    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(lang1, lang2, args, reverse=False):
    """
    Read text file and split into lines, split lines into pairs
    Normalize text, filter by length and content
    Make word lists from sentences in pairs
    """
    input_lang, output_lang, pairs = readLangs(
        lang1, lang2, args.datafile, reverse)
    print("Read %s sentence pairs" % len(pairs))
    max_length = args.max_length
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs by filtering the max_length" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def keyValuePairToTuple(list_):
    """
    Read the key-value pairs from the dictionary
    and generate the every possible combination pairs (2)

    Input : [A, B, C]
    Output: [(A,B), (A,C), (B,C)]
    """
    list_of_combinations = itertools.combinations(list_, 2)
    return [_ for _ in list_of_combinations]


def preprocessParaphrased(paraphrase_file):

    with open(paraphrase_file) as f:
        raw_json = json.load(f)

    with open('original-paraphrase.txt', 'w') as f:
        for question in raw_json:
            original_n_paraphrases = []
            original_n_paraphrases.append(question)
            original_n_paraphrases.extend(raw_json[question])

            list_of_combinations = keyValuePairToTuple(original_n_paraphrases)
            for tuple in list_of_combinations:
                f.write('\t'.join([str(abc) for abc in tuple]) + '\n')
    f.close()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device="cuda").view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def load_jsonl(jsonl_path):
    sql_data = []
    print("Loading data from %s" % jsonl_path)
    with open(jsonl_path, encoding='utf-8-sig') as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            sql_data.append(sql)
    return sql_data


def read_json(filename):
    with open(filename, encoding='utf-8-sig') as f:
        raw_json = json.loads(f.read())
    return raw_json


def preprocess(filename):
    list_of_sentences = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            list_of_sentences.append(line)
    return list_of_sentences

def load_meta(glovePath):
    print("Loading the glove...")
    model1 = gensim.models.KeyedVectors.load_word2vec_format(
        glovePath, binary=False, unicode_errors='ignore')
    print("Successfully loaded the file.")
    return model1