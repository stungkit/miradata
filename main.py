import sys
import re
import os
import string
import logging
import argparse
import numpy as np
from collections import defaultdict
import pandas as pd
import io
import json
import random
import math
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from config import set_args
from src.data_science.key_alias import *
from src.data_science.paraphrase import *
from src.data_science.utils import *
from src.data_science.sentence_similarity import *


def main():
    args = set_args()

    # set data dir
    data_dir = args.data_dir
    model_dir = args.model_dir

    if args.philly_on:
        os.makedirs(model_dir, exist_ok=True)
        model_dir = os.path.abspath(model_dir)
        data_dir = data_dir
    else:
        os.makedirs(model_dir, exist_ok=True)
        model_dir = os.path.abspath(model_dir)

    opt = vars(args)
    embedding = load_meta(args.embedding_path)
    schema_meta = read_json(args.schema_meta_path)

    # Get representative columns (k-medoid methods)
    if args.get_rep_aliases:
        print("Get representative aliases (k-medoid methods)")
        get_rep_aliases(schema_meta, args.reduction_factor, embedding, args.embedding_dim, args.min_num_aliases, args.max_num_cluster)
    
    # Get paraphrases (back translation)
    if args.get_paraphrase:
        print("Get paraphrases (back-translation method)")
        get_paraphrase(args.para_input)
    
    # Analyze confusing aliases (k-means)
    if args.analyze_confusion:
        print("Analyze confusing aliases (k-means)")
        confusion_analysis(schema_meta, args.reduction_factor, embedding, args.embedding_dim)

    # Analyze Similar Columns
    if args.analyze_sim_cols:
        print("Analyze Similar Columns")
        similar_column_analysis(schema_meta, embedding)


if __name__ == '__main__':
    main()