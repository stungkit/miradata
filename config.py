#/usr/bin/env python3
import os
import argparse
import multiprocessing


# Configuration

def model_config(parser):
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--reduction_factor', type=int, default=0.5)
    parser.add_argument('--max_num_cluster', type=int, default=20)
    parser.add_argument('--min_num_aliases', type=int, default=5)
    parser.add_argument('--philly_on', action='store_true')
    parser.add_argument('--get_rep_aliases', action='store_true')
    parser.add_argument('--analyze_confusion', action='store_true')
    parser.add_argument('--analyze_sim_cols', action='store_true')
    parser.add_argument('--model_dir', default='checkpoint')
    return parser

def data_config(parser):
    parser.add_argument('--embedding_path', default='~/Desktop/squad_vteam/data_v2/glove.840B.300d.w2vformat.txt')
    parser.add_argument('--schema_meta_path', default="data/raw/offlineSchemaMetadata.json")
    parser.add_argument('--log_file', default='dataFactory.log', help='path for log file.')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--log_dir',default=None, type=str, help='will set by philly.')
    parser.add_argument('--train_data', help='path to preprocessed training data file.')
    parser.add_argument('--dev_data', help='path to preprocessed validation data file.')
    parser.add_argument('--dev_gold', help='path to preprocessed validation data file.')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(), help='number of threads for preprocessing.')
    return parser

def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    args = parser.parse_args()
    return args
