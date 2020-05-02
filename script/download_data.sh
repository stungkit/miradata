#!/bin/sh

DATA_DIR=$(pwd)/../data/external

## Download SQuAD data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $DATA_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $DATA_DIR/dev-v1.1.json

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json  -O $DATA_DIR/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json  -O $DATA_DIR/dev-v2.0.json


# Download GloVe
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $DATA_DIR/glove.840B.300d.zip
unzip $DATA_DIR/glove.840B.300d.zip -d $DATA_DIR
rm $DATA_DIR/glove.840B.300d.zip

# Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O $DATA_DIR/MT-LSTM.pt
