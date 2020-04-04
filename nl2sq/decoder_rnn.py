from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pseudo_attn = torch.zeros(1, max_length)

    # encoder_output is not used in vanila decoder.
    def forward(self, input, hidden, encoder_output):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, self.pseudo_attn

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
