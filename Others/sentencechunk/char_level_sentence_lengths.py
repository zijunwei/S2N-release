from __future__ import print_function
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
import sys
import numpy as np
from sentencechunk.neural_sequence_labeling.utils import UNK, NUM, PAD
import unicodedata
import re

def is_digit(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(s)
    if result:
        return True
    return False


user_root = os.path.expanduser('~')
dataset_root = os.path.join(user_root, 'datasets')
data_head_dir = os.path.join(dataset_root, 'conll2003/en/raw/')

target_file = os.path.join(data_head_dir, 'train.txt')
#
# with open(target_file, 'r') as f:
#     n_sentences = 0
#     char_lengths = []
#     words, pos_tags, chunk_tags, ner_tags = [], [], [], []
#     document_line = True
#     for line in f:
#         line = line.strip()
#         if line.startswith("-DOCSTART-"):
#             document_line = True
#             continue
#         if len(line) == 0:
#             if document_line:
#                 document_line=False
#                 continue
#             else:
#                 sentence_len = 0
#                 for s_word, s_tag_type in zip(words, pos_tags):
#                     if s_tag_type in ['(', ')', '$', ':', "'", ]:
#                     word_length = len(s_word)
#                 n_sentences += 1
#                 words, pos_tags, chunk_tags, ner_tags = [], [], [], []
#
#         else:
#             word, pos, chunk, ner = line.split(' ')
#             #     word = NUM
#             words += [word]
#             pos_tags += [pos]
#             chunk_tags += [chunk]
#             ner_tags += [ner]
