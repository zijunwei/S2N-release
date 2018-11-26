import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from SDN.PointerGRU2Heads_v6_mclass import PointerNetwork
import WordEmbeddings


class CNERPointer(nn.Module):
    def __init__(self, char_dim,
                 seg_dim,
                 hidden_dim,
                 max_length, output_classes=4, dropout=0.5, embedding_path=None, id_to_word=None, easy_load=None):
        super(CNERPointer, self).__init__()
        self.char_dim = char_dim
        self.seg_dim = seg_dim
        self.embedding_dim = self.char_dim + self.seg_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.EMB = WordEmbeddings.WEmbed(w_dim=char_dim, p_dim=seg_dim, p_size=4,
                                         embedding_path=embedding_path, id_to_char=id_to_word, easy_load=easy_load)
        self.Pointer = PointerNetwork(input_dim=self.embedding_dim, embedding_dim=self.embedding_dim,
                                      hidden_dim=self.hidden_dim, max_decoding_len=max_length, dropout=dropout, output_class=output_classes)


    def forward(self, word_vectors, seg_vectors, deocde_len=None):
        assert word_vectors.shape==seg_vectors.shape, 'WORNG'
        batch_size = word_vectors.shape[0]
        sentence_length = word_vectors.shape[1]

        embeddings = self.EMB(word_vectors, seg_vectors)
        embeddings = embeddings.permute(0,2,1)
        head_pointer_probs, head_positions,\
        tail_pointer_probs,tail_positions, cls_scores, enc_action_scores=self.Pointer(embeddings, deocde_len)
        return head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, enc_action_scores