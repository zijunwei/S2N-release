import torch
import torch.nn as nn
from torch.autograd import Variable
import re
import codecs
import numpy as np
#TODO: bi-directional, multi-layer
import progressbar
import os
def load_word2vec(emb_path, id_to_char, word_dim, save_file=None):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    n_words = len(id_to_char)
    new_weights = np.zeros([n_words, word_dim])
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else: #TODO: clearly the first line is counted here as error, remvoe it!
            if i!=0:
                emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    # Lookup table initialization
    pbar = progressbar.ProgressBar(max_value=n_words)
    for i in range(n_words):
        pbar.update(i)
        word = id_to_char[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
        else:
            print("{:s} Not Found in Dict".format(word.encode('utf-8')))

    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    if save_file is not None:
        np.save(save_file, new_weights)
    return new_weights



class WEmbed(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, w_dim=100, w_size=10, p_dim=20, p_size=4, embedding_path=None, id_to_char=None, easy_load=None):
        super(WEmbed, self).__init__()

        self.pose_embed = nn.Embedding(p_size, p_dim)

        if easy_load is not None and os.path.isfile(easy_load):
            numpy_weights = np.load(easy_load)
            w_size = numpy_weights.shape[0]
            w_dim = numpy_weights.shape[1]
            self.word_embed = nn.Embedding(w_size, w_dim)
            self.word_embed.weight = nn.Parameter(torch.from_numpy(numpy_weights).float())
            return

        if embedding_path is not None and id_to_char is not None:
            w_size = len(id_to_char)
            self.word_embed = nn.Embedding(w_size, w_dim)
            self.initialize_w_with_pdefined(embedding_path, id_to_char, w_dim, easy_load)
            return

        if id_to_char is not None:
            w_size = len(id_to_char)

        self.word_embed = nn.Embedding(w_size, w_dim)

    def initialize_w_with_pdefined(self, embedding_path, id_to_word, w_dim=100, save_file=None):
        numpy_weights = load_word2vec(embedding_path, id_to_word, w_dim, save_file)
        self.word_embed.weight = nn.Parameter(torch.from_numpy(numpy_weights).float())



        # print("Encoder Created")
    def forward(self, word_input, pose_input):
        # x [sourceL, batch_size, feature_dim]
        # output, hidden = self.rnn(x, hidden)
        word_encode = self.word_embed(word_input)
        pose_encode = self.pose_embed(pose_input)
        encode = torch.cat([word_encode, pose_encode], dim=-1)
        return encode


if __name__ == '__main__':
    import pickle as pkl
    map_file = '/home/zwei/Dev/NetModules/sentencechunk/chinese_ner/maps_py2.pkl'
    pdefined_embedding = '/home/zwei/Dev/NetModules/sentencechunk/chinese_ner/data/vec.txt'
    with open(map_file, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pkl.load(f)
    word_input = torch.LongTensor([3, 1])
    pose_input = torch.LongTensor([2, 0])
    eb = WEmbed(w_size=len(id_to_char))

    eb.initialize_w_with_pdefined(pdefined_embedding, id_to_char, w_dim=100)
    output = eb(word_input, pose_input)
    print("DEV")


