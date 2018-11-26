# -*- coding: utf-8 -*-

import codecs
import torch.utils.data as data
import sentencechunk.chinese_ner.datasets.NER_utils as NER_utils

def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    with codecs.open(path, 'r', 'utf8') as fread:
        # n_lines = len(fread)
        print("Read from {:s}".format(path))
        # pbar = progressbar.ProgressBar(max_value=n_lines)
        for line_idx, line in enumerate(fread):
            assert line_idx==num,'ER'
            num += 1

            line = line.rstrip()
            # print(list(line))
            if not line: #Update: only deal with space between sentences
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:# remove the DOCstart
                        sentences.append(sentence)
                    sentence = []
            else:
                if line[0] == " ":#Update: this part is never used in Chinese ner!
                    line = "$" + line[1:]
                    word = line.split()
                    # word[0] = " "
                else:
                    word= line.split()
                assert len(word) >= 2, ([word[0]])
                sentence.append(word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)

    return sentences

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not NER_utils.iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = NER_utils.iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


class NER(data.Dataset):
    def __init__(self, dataset_path, dataset_split='train', tag_scheme='iobes'):

        if dataset_split == 'train':
            self.train = True  # training set or test set
        else:
            self.train = False

        self.dataset_path = dataset_path
        self.tag_scheme = tag_scheme
        # print("Loading data from {:s}".format(self.dataset_path))
        self.sentences = load_sentences(self.dataset_path)
        update_tag_scheme(self.sentences, self.tag_scheme)
        print("DEBUG")


if __name__ == '__main__':
    dset = NER('/home/zwei/Dev/NetModules/sentencechunk/chinese_ner/data/example.dev')
    print("DBU")
