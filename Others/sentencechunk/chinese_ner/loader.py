import os
import re
import codecs
import numpy as np
from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features
import progressbar

def load_sentences(path, lower=False, zeros=False):
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

            # pbar.update(line_idx)
            line = zero_digits(line.rstrip()) if zeros else line.rstrip()
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
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower=False):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char





def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def convertBIOU2SegmentsBatch(input_batch, id_to_tag):
    batch_segments = []
    max_len = 0
    for s_input_seq in input_batch:
        s_segment = convertBIOUS2segments(s_input_seq, id_to_tag)
        batch_segments.append(s_segment)
        if max_len<= len(s_segment):
            max_len = len(s_segment)
    return batch_segments, max_len


def convertBIOU2SegmentsMultiClsBatch(input_batch, id_to_tag):
    batch_segments = []
    batch_classes = []
    max_len = 0
    for s_input_seq in input_batch:
        s_segment, s_class = convertBIOUS2segmentsMultiCls(s_input_seq, id_to_tag)
        batch_segments.append(s_segment)
        batch_classes.append(s_class)
        if max_len<= len(s_segment):
            max_len = len(s_segment)
    return batch_segments, batch_classes, max_len


def createPytorchLabelsMultiCls(segments, classes, max_len):
    batch_size = len(segments)
    numpy_segments = np.zeros([batch_size, max_len, 2])
    numpy_gtvalids = np.zeros([batch_size, 1])
    numpy_classes = np.zeros([batch_size, max_len])
    for batch in range(batch_size):
        s_segments = segments[batch]
        s_classes = classes[batch]
        numpy_gtvalids[batch]=len(s_segments)
        for i, (s_loc, s_class) in enumerate(zip(s_segments, s_classes)):
            numpy_segments[batch, i] = s_loc
            numpy_classes[batch, i]=s_class
    return numpy_segments, numpy_classes, numpy_gtvalids


def createPytorchLabels(segments, max_len):
    batch_size = len(segments)
    numpy_segments = np.zeros([batch_size, max_len, 2])
    numpy_gtvalids = np.zeros([batch_size, 1])

    for batch in range(batch_size):
        s_segments = segments[batch]
        numpy_gtvalids[batch]=len(s_segments)
        for i, s_loc in enumerate(s_segments):
            numpy_segments[batch, i] = s_loc
    return numpy_segments, numpy_gtvalids


# def convertPytorchBatch(input_batch, id_to_tag):
#     batch_segments = []
#     max_len = 0
#     for s_input_seq in input_batch:
#         s_segment = convertBIOUS2segments(s_input_seq, id_to_tag)
#         batch_segments.append(s_segment)
#         if max_len<= len(s_segment):
#             max_len = len(s_segment)
#
#     return batch_segments, max_len

def convertBIOUS2segmentsMultiCls(input_sequence, id_to_tag):
    # sort class by LOC, ORG, PER
    segments = []
    classes = []
    starting_idx = 0
    # ending_idx = 0
    for s_idx, s_id in enumerate(input_sequence):
        if id_to_tag[s_id][0] == 'B':
            starting_idx = s_idx
        elif id_to_tag[s_id][0] == 'E':
            ending_idx = s_idx
            assert id_to_tag[input_sequence[starting_idx]][1:]==id_to_tag[input_sequence[ending_idx]][1:], 'Wrong!'
            segments.append([starting_idx, ending_idx])
            s_class = 0
            if id_to_tag[s_id][2:] == 'LOC':
                s_class = 1
            elif id_to_tag[s_id][2:] == 'ORG':
                s_class = 2
            elif id_to_tag[s_id][2:] == 'PER':
                s_class = 3
            else:
                print("Something wrong")
            classes.append(s_class)

        elif id_to_tag[s_id][0] == 'S':
            starting_idx = s_idx
            ending_idx = s_idx
            s_class = 0
            if id_to_tag[s_id][2:] == 'LOC':
                s_class = 1
            elif id_to_tag[s_id][2:] == 'ORG':
                s_class = 2
            elif id_to_tag[s_id][2:] == 'PER':
                s_class = 3
            else:
                print("Something wrong")
            classes.append(s_class)
            segments.append([starting_idx, ending_idx])
        elif id_to_tag[s_id][0] == 'O':
            continue
        elif id_to_tag[s_id][0] == 'I':
            assert id_to_tag[input_sequence[starting_idx]][1:] == id_to_tag[s_id][1:], 'Wrong HERE!'
        else:
            print("Some thing wrong!")
    assert len(segments) == len(classes)
    return segments, classes


def convertBIOUS2segments(input_sequence, id_to_tag):
    segments = []
    starting_idx = 0
    # ending_idx = 0
    for s_idx, s_id in enumerate(input_sequence):
        if id_to_tag[s_id][0] == 'B':
            starting_idx = s_idx
        elif id_to_tag[s_id][0] == 'E':
            ending_idx = s_idx
            assert id_to_tag[input_sequence[starting_idx]][1:]==id_to_tag[input_sequence[ending_idx]][1:], 'Wrong!'
            segments.append([starting_idx, ending_idx])
        elif id_to_tag[s_id][0] == 'S':
            starting_idx = s_idx
            ending_idx = s_idx
            segments.append([starting_idx, ending_idx])
        elif id_to_tag[s_id][0] == 'O':
            continue
        elif id_to_tag[s_id][0] == 'I':
            assert id_to_tag[input_sequence[starting_idx]][1:] == id_to_tag[s_id][1:], 'Wrong HERE!'
        else:
            print("Some thing wrong!")

    return segments



def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

