# -*- coding: utf-8 -*-

# From: https://github.com/zjy-ucas/ChineseNER and https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF

import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import codecs
import pickle
import itertools
from collections import OrderedDict
import argparse
#
# import numpy as np
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
import loader
# from loader import convertBIOUS2segments, convertBIOU2SegmentsBatch, createPytorchLabels
from utils import get_logger, make_path, clean, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
import PyUtils.dir_utils as dir_utils
from PtUtils import cuda_model
import pprint as pp
import torch
import torch.nn as nn
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
import NERModel
from torch.autograd import Variable
from Losses import h_assign
import numpy as np
from Losses.losses import EMD_L2, to_one_hot
from PyUtils.AverageMeter import AverageMeter
import Losses.Metrics as Metrics
import progressbar
def str2bool(v):
    return v.lower() in ('true', '1', 'y', 'yes')
import Devs_ActionProp.PropEval.Utils as PropUtils
user_root = os.path.expanduser('~')

parser = argparse.ArgumentParser(description="Pytorch implementation for Chinese Named Entity Recognition")
# Data
parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--seg_dim', type=int, default=20, help='Embedding size for segmentation positional, 0 if not used')
parser.add_argument('--char_dim', type=int, default=100, help='Embedding size for characters')

# parser.add_argument('--input_dim', type=int, default=500, help='Number of hidden units')
# parser.add_argument('--embedding_dim', type=int, default=500, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units')
parser.add_argument('--hassign_thres', default=1.0, type=float, help='hassignment_threshold')
parser.add_argument('--alpha', default=0.1, type=float, help='trade off between classification and localization')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate for training a network')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--tag_schema',   default="iobes", type=str, help="tagging schema iobes or iob")


parser.add_argument('--eval', type=str, default='/home/zwei/Dev/NetModules/sentencechunk/chinese_ner/X', help='check-point file')
parser.add_argument('--fileid', type=int, default=41, help='file id in ckpt file')



# config for the model
def config_model(char_to_id, tag_to_id, args=None):
    config = OrderedDict()
    config["model_type"] = dir_utils.get_stem(__file__)
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = args.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = args.seg_dim
    config["lstm_dim"] = args.hidden_dim
    config["batch_size"] = args.batch_size

    config["emb_file"] = 'data/vec.txt'
    # config["clip"] = FLAGS.clip
    config["dropout"] = args.dropout    # Update: 0 means not drop out at all
    config["tag_schema"] = args.tag_schema

    config['train_file'] = 'data/example.train'
    config['dev_file'] = 'data/example.dev'
    config['test_file'] = 'data/example.test'
    config['map_file'] = 'map.pkl'

    return config


def main():
    # load data sets
    global  args
    args = parser.parse_args()
    pp.pprint(vars(args))
    # running_name = 'X'
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    # use_cuda = False


    # train_file = 'data/example.train'
    # dev_file = 'data/example.dev'
    test_file = 'data/example.test'
    # embedding_file = 'data/vec.txt'
    map_file = 'map.pkl'
    # config_file = 'config_file_pytorch'
    tag_file = 'tag.pkl'
    # embedding_easy_file = 'data/easy_embedding.npy'
    # train_sentences = load_sentences(train_file)
    # dev_sentences = load_sentences(dev_file)
    test_sentences = load_sentences(test_file)
    # train_sentences = dev_sentences
    # update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)
    # update_tag_scheme(dev_sentences, args.tag_schema)

    if not os.path.isfile(tag_file):
        print("Tag file {:s} Not found".format(tag_file))
        sys.exit(-1)
    else:
        with open(tag_file, 'rb') as t:
            tag_to_id, id_to_tag = pickle.load(t)

    if not os.path.isfile(map_file):
        print("Map file {:s} Not found".format(map_file))
        # create dictionary for word
        # dico_chars_train = char_mapping(train_sentences)[0]
        # dico_chars, char_to_id, id_to_char = augment_with_pretrained(
        #     dico_chars_train.copy(),
        #     embedding_file,
        #     list(itertools.chain.from_iterable(
        #         [[w[0] for w in s] for s in test_sentences])
        #     )
        # )
        # # _, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        #
        # with open(map_file, "wb") as f:
        #     pickle.dump([char_to_id, id_to_char], f)
    else:
        with open(map_file, "rb") as f:
            char_to_id, id_to_char = pickle.load(f)


    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id)

    print("{:d} sentences in  test.".format(len(test_data)))

    test_manager = BatchManager(test_data, 1)


    save_places = dir_utils.save_places(args.eval)

    # log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(os.path.join(save_places.log_save_dir, 'evaluation-{:d}.txt'.format(args.fileid)))
    config = config_model(char_to_id, tag_to_id, args)
    print_config(config, logger)


    logger.info("start training")

    #Update: create model and embedding!
    model = NERModel.CNERPointer(char_dim=args.char_dim, seg_dim=args.seg_dim, hidden_dim=args.hidden_dim, max_length=15,
                                 output_classes=4, dropout=args.dropout, embedding_path=None, id_to_word=id_to_char, easy_load=None)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    #Update: this won't work!
    # model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    if use_cuda:
        model = model.cuda()

    model.eval()
    if args.eval is not None:
        # if os.path.isfile(args.resume):
        ckpt_filename = os.path.join(save_places.model_save_dir, 'checkpoint_{:04d}.pth.tar'.format(args.fileid))
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        train_iou = checkpoint['IoU']
        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))

    ner_results = evaluate(model, test_manager, id_to_tag, use_cuda, max_len=5)
    eval_lines = test_ner(ner_results, save_places.summary_save_dir)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])
    return f1





def evaluate(model, data_manager, id_to_tag, use_cuda=True, max_len=5):
    results = []
    model.eval()
    pbar = progressbar.ProgressBar(max_value=data_manager.len_data)

    for batch_idx, batch in enumerate(data_manager.iter_batch(shuffle=False)):
        pbar.update(batch_idx)
        strings = batch[0][0]
        word_vectors = torch.LongTensor(batch[1])
        seg_vectors = torch.LongTensor(batch[2])
        tags = batch[-1][0]

        # batch_size = word_vectors.shape[0]
        # input_length = word_vectors.shape[1]

        word_input = Variable(word_vectors)
        seg_input = Variable(seg_vectors)

        if use_cuda:
            word_input = word_input.cuda()
            seg_input = seg_input.cuda()

        # tagging_BIOUS = batch[3]
        # segments, classes, max_len = loader.convertBIOU2SegmentsMultiClsBatch(tagging_BIOUS, id_to_tag)
        # gt_positions, gt_classes, gt_valids = loader.createPytorchLabelsMultiCls(segments, classes, max_len)

        _, head_positions, _, tail_positions, cls_scores, _ = model(
            word_input, seg_input, max_len)
        cls_scores = F.softmax(cls_scores, dim=-1)
        head_positions = head_positions.data.squeeze(0).cpu().numpy() # Update: only for batch size 1
        tail_positions = tail_positions.data.squeeze(0).cpu().numpy() # Update: only for batch size 1
        cls_scores = cls_scores.data.squeeze(0).cpu().numpy()

        # most_confident_scores = np.max(cls_scores, axis=1)
        predicted_locations = np.stack((head_positions, tail_positions), axis=-1)
        predicted_locations, nms_cls_scores = PropUtils.non_maxima_supressionv2(predicted_locations, cls_scores[:, 1:])# Not counting the background!

        #Update: remove repeatitives
        # pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
        sentence_len = len(strings)
        predicted_tags = ['O']*sentence_len
        for (s_location, s_score) in zip(predicted_locations, nms_cls_scores):
            max_cate = np.argmax(s_score)
            if s_score[max_cate]>0.5:
                cate_dict = {0:'LOC', 1:'ORG', 2:'PER'}

                if s_location[0] == s_location[1]:
                    predicted_tags[int(s_location[0])] = 'S-{:s}'.format(cate_dict[max_cate])
                else:
                    predicted_tags[int(s_location[0])]='B-{:s}'.format(cate_dict[max_cate])
                    predicted_tags[int(s_location[1])]='E-{:s}'.format(cate_dict[max_cate])
                    if int(s_location[0]+1) < int(s_location[1]-1):
                        for s_idx in range(int(s_location[0]+1),int(s_location[1]-1)):
                            predicted_tags[s_idx]='I-{:s}'.format(cate_dict[max_cate])
                    if int(s_location[0]+1) == int(s_location[1]-1):
                        predicted_tags[int(s_location[0]+1)]='I-{:s}'.format(cate_dict[max_cate])

            else:
                continue



        # batch_paths = decode(scores, sentence_len)
        # for i in range(len(strings)):
        result = []
        # string = strings[i][:sentence_len]
        gold = iobes_iob([id_to_tag[int(x)] for x in tags[:sentence_len]])
        pred = iobes_iob([unicode(x, 'utf-8') for x in predicted_tags[:sentence_len]])
        for char, s_gold, s_pred in zip(strings, gold, pred):
            result.append(" ".join([char, s_gold, s_pred]))
        results.append(result)
    return results




# def decode(logits, lengths, matrix=None, num_tags=13):
#     """
#     :param logits: [batch_size, num_steps, num_tags]float32, logits
#     :param lengths: [batch_size]int32, real length of each sequence
#     :param matrix: transaction matrix for inference
#     :return:
#     """
#     # inference final labels usa viterbi Algorithm
#     if matrix is None:
#         matrix = np.eye(num_tags+1)
#     paths = []
#     small = -1000.0
#     start = np.asarray([[small]*num_tags +[0]])
#     for score, length in zip(logits, lengths):
#         score = score[:length]
#         pad = small * np.ones([length, 1])
#         logits = np.concatenate([score, pad], axis=1)
#         logits = np.concatenate([start, logits], axis=0)
#         path, _ = viterbi_decode(logits, matrix)
#
#         paths.append(path[1:])
#     return paths


# def viterbi_decode(score, transition_params):
#   """Decode the highest scoring sequence of tags outside of TensorFlow.
#
#   This should only be used at test time.
#
#   Args:
#     score: A [seq_len, num_tags] matrix of unary potentials.
#     transition_params: A [num_tags, num_tags] matrix of binary potentials.
#
#   Returns:
#     viterbi: A [seq_len] list of integers containing the highest scoring tag
#         indicies.
#     viterbi_score: A float containing the score for the Viterbi sequence.
#   """
#   trellis = np.zeros_like(score)
#   backpointers = np.zeros_like(score, dtype=np.int32)
#   trellis[0] = score[0]
#
#   for t in range(1, score.shape[0]):
#     v = np.expand_dims(trellis[t - 1], 1) + transition_params
#     trellis[t] = score[t] + np.max(v, 0)
#     backpointers[t] = np.argmax(v, 0)
#
#   viterbi = [np.argmax(trellis[-1])]
#   for bp in reversed(backpointers[1:]):
#     viterbi.append(bp[viterbi[-1]])
#   viterbi.reverse()
#
#   viterbi_score = np.max(trellis[-1])
#   return viterbi, viterbi_score


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags



if __name__ == "__main__":
    main()


