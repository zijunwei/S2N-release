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
from loader import convertBIOUS2segments, convertBIOU2SegmentsBatch, createPytorchLabels
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

user_root = os.path.expanduser('~')

parser = argparse.ArgumentParser(description="Pytorch implementation for Chinese Named Entity Recognition")
# Data
parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--eval', '-e', default='y', type=str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--seg_dim', type=int, default=20, help='Embedding size for segmentation positional, 0 if not used')
parser.add_argument('--char_dim', type=int, default=100, help='Embedding size for characters')

# parser.add_argument('--input_dim', type=int, default=500, help='Number of hidden units')
# parser.add_argument('--embedding_dim', type=int, default=500, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units')
parser.add_argument('--hassign_thres', default=0.5, type=float, help='hassignment_threshold')
parser.add_argument('--alpha', default=0.1, type=float, help='trade off between classification and localization')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate for training a network')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--tag_schema',   default="iobes", type=str, help="tagging schema iobes or iob")





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
    running_name = 'X'
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    # use_cuda = False


    train_file = 'data/example.train'
    dev_file = 'data/example.dev'
    test_file = 'data/example.test'
    embedding_file = 'data/vec.txt'
    map_file = 'map.pkl'
    config_file = 'config_file_pytorch'
    tag_file = 'tag.pkl'
    embedding_easy_file = 'data/easy_embedding.npy'
    train_sentences = load_sentences(train_file)
    dev_sentences = load_sentences(dev_file)
    test_sentences = load_sentences(test_file)
    # train_sentences = dev_sentences
    update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)
    update_tag_scheme(dev_sentences, args.tag_schema)

    if not os.path.isfile(tag_file):
        _, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(tag_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(tag_file, 'rb') as t:
            tag_to_id, id_to_tag = pickle.load(t)

    if not os.path.isfile(map_file):
        # create dictionary for word
        dico_chars_train = char_mapping(train_sentences)[0]
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(
            dico_chars_train.copy(),
            embedding_file,
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in test_sentences])
            )
        )
        # _, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        with open(map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char], f)
    else:
        with open(map_file, "rb") as f:
            char_to_id, id_to_char = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, args.batch_size)
    dev_manager = BatchManager(dev_data, 50)
    test_manager = BatchManager(test_data, 50)
    # make path for store log and model if not exist
    # make_path(FLAGS)
    if os.path.isfile(config_file):
        config = load_config(config_file)
    else:
        config = config_model(char_to_id, tag_to_id, args)
        save_config(config, config_file)
    # make_path(running_name)

    save_places = dir_utils.save_places(running_name)

    # log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(os.path.join(save_places.log_save_dir, '{:s}.txt'.format(dir_utils.get_date_str())))
    print_config(config, logger)


    logger.info("start training")
    # loss = []

    #Update: create model and embedding!
    model = NERModel.CNERPointer(char_dim=args.char_dim, seg_dim=args.seg_dim, hidden_dim=args.hidden_dim, max_length=15,
                                 embedding_path=embedding_file, id_to_word=id_to_char, easy_load=embedding_easy_file)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    #Update: this won't work!
    # model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    if use_cuda:
        model = model.cuda()


    model_optim = optim.Adam(filter(lambda p:p.requires_grad,  model.parameters()), lr=float(args.lr))
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=10)

    for epoch in range(args.start_epoch, args.nof_epoch+args.start_epoch):
        total_losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()
        Accuracy = AverageMeter()
        IOU = AverageMeter()
        ordered_IOU = AverageMeter()
        model.train()
        pbar = progressbar.ProgressBar(max_value=train_manager.len_data)

        for batch_idx, batch in enumerate(train_manager.iter_batch(shuffle=True)):
            pbar.update(batch_idx)
            word_vectors = torch.LongTensor(batch[1])
            seg_vectors = torch.LongTensor(batch[2])

            batch_size = word_vectors.shape[0]
            input_length = word_vectors.shape[1]

            word_input = Variable(word_vectors)
            seg_input = Variable(seg_vectors)

            if use_cuda:
                word_input = word_input.cuda()
                seg_input = seg_input.cuda()

            tagging_BIOUS = batch[3]
            segments, max_len = convertBIOU2SegmentsBatch(tagging_BIOUS, id_to_tag)
            gt_positions, gt_valids = createPytorchLabels(segments, max_len)

            head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(word_input, seg_input, max_len)

            pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

            assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions, gt_valids,
                                                                        thres=args.hassign_thres)

            if np.sum(assigned_scores)>=1:
                iou_rate, effective_positives = Metrics.get_avg_iou2(np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
                                               np.reshape(assigned_locations, (-1, 2)), np.reshape(assigned_scores,
                                                                                                   assigned_scores.shape[
                                                                                                       0] *
                                                                                                   assigned_scores.shape[
                                                                                                       1]))

                IOU.update(iou_rate/(effective_positives), effective_positives)
                # ordered_IOU.update(ordered_iou_rate/(args.batch_size*args.n_outputs),args.batch_size*args.n_outputs)

                # n_effective_batches += 1

            assigned_scores = Variable(torch.LongTensor(assigned_scores),requires_grad=False)
            assigned_locations = Variable(torch.LongTensor(assigned_locations), requires_grad=False)
            if use_cuda:
                assigned_scores = assigned_scores.cuda()
                assigned_locations = assigned_locations.cuda()

            cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])
            assigned_scores = assigned_scores.contiguous().view(-1)


            cls_loss = F.cross_entropy(cls_scores, assigned_scores)

            if torch.sum(assigned_scores)>0:
                # print("HAHA")
                assigned_head_positions = assigned_locations[:,:,0]
                assigned_head_positions = assigned_head_positions.contiguous().view(-1)
                #
                assigned_tail_positions = assigned_locations[:,:,1]
                assigned_tail_positions = assigned_tail_positions.contiguous().view(-1)


                head_pointer_probs = head_pointer_probs.contiguous().view(-1, head_pointer_probs.size()[-1])
                tail_pointer_probs = tail_pointer_probs.contiguous().view(-1, tail_pointer_probs.size()[-1])


                # mask here: if there is non in assigned scores, no need to compute ...

                assigned_head_positions = torch.masked_select(assigned_head_positions, assigned_scores.byte())
                assigned_tail_positions = torch.masked_select(assigned_tail_positions, assigned_scores.byte())

                head_pointer_probs = torch.index_select(head_pointer_probs, dim=0, index=assigned_scores.nonzero().squeeze(1))
                tail_pointer_probs = torch.index_select(tail_pointer_probs, dim=0, index=assigned_scores.nonzero().squeeze(1))

                assigned_head_positions = to_one_hot(assigned_head_positions, input_length)
                assigned_tail_positions = to_one_hot(assigned_tail_positions, input_length)

                prediction_head_loss = EMD_L2(head_pointer_probs, assigned_head_positions, needSoftMax=True)
                prediction_tail_loss = EMD_L2(tail_pointer_probs, assigned_tail_positions, needSoftMax=True)
                loc_losses.update(prediction_head_loss.data.item() + prediction_tail_loss.data.item(),
                                  batch_size)
                total_loss = args.alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss
            else:
                total_loss = cls_loss

            model_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            model_optim.step()
            cls_losses.update(cls_loss.data.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)


        logger.info(
            "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                epoch,
                model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))

        optim_scheduler.step(total_losses.avg)


        total_losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()
        Accuracy = AverageMeter()
        IOU = AverageMeter()
        ordered_IOU = AverageMeter()
        model.eval()
        pbar = progressbar.ProgressBar(max_value=dev_manager.len_data)

        for batch_idx, batch in enumerate(dev_manager.iter_batch(shuffle=True)):
            pbar.update(batch_idx)
            word_vectors = torch.LongTensor(batch[1])
            seg_vectors = torch.LongTensor(batch[2])

            batch_size = word_vectors.shape[0]
            input_length = word_vectors.shape[1]

            word_input = Variable(word_vectors)
            seg_input = Variable(seg_vectors)

            if use_cuda:
                word_input = word_input.cuda()
                seg_input = seg_input.cuda()

            tagging_BIOUS = batch[3]
            segments, max_len = convertBIOU2SegmentsBatch(tagging_BIOUS, id_to_tag)

            head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(
                word_input, seg_input, max_len)

            pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
            gt_positions, gt_valids = createPytorchLabels(segments, max_len)

            assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions,
                                                                        gt_valids,
                                                                        thres=args.hassign_thres)

            if np.sum(assigned_scores) >= 1:
                iou_rate, effective_positives = Metrics.get_avg_iou2(
                    np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
                    np.reshape(assigned_locations, (-1, 2)), np.reshape(assigned_scores,
                                                                        assigned_scores.shape[
                                                                            0] *
                                                                        assigned_scores.shape[
                                                                            1]))

                IOU.update(iou_rate / (effective_positives), effective_positives)
                # ordered_IOU.update(ordered_iou_rate/(args.batch_size*args.n_outputs),args.batch_size*args.n_outputs)

                # n_effective_batches += 1

            assigned_scores = Variable(torch.LongTensor(assigned_scores), requires_grad=False)
            assigned_locations = Variable(torch.LongTensor(assigned_locations), requires_grad=False)
            if use_cuda:
                assigned_scores = assigned_scores.cuda()
                assigned_locations = assigned_locations.cuda()

            cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])
            assigned_scores = assigned_scores.contiguous().view(-1)

            cls_loss = F.cross_entropy(cls_scores, assigned_scores)

            if torch.sum(assigned_scores) > 0:
                # print("HAHA")
                assigned_head_positions = assigned_locations[:, :, 0]
                assigned_head_positions = assigned_head_positions.contiguous().view(-1)
                #
                assigned_tail_positions = assigned_locations[:, :, 1]
                assigned_tail_positions = assigned_tail_positions.contiguous().view(-1)

                head_pointer_probs = head_pointer_probs.contiguous().view(-1, head_pointer_probs.size()[-1])
                tail_pointer_probs = tail_pointer_probs.contiguous().view(-1, tail_pointer_probs.size()[-1])

                # mask here: if there is non in assigned scores, no need to compute ...

                assigned_head_positions = torch.masked_select(assigned_head_positions, assigned_scores.byte())
                assigned_tail_positions = torch.masked_select(assigned_tail_positions, assigned_scores.byte())

                head_pointer_probs = torch.index_select(head_pointer_probs, dim=0,
                                                        index=assigned_scores.nonzero().squeeze(1))
                tail_pointer_probs = torch.index_select(tail_pointer_probs, dim=0,
                                                        index=assigned_scores.nonzero().squeeze(1))

                assigned_head_positions = to_one_hot(assigned_head_positions, input_length)
                assigned_tail_positions = to_one_hot(assigned_tail_positions, input_length)

                prediction_head_loss = EMD_L2(head_pointer_probs, assigned_head_positions, needSoftMax=True)
                prediction_tail_loss = EMD_L2(tail_pointer_probs, assigned_tail_positions, needSoftMax=True)
                loc_losses.update(prediction_head_loss.data.item() + prediction_tail_loss.data.item(),
                                  batch_size)
                total_loss = args.alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss
            else:
                total_loss = cls_loss

            # model_optim.zero_grad()
            # total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # model_optim.step()
            cls_losses.update(cls_loss.data.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)

        logger.info(
            "Val -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                epoch,
                model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg,
                IOU.avg, ordered_IOU.avg))

        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': total_losses.avg,
                'cls_loss': cls_losses.avg,
                'loc_loss': loc_losses.avg,
                'IoU': IOU.avg}, (epoch + 1), file_direcotry=save_places.model_save_dir)


def save_checkpoint(state, epoch, file_direcotry):
    filename = 'checkpoint_{:04d}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    file_path = os.path.join(file_direcotry, filename.format(epoch))
    torch.save(state, file_path)

    # optim_scheduler.step(total_losses.avg)







if __name__ == "__main__":
    main()


