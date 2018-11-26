#!/usr/bin/env python
#Update: A cleaned version of gru2heads_inception_s4_3_EMD.py

import argparse
import numpy as np
import os, sys
# project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(project_root)
sys.path.append(project_root)
import torch.nn.utils.clip_grad
import torch
print(torch.__version__)
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SDN.PointerGRU2Heads_v7_Regression import PointerNetwork
from Devs_vsSum.datasets import vsSumLoader3_c3dd
from Devs_vsSum.datasets import vsSum_Evaluator_MultiScale as Evaluator
from PtUtils import cuda_model
from Losses import h_assign as h_match
from Losses import f_assign as f_match

from Losses.losses import EMD_L2, to_one_hot
import progressbar
from PyUtils.AverageMeter import AverageMeter
from PyUtils import dir_utils
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')
import PyUtils.log_utils as log_utils 
import random

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--seq_len', default=100, type=int, help='clip size')
parser.add_argument('--net_outputs', default=6, type=int, help='number of intervals for lstm outputs')
parser.add_argument('--split', default=0, type=int, help='cross validation split: 0-4')
parser.add_argument('--eval_metrics', default='max', type=str, help='evaluation matrix: max, avg')
parser.add_argument('--location', default='home', type=str, help='dataset/code locations')
parser.add_argument('--sample_rate', default=8, type=int, help='sample rate')
parser.add_argument('--set_cls_weight', default='False', type=str2bool, help='whether to set classification loss')
parser.add_argument('--cls_pos_weight', default=1.0, type=float, help='weigth on positive')
# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--eval', '-e', default='y', type=str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=500, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=500, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
parser.add_argument('--hassign_thres', default=0.75, type=float, help='hassignment_threshold')
parser.add_argument('--alpha', default=0.1, type=float, help='trade off between classification and localization')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate for training a network')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--hmatch', default=1, type=int, help='hungarian matching or fix matching')
parser.add_argument('--EMD', default=1, type=int, help='Using EMD loss or cls loss ')
parser.add_argument('--dataset', default='TVSum', type=str, help='dataset name')

# parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/ckpts/SDN_mnist_EMD_hmatch-assgin0.75-alpha0.1000-dim512-dropout0.5000-seqlen100-ckpt/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')

loss_type={0: 'CLS', 1: 'EMD'}
match_type = {0: 'fix', 1: 'hug'}

def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    script_name_stem = dir_utils.get_stem(__file__)
    save_directory = dir_utils.get_dir(os.path.join(project_root, 'ckpts', '{:s}-{:s}-{:s}-split-{:d}-claweight-{:s}-{:.1f}-assgin{:.2f}-alpha{:.4f}-dim{:d}-dropout{:.4f}-seqlen{:d}-samplerate-{:d}-{:s}-{:s}'.
                                  format(script_name_stem, args.dataset, args.eval_metrics, args.split, str(args.set_cls_weight), args.cls_pos_weight, args.hassign_thres, args.alpha, args.hidden_dim, args.dropout, args.seq_len, args.sample_rate, loss_type[args.EMD], match_type[args.hmatch])))
    log_file = os.path.join(save_directory, 'log-{:s}.txt'.format(dir_utils.get_date_str()))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)


    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, dropout=args.dropout, n_enc_layers=2, output_classes=2)
    hassign_thres = args.hassign_thres
    logger.info("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    logger.info('Saving logs to {:s}'.format(log_file))

    if args.resume is not None:

        ckpt_idx = 48

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        train_iou = checkpoint['IoU']
        args.start_epoch = checkpoint['epoch']

        logger.info("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))


    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    # get train/val split
    if args.dataset == 'SumMe':
        train_val_perms = np.arange(25)
    elif args.dataset == 'TVSum':
        train_val_perms = np.arange(50)
    # fixed permutation
    random.Random(0).shuffle(train_val_perms)
    train_val_perms = train_val_perms.reshape([5, -1])
    train_perms = np.delete(train_val_perms, args.split, 0).reshape([-1])
    val_perms = train_val_perms[args.split]
    logger.info(" training split: " + str(train_perms))
    logger.info(" val split: " + str(val_perms))

    if args.location == 'home':
        data_path = os.path.join(os.path.expanduser('~'), 'datasets')
    else:
        data_path = os.path.join('/nfs/%s/boyu/SDN'%(args.location), 'datasets')
    train_dataset = vsSumLoader3_c3dd.cDataset(dataset_name=args.dataset, split='train', seq_length=args.seq_len, overlap=0.9, sample_rate=[args.sample_rate],
                train_val_perms=train_perms, data_path=data_path)
    # val_dataset = vsSumLoader3_c3dd.cDataset(dataset_name=args.dataset, split='val', seq_length=args.seq_len, overlap=0.9, sample_rate=[8])
    val_evaluator = Evaluator.Evaluator(dataset_name=args.dataset, split='val', seq_length=args.seq_len, overlap=0.9, sample_rate=[args.sample_rate],
                sum_budget=0.15, train_val_perms=val_perms, eval_metrics=args.eval_metrics, data_path=data_path)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    # val_dataloader = DataLoader(val_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=False,
    #                               num_workers=4)

    model_optim = optim.Adam(filter(lambda p:p.requires_grad,  model.parameters()), lr=float(args.lr))
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=10)

    alpha=args.alpha
    # cls_weights = torch.FloatTensor([0.2, 1.0]).cuda()
    if args.set_cls_weight:
        cls_weights = torch.FloatTensor([1.*train_dataset.n_positive_train_samples/train_dataset.n_total_train_samples, args.cls_pos_weight]).cuda()
    else:
        cls_weights = torch.FloatTensor([0.5, 0.5]).cuda()
    logger.info(" total: {:d}, total pos: {:d}".format(train_dataset.n_total_train_samples, train_dataset.n_positive_train_samples))
    logger.info(" classify weight: " + str(cls_weights[0]) + str(cls_weights[1]))
    for epoch in range(args.start_epoch, args.nof_epoch+args.start_epoch):
            total_losses = AverageMeter()
            loc_losses = AverageMeter()
            cls_losses = AverageMeter()
            Accuracy = AverageMeter()
            IOU = AverageMeter()
            ordered_IOU = AverageMeter()
            model.train()
            pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
            for i_batch, sample_batch in enumerate(train_dataloader):
                pbar.update(i_batch)

                feature_batch = Variable(sample_batch[0])
                start_indices = Variable(sample_batch[1])
                end_indices = Variable(sample_batch[2])
                gt_valids = Variable(sample_batch[3])
                # seq_labels = Variable(sample_batch[3])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()

                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
                if args.hmatch:
                    assigned_scores, assigned_locations, total_valid, total_iou = h_match.Assign_Batch_v2(gt_positions, pred_positions, gt_valids, thres=hassign_thres)

                else:
                    assigned_scores, assigned_locations = f_match.Assign_Batch(gt_positions, pred_positions, gt_valids, thres=hassign_thres)
                    _, _, total_valid, total_iou = h_match.Assign_Batch_v2(gt_positions, pred_positions, gt_valids, thres=hassign_thres)

                if total_valid>0:
                    IOU.update(total_iou / total_valid, total_valid)

                assigned_scores = Variable(torch.LongTensor(assigned_scores),requires_grad=False)
                assigned_locations = Variable(torch.LongTensor(assigned_locations), requires_grad=False)
                if use_cuda:
                    assigned_scores = assigned_scores.cuda()
                    assigned_locations = assigned_locations.cuda()

                cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])
                assigned_scores = assigned_scores.contiguous().view(-1)


                cls_loss = F.cross_entropy(cls_scores, assigned_scores, weight=cls_weights)

                if total_valid>0:
                    assigned_head_positions = assigned_locations[:,:,0]
                    assigned_head_positions = assigned_head_positions.contiguous().view(-1)
                    #
                    assigned_tail_positions = assigned_locations[:,:,1]
                    assigned_tail_positions = assigned_tail_positions.contiguous().view(-1)


                    head_pointer_probs = head_pointer_probs.contiguous().view(-1, head_pointer_probs.size()[-1])
                    tail_pointer_probs = tail_pointer_probs.contiguous().view(-1, tail_pointer_probs.size()[-1])

                    assigned_head_positions = torch.masked_select(assigned_head_positions, assigned_scores.byte())
                    assigned_tail_positions = torch.masked_select(assigned_tail_positions, assigned_scores.byte())

                    head_pointer_probs = torch.index_select(head_pointer_probs, dim=0, index=assigned_scores.nonzero().squeeze(1))
                    tail_pointer_probs = torch.index_select(tail_pointer_probs, dim=0, index=assigned_scores.nonzero().squeeze(1))

                    if args.EMD:
                        assigned_head_positions = to_one_hot(assigned_head_positions, args.seq_len)
                        assigned_tail_positions = to_one_hot(assigned_tail_positions, args.seq_len)

                        prediction_head_loss = EMD_L2(head_pointer_probs, assigned_head_positions, needSoftMax=True)
                        prediction_tail_loss = EMD_L2(tail_pointer_probs, assigned_tail_positions, needSoftMax=True)
                    else:
                        prediction_head_loss = F.cross_entropy(head_pointer_probs, assigned_head_positions)
                        prediction_tail_loss = F.cross_entropy(tail_pointer_probs, assigned_tail_positions)
                    loc_losses.update(prediction_head_loss.data.item() + prediction_tail_loss.data.item(),
                                      feature_batch.size(0))
                    total_loss = alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss
                else:
                    total_loss = cls_loss

                model_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                model_optim.step()
                cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                total_losses.update(total_loss.item(), feature_batch.size(0))


            logger.info(
                "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))

            optim_scheduler.step(total_losses.avg)

            model.eval()

            # IOU = AverageMeter()
            # pbar = progressbar.ProgressBar(max_value=len(val_evaluator))
            # for i_batch, sample_batch in enumerate(val_dataloader):
            #     pbar.update(i_batch)

            #     feature_batch = Variable(sample_batch[0])
            #     start_indices = Variable(sample_batch[1])
            #     end_indices = Variable(sample_batch[2])
            #     gt_valids = Variable(sample_batch[3])
            #     # valid_indices = Variable(sample_batch[3])

            #     if use_cuda:
            #         feature_batch = feature_batch.cuda()
            #         start_indices = start_indices.cuda()
            #         end_indices = end_indices.cuda()

            #     gt_positions = torch.stack([start_indices, end_indices], dim=-1)

            #     head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(
            #         feature_batch)#Update: compared to the previous version, we now update the matching rules

            #     pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
            #     pred_scores = cls_scores[:, :, -1]
            #     #TODO: should NOT change here for evaluation!
            #     assigned_scores, assigned_locations, total_valid, total_iou = h_match.Assign_Batch_v2(gt_positions, pred_positions, gt_valids, thres=hassign_thres)
            #     if total_valid>0:
            #         IOU.update(total_iou / total_valid, total_valid)

            F1s = val_evaluator.Evaluate(model)

            logger.info(
                "Val -- Epoch :{:06d}, LR: {:.6f},\tF1s:{:.4f}".format(
                    epoch, model_optim.param_groups[0]['lr'], F1s))


            if epoch % 1 == 0:
                save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss':total_losses.avg,
            'cls_loss': cls_losses.avg,
            'loc_loss': loc_losses.avg,
            'IoU': IOU.avg,
            'val_F1s': F1s}, (epoch+1), file_direcotry=save_directory)



def save_checkpoint(state, epoch, file_direcotry):
    filename = 'checkpoint_{:04d}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    file_path = os.path.join(file_direcotry, filename.format(epoch))
    torch.save(state, file_path)



if __name__ == '__main__':
    main()
