#!/usr/bin/env python
#Update: A cleaned version of gru2heads_inception_s4_3_EMD.py

import argparse
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import pprint as pp
import numpy as np
import torch.nn.utils.clip_grad
import torch
print(torch.__version__)
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PointerGRU2Heads_v4_mlayer_3in import PointerNetwork
from ActionLocalizationDevs.datasets.THUMOS14.dataloader_c3dd_aug_fast import THUMOST14
from PtUtils import cuda_model
from Losses import h_assign
from Losses.losses import EMD_L2, to_one_hot
import progressbar
from PyUtils.AverageMeter import AverageMeter
import Metrics
from PyUtils import dir_utils
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--seq_len', default=90, type=int, help='clip size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')

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
# parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/PtrNet2/gru2heads_proposal_s4-3_EMD_ckpt/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')


def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    # Pretty print the run args
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, dropout=args.dropout, n_enc_layers=2)
    hassign_thres = args.hassign_thres
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    script_name_stem = dir_utils.get_stem(__file__)
    save_directory = '{:s}-assgin{:.2f}-alpha{:.4f}-dim{:d}-dropout{:.4f}-ckpt'.format(script_name_stem, hassign_thres, args.alpha, args.hidden_dim, args.dropout)
    print("Save ckpt to {:s}".format(save_directory))
    if args.resume is not None:

        ckpt_idx = 3

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        train_iou = checkpoint['IoU']
        args.start_epoch = checkpoint['epoch']

        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))


    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    train_dataset  = THUMOST14(seq_length=args.seq_len, overlap=0.9, sample_rate=[4], dataset_split='train',rdDrop=True,rdOffset=True)

    val_dataset = THUMOST14(seq_length=args.seq_len, overlap=0.9, sample_rate=[4], dataset_split='val',rdDrop=False, rdOffset=False)


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    model_optim = optim.Adam(filter(lambda p:p.requires_grad,  model.parameters()), lr=float(args.lr))
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=20)

    alpha=args.alpha
    # cls_weights = torch.FloatTensor([0.05, 1.0]).cuda()
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

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()

                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

                assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions, gt_valids, thres=hassign_thres)
                # if np.sum(assigned_scores) > 1:
                #     print("DEBUG")
                # correct_predictions = np.sum(assigned_scores[:,:args.n_outputs])
                # cls_rate = correct_predictions*1./np.sum(assigned_scores)
                if np.sum(assigned_scores)>=1:
                    iou_rate, effective_positives = Metrics.get_avg_iou2(np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
                                                   np.reshape(assigned_locations, (-1, 2)), np.reshape(assigned_scores,
                                                                                                       assigned_scores.shape[
                                                                                                           0] *
                                                                                                       assigned_scores.shape[
                                                                                                           1]))
                    IOU.update(iou_rate/(effective_positives), effective_positives)


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

                    assigned_head_positions = to_one_hot(assigned_head_positions, 90)
                    assigned_tail_positions = to_one_hot(assigned_tail_positions, 90)

                    prediction_head_loss = EMD_L2(head_pointer_probs, assigned_head_positions, needSoftMax=True)
                    prediction_tail_loss = EMD_L2(tail_pointer_probs, assigned_tail_positions, needSoftMax=True)
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


            print(
                "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))

            optim_scheduler.step(total_losses.avg)

            model.eval()
            total_losses = AverageMeter()
            loc_losses = AverageMeter()
            cls_losses = AverageMeter()
            Accuracy = AverageMeter()
            IOU = AverageMeter()
            ordered_IOU = AverageMeter()
            pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
            for i_batch, sample_batch in enumerate(val_dataloader):
                pbar.update(i_batch)

                feature_batch = Variable(sample_batch[0])
                start_indices = Variable(sample_batch[1])
                end_indices = Variable(sample_batch[2])
                gt_valids = Variable(sample_batch[3])
                # valid_indices = Variable(sample_batch[3])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()

                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(
                    feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

                assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions, gt_valids, thres=hassign_thres)
                # if np.sum(assigned_scores) > 1:
                #     print("DEBUG")
                # correct_predictions = np.sum(assigned_scores[:,:args.n_outputs])
                # cls_rate = correct_predictions*1./np.sum(assigned_scores)
                if np.sum(assigned_scores) >= 1:
                    iou_rate, effective_positives = Metrics.get_avg_iou2(
                        np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
                        np.reshape(assigned_locations, (-1, 2)), np.reshape(assigned_scores,
                                                                            assigned_scores.shape[
                                                                                0] *
                                                                            assigned_scores.shape[
                                                                                1]))
                    IOU.update(iou_rate / (effective_positives), effective_positives)


                assigned_scores = Variable(torch.LongTensor(assigned_scores), requires_grad=False)
                assigned_locations = Variable(torch.LongTensor(assigned_locations), requires_grad=False)
                if use_cuda:
                    assigned_scores = assigned_scores.cuda()
                    assigned_locations = assigned_locations.cuda()

                cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])
                assigned_scores = assigned_scores.contiguous().view(-1)

                cls_loss = F.cross_entropy(cls_scores, assigned_scores, weight=cls_weights)

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

                    assigned_head_positions = to_one_hot(assigned_head_positions, 90)
                    assigned_tail_positions = to_one_hot(assigned_tail_positions, 90)

                    prediction_head_loss = EMD_L2(head_pointer_probs, assigned_head_positions, needSoftMax=True)
                    prediction_tail_loss = EMD_L2(tail_pointer_probs, assigned_tail_positions, needSoftMax=True)
                    loc_losses.update(prediction_head_loss.data.item() + prediction_tail_loss.data.item(),
                                      feature_batch.size(0))
                    total_loss = alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss
                else:
                    total_loss = cls_loss

                cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                total_losses.update(total_loss.item(), feature_batch.size(0))

            print(
                "Val -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg,
                    IOU.avg, ordered_IOU.avg))


            if epoch % 1 == 0:
                save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss':total_losses.avg,
            'cls_loss': cls_losses.avg,
            'loc_loss': loc_losses.avg,
            'IoU': IOU.avg}, (epoch+1), file_direcotry=save_directory)



def save_checkpoint(state, epoch, file_direcotry):
    filename = 'checkpoint_{:04d}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    file_path = os.path.join(file_direcotry, filename.format(epoch))
    torch.save(state, file_path)



if __name__ == '__main__':
    main()
