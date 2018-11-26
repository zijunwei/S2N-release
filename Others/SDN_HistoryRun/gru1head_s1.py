#!/usr/bin/env python
#Update: Compared to v4, now the layer is bidirectional and set some learnable init states for encoder

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
from PointerSeqConv1Head import PointerNetwork
from ActionLocalizationDevs.datasets.THUMOS14.dataloader_c3d_1pointer import THUMOST14
from PtUtils import cuda_model
from Losses import h_assign
import progressbar
from PyUtils.AverageMeter import AverageMeter
import Metrics
from PyUtils import dir_utils
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--seq_len', default=27, type=int, help='clip size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--eval', '-e', default='y', type=str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='1', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=4096, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=512, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
# parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/PtrNet2/gru2heads_proposal_s4-2_ckpt-2/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')


def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    # Pretty print the run args
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim, max_decoding_len=args.net_outputs, dropout=0.5)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    save_directory = 'gru1head_s1_ckpt'
    if args.resume is not None:

        ckpt_idx = 9

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        train_iou = checkpoint['IoU']
        args.start_epoch = checkpoint['epoch']

        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))


    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    train_dataset = THUMOST14(seq_length=args.seq_len, overlap=3, sample_rate=1, dataset_split='train')
    val_dataset = THUMOST14(seq_length=args.seq_len, overlap=3, sample_rate=1, dataset_split='val')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    model_optim = optim.Adam(filter(lambda p:p.requires_grad,  model.parameters()), lr=float(args.lr))
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min')



    alpha=0.1

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
                cur_batch_size = feature_batch.shape[0]
                pointer_positions = sample_batch[1]
                valid_indices = sample_batch[2]
                valid_indicators = torch.zeros([cur_batch_size, args.net_outputs]).long()
                assigned_positions = torch.zeros([cur_batch_size, args.net_outputs]).long()

                for batch_idx in range(cur_batch_size):
                    bounded_valid_idx = min(valid_indices[batch_idx, 0], args.net_outputs)
                    valid_indicators[batch_idx,:bounded_valid_idx]=1
                    assigned_positions[batch_idx,:bounded_valid_idx] = pointer_positions[batch_idx,:bounded_valid_idx]

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    assigned_positions = assigned_positions.cuda()
                    valid_indicators = valid_indicators.cuda()

                pred_pointer_probs, pred_positions, cls_scores = model(feature_batch)

                valid_indicators = valid_indicators.contiguous().view(-1)
                assigned_positions = assigned_positions.contiguous().view(-1)
                cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])


                cls_loss = F.cross_entropy(cls_scores, valid_indicators)

                if torch.sum(valid_indicators)>0:
                    pred_pointer_probs = pred_pointer_probs.contiguous().view(-1, pred_pointer_probs.size()[-1])



                    assigned_positions = torch.masked_select(assigned_positions, valid_indicators.byte())

                    pred_pointer_probs = torch.index_select(pred_pointer_probs, dim=0, index=valid_indicators.nonzero().squeeze(1))
                    prediction_head_loss = F.cross_entropy((pred_pointer_probs), assigned_positions)
                    loc_losses.update(prediction_head_loss.data.item(),
                                      feature_batch.size(0))
                    total_loss = alpha * (prediction_head_loss) + cls_loss
                else:
                    total_loss = cls_loss

                model_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                model_optim.step()
                cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                total_losses.update(total_loss.data.item(), feature_batch.size(0))


            print(
                "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))

            if epoch % 1 == 0:
                save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss':total_losses.avg,
            'cls_loss': cls_losses.avg,
            'loc_loss': loc_losses.avg,
            'IoU': IOU.avg}, (epoch+1), file_direcotry=save_directory)
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
                cur_batch_size = feature_batch.shape[0]

                pointer_positions = sample_batch[1]
                valid_indices = sample_batch[2]
                valid_indicators = torch.zeros([cur_batch_size, args.net_outputs]).long()
                assigned_positions = torch.zeros([cur_batch_size, args.net_outputs]).long()

                for batch_idx in range(cur_batch_size):
                    bounded_valid_idx = min(valid_indices[batch_idx, 0], args.net_outputs)
                    valid_indicators[batch_idx,:bounded_valid_idx]=1
                    assigned_positions[batch_idx,:bounded_valid_idx] = pointer_positions[batch_idx,:bounded_valid_idx]

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    assigned_positions = assigned_positions.cuda()
                    valid_indicators = valid_indicators.cuda()

                pred_pointer_probs, pred_positions, cls_scores = model(feature_batch)

                valid_indicators = valid_indicators.contiguous().view(-1)
                assigned_positions = assigned_positions.contiguous().view(-1)
                cls_scores = cls_scores.contiguous().view(-1, cls_scores.size()[-1])

                cls_loss = F.cross_entropy(cls_scores, valid_indicators)

                if torch.sum(valid_indicators) > 0:
                    pred_pointer_probs = pred_pointer_probs.contiguous().view(-1, pred_pointer_probs.size()[-1])

                    assigned_positions = torch.masked_select(assigned_positions, valid_indicators.byte())

                    pred_pointer_probs = torch.index_select(pred_pointer_probs, dim=0,
                                                            index=valid_indicators.nonzero().squeeze(1))
                    prediction_head_loss = F.cross_entropy((pred_pointer_probs), assigned_positions)
                    loc_losses.update(prediction_head_loss.data.item(),
                                      feature_batch.size(0))
                    total_loss = alpha * (prediction_head_loss) + cls_loss
                else:
                    total_loss = cls_loss


                cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                total_losses.update(total_loss.data.item(), feature_batch.size(0))

            print(
                "Val -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg,
                    IOU.avg, ordered_IOU.avg))



def save_checkpoint(state, epoch, file_direcotry):
    filename = 'checkpoint_{:04d}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    file_path = os.path.join(file_direcotry, filename.format(epoch))
    torch.save(state, file_path)



if __name__ == '__main__':
    main()
