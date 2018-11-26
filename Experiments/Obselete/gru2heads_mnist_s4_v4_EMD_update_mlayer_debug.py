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
from Exp_Synthetic2.Minist_Apperance.CreateMNIST_2Heads_online_v4 import MNIST
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
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--seq_len', default=100, type=int, help='clip size')
parser.add_argument('--net_outputs', default=10, type=int, help='number of intervals for lstm outputs')

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--eval', '-e', default='y', type=str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=784, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=784, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
# parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/PtrNet2/gru2heads_mnist_s4-v4_EMD_mlayer_ckpt/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')


def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    # Pretty print the run args
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, dropout=0.5, n_enc_layers=2)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    save_directory = 'gru2heads_mnist_s4-v4_EMD_mlayer_ckpt'
    if args.resume is not None:

        ckpt_idx = 75

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        train_iou = checkpoint['IoU']
        args.start_epoch = checkpoint['epoch']

        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))

    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    train_dataset = MNIST(dataset_split='train')
    val_dataset =MNIST(dataset_split='val')


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    model_optim = optim.Adam(filter(lambda p:p.requires_grad,  model.parameters()), lr=float(args.lr))
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min')

    alpha=0.1
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
                gt_valids = Variable(sample_batch[4])
                seq_labels = Variable(sample_batch[3])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()

                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

                assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions, gt_valids, thres=0.5)
                print "Output at {:d}".format(i_batch)
                # n_valid = valid_indices.data[0, 0]
                # view_idx = valid_indices.nonzero()[0][0].item()
                # n_valid = valid_indices[view_idx, 0].item()
                print"GT:"
                print(assigned_locations[0])
                print("Pred")
                print (pred_positions[0])
                _, head_sort = head_pointer_probs[0, 0, :].sort()
                _, tail_sort = tail_pointer_probs[0, 0, :].sort()
                for label_idx, label in enumerate(seq_labels[0]):
                    print('{:d}\t{:d}'.format(label_idx, label.item()))
                print("END of {:d}".format(i_batch))

if __name__ == '__main__':
    main()
