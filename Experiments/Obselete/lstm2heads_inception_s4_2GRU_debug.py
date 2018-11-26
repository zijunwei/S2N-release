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
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SDN.PointerGRU2Heads_v4_3in import PointerNetwork
from ActionLocalizationDevs.datasets.THUMOS14.dataloader_inception_aug_fast import THUMOST14
from PtUtils import cuda_model
from  Losses import h_assign as h_match
import progressbar
from PyUtils.AverageMeter import AverageMeter
import Losses.Metrics


def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
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
parser.add_argument('--input_dim', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=1024, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
# parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/PtrNet2/gru2heads_proposal_s4-3_EMD_AUG_ckpt/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')


def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, dropout=0.5)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    # save_directory = 'gru2heads_proposal_s4-2_ckpt'
    if args.resume is not None:

        ckpt_idx = 2

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        args.start_epoch = checkpoint['epoch']

        train_iou = checkpoint['IoU']
        train_tloss = checkpoint['loss']
        train_cls_loss = checkpoint['cls_loss']
        train_loc_loss = checkpoint['loc_loss']

        print("=> loading checkpoint '{}', total loss: {:.04f},\t cls_loss: {:.04f},\t loc_loss: {:.04f},"
              " \tcurrent iou: {:.04f}".format(ckpt_filename, train_tloss, train_cls_loss, train_loc_loss, train_iou))



    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    train_dataset = THUMOST14(seq_length=args.seq_len, overlap=0.9, sample_rate=[1, 2, 4], dataset_split='val')


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
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
            #Update here!
            model.eval()
            pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
            for i_batch, sample_batch in enumerate(train_dataloader):
                # pbar.update(i_batch)


                feature_batch = Variable(sample_batch[0])
                start_indices = Variable(sample_batch[1])
                end_indices = Variable(sample_batch[2])
                valid_indices = Variable(sample_batch[3])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()
                #
                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

                assigned_scores, assigned_locations = h_match.Assign_Batch(gt_positions, pred_positions, valid_indices, thres=0.5)
                if valid_indices.byte().any()>0:
                    print "Output at {:d}".format(i_batch)
                    # n_valid = valid_indices.data[0, 0]
                    view_idx = valid_indices.nonzero()[0][0].item()
                    n_valid = valid_indices[view_idx,0].item()
                    print"GT:"
                    print(gt_positions[view_idx,:n_valid,:])
                    print("Pred")
                    print (pred_positions[view_idx])
                    _, head_sort = head_pointer_probs[view_idx, 0, :].sort()
                    _, tail_sort = tail_pointer_probs[view_idx, 0, :].sort()
                    print("END of {:d}".format(i_batch))
                #
                    iou_rate, effective_positives = Losses.Metrics.get_avg_iou2(np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
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

                    assigned_head_positions = assigned_locations[:,:,0]
                    assigned_head_positions = assigned_head_positions.contiguous().view(-1)
                    #
                    assigned_tail_positions = assigned_locations[:,:,1]
                    assigned_tail_positions = assigned_tail_positions.contiguous().view(-1)

                    head_pointer_probs = head_pointer_probs.contiguous().view(-1, head_pointer_probs.size()[-1])
                    tail_pointer_probs = tail_pointer_probs.contiguous().view(-1, tail_pointer_probs.size()[-1])

                    # start_indices = start_indices.contiguous().view(-1)
                    # end_indices = end_indices.contiguous().view(-1)
                    # with case instances....
                    prediction_head_loss = F.cross_entropy((head_pointer_probs), assigned_head_positions, reduce=False)
                    prediction_head_loss = torch.mean(prediction_head_loss * assigned_scores.float())
                    prediction_tail_loss = F.cross_entropy((tail_pointer_probs), assigned_tail_positions, reduce=False)
                    prediction_tail_loss = torch.mean(prediction_tail_loss * assigned_scores.float())


                    total_loss = alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss
                #
                # # model_optim.zero_grad()
                # # total_loss.backward()
                # # torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
                # # model_optim.step()
                    cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                    loc_losses.update(prediction_head_loss.data.item() + prediction_tail_loss.data.item(), feature_batch.size(0))
                    total_losses.update(total_loss.data.item(), feature_batch.size(0))
                #
                #
                print(
                    "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                        epoch,
                        model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))
            break
                #




if __name__ == '__main__':
    main()
