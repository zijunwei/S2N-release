#!/usr/bin/env python
# debug version 1:
# Added the hungarian situation
# added scores, use hunguarin to assign the scores.
# following the order: cls, rank, IOU
import argparse
import os
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
from PointerLSTM2Heads import PointerNetwork
from ActionLocalizationDevs.datasets.THUMOS14.dataloader_c3d import THUMOST14
from PtUtils import cuda_model
from Losses import h_assign
import progressbar
from PyUtils.AverageMeter import AverageMeter
import Metrics
from PyUtils import dir_utils
import random

def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--seq_len', default=100, type=int, help='clip size')
parser.add_argument('--n_outputs', default=2, type=int, help='number of intervals')
parser.add_argument('--net_outputs', default=10, type=int, help='number of intervals for lstm outputs')

# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=500, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--eval', '-e', default='y', type=str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='1', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=4096, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=4096, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units')
# parser.add_argument('--resume', type=str, default='/home/zwei/Dev/NetModules/PtrNet2/lstm2heads_proposal_ckpts/checkpoint_{:04d}.pth.tar', help='check-point file')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')


def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    # Pretty print the run args
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))
    if args.resume is not None:

        ckpt_idx = 711

        ckpt_filename = args.resume.format(ckpt_idx)
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        train_iou = checkpoint['IoU']
        args.start_epoch = checkpoint['epoch']

        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))


    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    pdefined_seq_len = [50, 100, 150]
    data_loaders_mscale = []
    for s_len in pdefined_seq_len:
        train_dataset = THUMOST14(seq_length=s_len)


        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=4)
        data_loaders_mscale.append(train_dataloader)




    model_optim = optim.Adam(model.parameters(), lr=float(args.lr))




    alpha=1.

    for epoch in range(args.start_epoch, args.nof_epoch+args.start_epoch):
            total_losses = AverageMeter()
            loc_losses = AverageMeter()
            cls_losses = AverageMeter()
            Accuracy = AverageMeter()
            IOU = AverageMeter()
            ordered_IOU = AverageMeter()
            model.train()
            loader_pick_id = random.randint(0, 2)
            seq_len = pdefined_seq_len[loader_pick_id]
            s_loader = data_loaders_mscale[loader_pick_id]

            pbar = progressbar.ProgressBar(max_value=len(s_loader))
            n_effective_batches = 0
            for i_batch, sample_batch in enumerate(s_loader):
                pbar.update(i_batch)


                feature_batch = Variable(sample_batch[0])
                start_indices = Variable(sample_batch[1])
                end_indices = Variable(sample_batch[2])
                valid_indices = Variable(sample_batch[3])

                # gt_index_batch = sample_batch[1].numpy()
                # score_batch = Variable(sample_batch[2])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    start_indices = start_indices.cuda()
                    end_indices = end_indices.cuda()

                gt_positions = torch.stack([start_indices, end_indices], dim=-1)

                head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores = model(feature_batch)

                pred_positions = torch.stack([head_positions, tail_positions], dim=-1)

                assigned_scores, assigned_locations = h_assign.Assign_Batch(gt_positions, pred_positions, valid_indices, thres=0.25)
                # correct_predictions = np.sum(assigned_scores[:,:args.n_outputs])
                # cls_rate = correct_predictions*1./np.sum(assigned_scores)

                iou_rate = Metrics.get_avg_iou(np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)),
                                               np.reshape(assigned_locations, (-1, 2)), np.reshape(assigned_scores,
                                                                                                   assigned_scores.shape[
                                                                                                       0] *
                                                                                                   assigned_scores.shape[
                                                                                                       1]))


                # _, top_assigned_locations = h_assign_proposal.Assign_Batch(gt_positions, pred_positions[:, : args.n_outputs, :], valid_indices, thres=0.5)
                #
                # ordered_iou_rate = Metrics.get_avg_iou(np.reshape(pred_positions[:,:args.n_outputs,:].data.cpu().numpy(), (-1, 2)),
                #                                np.reshape(top_assigned_locations, (-1, 2)))

                # Accuracy.update(cls_rate, np.sum(assigned_scores))

                # iou_rate = Metrics.get_avg_iou(np.reshape(pred_positions.data.cpu().numpy(), (-1, 2)), np.reshape(gt_positions.data.cpu().numpy(), (-1, 2)))
                IOU.update(iou_rate/(args.batch_size*args.n_outputs),args.batch_size*args.n_outputs)
                # ordered_IOU.update(ordered_iou_rate/(args.batch_size*args.n_outputs),args.batch_size*args.n_outputs)

                n_effective_batches += 1

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
                assigned_tail_positions = assigned_locations[:,:,0]
                assigned_tail_positions = assigned_tail_positions.contiguous().view(-1)

                head_pointer_probs = head_pointer_probs.contiguous().view(-1, head_pointer_probs.size()[-1])
                tail_pointer_probs = tail_pointer_probs.contiguous().view(-1, tail_pointer_probs.size()[-1])

                # start_indices = start_indices.contiguous().view(-1)
                # end_indices = end_indices.contiguous().view(-1)
                # with case instances....
                prediction_head_loss = F.nll_loss(torch.log(head_pointer_probs+1e-8), assigned_head_positions, reduce=False)
                prediction_head_loss = torch.mean(prediction_head_loss * assigned_scores.float())
                prediction_tail_loss = F.nll_loss(torch.log(tail_pointer_probs+1e-8), assigned_tail_positions, reduce=False)
                prediction_tail_loss = torch.mean(prediction_tail_loss * assigned_scores.float())


                total_loss = alpha * (prediction_head_loss + prediction_tail_loss) + cls_loss

                model_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
                model_optim.step()
                cls_losses.update(cls_loss.data[0], feature_batch.size(0))
                loc_losses.update(prediction_head_loss.data[0] + prediction_tail_loss.data[0], feature_batch.size(0))
                total_losses.update(total_loss.data[0], feature_batch.size(0))


            print(
                "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))
            if epoch % 10 == 0:
                save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss':total_losses.avg,
            'cls_loss': cls_losses.avg,
            'loc_loss': loc_losses.avg,
            'IoU': IOU.avg}, (epoch+1), file_direcotry='lstm2heads_proposal_ckpts_non_repeat')


def save_checkpoint(state, epoch, file_direcotry):
    filename = 'checkpoint_{:04d}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    file_path = os.path.join(file_direcotry, filename.format(epoch))
    torch.save(state, file_path)



if __name__ == '__main__':
    main()
