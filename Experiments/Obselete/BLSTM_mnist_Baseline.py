#!/usr/bin/env python
#Update: A cleaned version of gru2heads_inception_s4_3_EMD.py

import argparse
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import torch.nn.utils.clip_grad
import torch
print(torch.__version__)
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SDN.BaseLSTM import BaseLSTMNetwork
from Devs_SyntheticV2.SDN_MNIST_s1_1_LSTM_online import MNIST #Update: output top 3!
from PtUtils import cuda_model

import progressbar
from PyUtils.AverageMeter import AverageMeter
from PyUtils import dir_utils
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')
from PyUtils import log_utils

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
parser.add_argument('--seq_len', default=100, type=int, help='clip size')
parser.add_argument('--net_outputs', default=6, type=int, help='number of intervals for lstm outputs')

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
parser.add_argument('--hassign_thres', default=0.75, type=float, help='hassignment_threshold')
parser.add_argument('--alpha', default=0.1, type=float, help='trade off between classification and localization')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate for training a network')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--hmatch', default=0, type=int, help='hungarian matching or fix matching')
parser.add_argument('--EMD', default=0, type=int, help='Using EMD loss or cls loss ')
# parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/ckpts/SDN_mnist_EMD_hmatch-assgin0.75-alpha0.1000-dim512-dropout0.5000-seqlen100-ckpt/checkpoint_{:04d}.pth.tar', type=str, help='resume from previous ')

loss_type={0: 'CLS', 1: 'EMD'}
match_type = {0: 'fix', 1: 'hug'}

def main():
    global args
    args = (parser.parse_args())
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    script_name_stem = dir_utils.get_stem(__file__)
    save_directory = dir_utils.get_dir(os.path.join(project_root, 'ckpts', '{:s}-assgin{:.2f}-alpha{:.4f}-dim{:d}-dropout{:.4f}-seqlen{:d}-{:s}-{:s}'.
                                  format(script_name_stem, args.hassign_thres, args.alpha, args.hidden_dim, args.dropout, args.seq_len, loss_type[args.EMD], match_type[args.hmatch])))
    log_file = os.path.join(save_directory, 'log-{:s}.txt'.format(dir_utils.get_date_str()))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)


    model = BaseLSTMNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, dropout=args.dropout, n_enc_layers=2)
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
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=10)

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
                labels = Variable(sample_batch[1])


                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    labels = labels.cuda()
                    # end_indices = end_indices.cuda()

                pred_labels = model(feature_batch)

                labels = labels.contiguous().view(-1)
                pred_labels = pred_labels.contiguous().view(-1, pred_labels.size()[-1])

                pred_probs = F.softmax(pred_labels, dim=1)[:, 1]
                pred_probs[pred_probs>0.5] = 1
                pred_probs[pred_probs<=0.5] = -1
                n_positives = torch.sum(labels).item()
                iou = torch.sum(pred_probs==labels.float()).item()*1. / n_positives
                IOU.update(iou, 1.)

                total_loss = F.cross_entropy(pred_labels, labels)

                model_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                model_optim.step()
                # cls_losses.update(cls_loss.data.item(), feature_batch.size(0))
                total_losses.update(total_loss.item(), feature_batch.size(0))


            logger.info(
                "Train -- Epoch :{:06d}, LR: {:.6f},\tloss={:.4f}, \t c-loss:{:.4f}, \tloc-loss:{:.4f}\tcls-Accuracy:{:.4f}\tloc-Avg-IOU:{:.4f}\t topIOU:{:.4f}".format(
                    epoch,
                    model_optim.param_groups[0]['lr'], total_losses.avg, cls_losses.avg, loc_losses.avg, Accuracy.avg, IOU.avg, ordered_IOU.avg))

            optim_scheduler.step(total_losses.avg)

            model.eval()

            IOU = AverageMeter()
            pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
            for i_batch, sample_batch in enumerate(val_dataloader):
                pbar.update(i_batch)

                feature_batch = Variable(sample_batch[0])
                labels = Variable(sample_batch[1])

                if use_cuda:
                    feature_batch = feature_batch.cuda()
                    labels = labels.cuda()

                labels = labels.contiguous().view(-1)

                pred_labels = model(feature_batch)

                pred_labels = pred_labels.contiguous().view(-1, pred_labels.size()[-1])

                pred_probs = F.softmax(pred_labels, dim=1)[:, 1]
                n_positives = torch.sum(labels).item()
                pred_probs[pred_probs > 0.5] = 1
                pred_probs[pred_probs <= 0.5] = -1

                iou = torch.sum(pred_probs == labels.float()).item() * 1. / n_positives
                IOU.update(iou, 1.)


            logger.info(
                "Val -- Epoch :{:06d}, LR: {:.6f},\tloc-Avg-IOU:{:.4f}".format(
                    epoch,model_optim.param_groups[0]['lr'], IOU.avg, ))


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
