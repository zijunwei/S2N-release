"""
Pytorch implementation of Pointer Network.
http://arxiv.org/pdf/1506.03134v1.pdf.
"""
import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import vsSummDevs.datasets.vsSumLoader2 as LocalDataLoader
import argparse
from PtrNet.PointerNet2PointDepOffset import PointerNet
from PtUtils import cuda_model
import PtrNet.Metrics as Metrics
from PyUtils import dir_utils, argparse_utils
import progressbar

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")
from PyUtils.AverageMeter import AverageMeter
# Data
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--clip_size', default=200, type=int, help='clip size')
# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from previous ')
parser.add_argument('--eval', '-e', default='n', type=argparse_utils.str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_size', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--hidden_size', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nlayers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')


def main():
    global args

    args = parser.parse_args()

    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    model = PointerNet(args.input_size,
                       args.hidden_size,
                       args.nlayers,
                       args.dropout,
                       args.bidir)
    if args.resume is not None:
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loading checkpoint '{:s}', epoch: {:d}".format(args.resume, args.start_epoch))

    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    train_dataset = LocalDataLoader.Dataset(dataset_name='SumMe', split='train',
                                            clip_size=args.clip_size, sample_rates=[1, 5, 10])
    val_dataset = LocalDataLoader.Dataset(dataset_name='SumMe', split='val',
                                            clip_size=args.clip_size)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    NLL = torch.nn.NLLLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr)

    best_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': float('inf'), 'val_loss': float('inf')}
    isBest_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': 0, 'val_loss': 0}
    for epoch in range(args.start_epoch, args.nof_epoch):
        total_losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()
        Accuracy_Top1 = AverageMeter()
        Accuracy_Top3 = AverageMeter()
        F1_Top1 = AverageMeter()
        F1_Top3 = AverageMeter()

        model.train()
        pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
        for i_batch, sample_batched in enumerate(train_dataloader):
            pbar.update(i_batch)

            feature_batch = Variable(sample_batched[0])
            gt_index_batch = Variable(sample_batched[1])

            if use_cuda:
                feature_batch = feature_batch.cuda()
                gt_index_batch = gt_index_batch.cuda()
            index_vector, segment_score = model(feature_batch)

            index_vector = index_vector.contiguous().view(-1, args.clip_size)
            gt_index_batch = gt_index_batch.view(-1)

            accuracy = Metrics.accuracy_topN(index_vector.data, gt_index_batch.data, topk=[1, 3])
            F1 = Metrics.accuracy_F1(index_vector.data, gt_index_batch.data, topk=[1, 3])

            loc_loss = NLL(torch.log(index_vector), gt_index_batch)

            total_loss = loc_loss

            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()

            total_losses.update(total_loss.data[0], feature_batch.size(0))
            loc_losses.update(loc_loss.data[0], feature_batch.size(0))
            Accuracy_Top1.update(accuracy[0][0], feature_batch.size(0))
            Accuracy_Top3.update(accuracy[1][0], feature_batch.size(0))
            F1_Top1.update(F1[0], feature_batch.size(0))
            F1_Top3.update(F1[1], feature_batch.size(0))

        print("Train -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[0]['lr'], loc_losses.avg, Accuracy_Top1.avg, Accuracy_Top3.avg,
                                                                                                                            F1_Top1.avg, F1_Top3.avg))
        if best_status['train_loss'] > total_losses.avg:
            best_status['train_loss']= total_losses.avg
            isBest_status['train_loss'] = 1
        if best_status['train_accuracy']<Accuracy_Top1.avg:
            best_status['train_accuracy'] = Accuracy_Top1.avg
            isBest_status['train_accuracy'] = 1

        model.eval()

        total_losses.reset()
        loc_losses.reset()
        cls_losses.reset()
        Accuracy_Top1.reset()
        Accuracy_Top3.reset()
        F1_Top1.reset()
        F1_Top3.reset()

        pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
        for i_batch, sample_batched in enumerate(val_dataloader):
            pbar.update(i_batch)

            feature_batch = Variable(sample_batched[0])
            gt_index_batch = Variable(sample_batched[1])

            if use_cuda:
                feature_batch = feature_batch.cuda()
                gt_index_batch = gt_index_batch.cuda()

            index_vector, segment_score = model(feature_batch)

            index_vector = index_vector.contiguous().view(-1, args.clip_size)
            gt_index_batch = gt_index_batch.view(-1)
            accuracy = Metrics.accuracy_topN(index_vector.data, gt_index_batch.data, topk=[1, 3])
            F1 = Metrics.accuracy_F1(index_vector.data, gt_index_batch.data, topk=[1, 3])
            loc_loss = NLL(torch.log(index_vector), gt_index_batch)

            total_loss = loc_loss

            loc_losses.update(loc_loss.data[0], feature_batch.size(0))

            total_losses.update(total_loss.data[0], feature_batch.size(0))
            Accuracy_Top1.update(accuracy[0][0], feature_batch.size(0))
            Accuracy_Top3.update(accuracy[1][0], feature_batch.size(0))
            F1_Top1.update(F1[0], feature_batch.size(0))
            F1_Top3.update(F1[1], feature_batch.size(0))

        print("Test -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[
                                                                                                      0]['lr'],
                                                                                                  loc_losses.avg,
                                                                                                  Accuracy_Top1.avg,
                                                                                                  Accuracy_Top3.avg, F1_Top1.avg, F1_Top3.avg))

        # val_F1_score = val_evaluator.EvaluateTop1(model, use_cuda)
        # print "Val F1 Score: {:f}".format(val_F1_score)
        if best_status['val_loss'] > total_losses.avg:
            best_status['val_loss'] = total_losses.avg
            isBest_status['val_loss']=1
        if best_status['val_accuracy'] < Accuracy_Top1.avg:
            best_status['val_accuracy'] = Accuracy_Top1.avg
            isBest_status['val_accuracy']=1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_loss':best_status['val_loss'],
            'val_accuracy': best_status['val_accuracy'],
            'train_loss': best_status['train_loss'],
            'train_accuracy': best_status['train_accuracy']
        }, isBest_status, file_direcotry='vsPtrDep_Localization')


        for item in isBest_status.keys():
            isBest_status[item]=0


def updateStatus(BestStatus, isBestStatus, entry_name, entry_score, isLarger=True):
    if isLarger:
        if BestStatus[entry_name] > entry_score:
            BestStatus[entry_name] = entry_score
            isBestStatus[entry_name] = 1
    else:
        if BestStatus[entry_name] < entry_score:
            BestStatus[entry_name] = entry_score
            isBestStatus[entry_name] = 1


def save_checkpoint(state, is_best_status, file_direcotry):
    filename = 'checkpoint_{:s}.pth.tar'
    file_direcotry = dir_utils.get_dir(file_direcotry)

    if is_best_status['val_accuracy']:
        file_path = os.path.join(file_direcotry, filename.format('val_accuracy'))
        torch.save(state, file_path)
    if is_best_status['val_loss']:
        file_path = os.path.join(file_direcotry, filename.format('val_loss'))
        torch.save(state, file_path)
    if is_best_status['train_accuracy']:
        file_path = os.path.join(file_direcotry, filename.format('train_accuracy'))
        torch.save(state, file_path)
    if is_best_status['train_loss']:
        file_path = os.path.join(file_direcotry, filename.format('train_loss'))
        torch.save(state, file_path)


if __name__ == '__main__':
    main()