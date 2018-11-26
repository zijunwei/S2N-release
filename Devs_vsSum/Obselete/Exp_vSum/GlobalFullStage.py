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
from vsSummDevs.datasets import vsSumLoader
import argparse
from PtrNet.PointerNetIndependent import PointerNet
from PtUtils import cuda_model
import PtrNet.Metrics as Metrics
from PyUtils import dir_utils
import progressbar
from Losses import losses
from Losses import loss_transforms
import vsSum_Global_Evaluator
parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data

parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--clip_size', default=200, type=int, help='length of each clip')

# Train
parser.add_argument('--nof_epoch', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument("--gpu_id", default='1', type=str)
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
    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    overlap_ratio = 0.7


    train_dataset = vsSumLoader.Dataset(dataset_name='TVSum', split='train', clip_size=args.clip_size, output_score=True)

    val_dataset = vsSumLoader.Dataset(dataset_name='TVSum', split='val', clip_size=args.clip_size, output_score=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    train_evaluator = vsSum_Global_Evaluator.Evaluator(dataset_name='TVSum', split='train', clip_size=args.clip_size)
    val_evaluator = vsSum_Global_Evaluator.Evaluator(dataset_name='TVSum', split='val', clip_size=args.clip_size)


    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr)


    best_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': float('inf'), 'val_loss': float('inf')}
    isBest_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': 0, 'val_loss': 0}
    for epoch in range(args.nof_epoch):
        train_top_1_accuracy = AverageMeter()
        train_top_3_accuracy = AverageMeter()
        train_losses = AverageMeter()
        train_loc_losses = AverageMeter()
        train_cls_losses = AverageMeter()

        train_f1_1_accuracy = AverageMeter()
        train_f1_3_accuracy = AverageMeter()

        model.train()
        # train_iterator = tqdm(train_dataloader, unit='Batch')
        pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
        for i_batch, sample_batched in enumerate(train_dataloader):
            pbar.update(i_batch)
            # train_iterator.set_description('Train Batch %i/%i' % (epoch + 1, args.nof_epoch))

            train_batch = Variable(sample_batched[0])
            target_batch = Variable(sample_batched[1])
            target_score_batch = Variable(sample_batched[2])

            if use_cuda:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()
                target_score_batch = target_score_batch.cuda()
            index_vector, segment_score = model(train_batch)

            segment_indices = loss_transforms.torchVT_scores2indices(index_vector)
            overlap = loss_transforms.IoU_OverlapsHardThres(segment_indices.data, target_batch.data, thres=overlap_ratio)
            #TODO: here you convert to CPU...
            overlap = Variable(overlap.cpu(), requires_grad=False)
            if use_cuda:
                overlap = overlap.cuda()
            cls_loss = losses.WeightedMSE(segment_score, target_score_batch, overlap)
            # cls_loss = MSE(segment_score, target_score_batch)

            index_vector = index_vector.contiguous().view(-1, args.clip_size)

            target_batch = target_batch.view(-1)

            accuracy = Metrics.accuracy_topN(index_vector.data, target_batch.data, topk=[1, 3])
            F1 = Metrics.accuracy_F1(index_vector.data, target_batch.data, topk=[1, 3])

            loc_loss = CCE(index_vector, target_batch)
            # if math.isnan(loss.data[0]):
            #     print"Is Nan"
            total_loss = loc_loss + cls_loss

            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()

            train_losses.update(total_loss.data[0], train_batch.size(0))
            train_loc_losses.update(loc_loss.data[0], train_batch.size(0))
            train_cls_losses.update(cls_loss.data[0], train_batch.size(0))
            train_top_1_accuracy.update(accuracy[0][0], train_batch.size(0))
            train_top_3_accuracy.update(accuracy[1][0], train_batch.size(0))
            train_f1_1_accuracy.update(F1[0], train_batch.size(0))
            train_f1_3_accuracy.update(F1[1], train_batch.size(0))

        print("Train -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\tcls-loss={:.4f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[0]['lr'], train_loc_losses.avg, train_cls_losses.avg, train_top_1_accuracy.avg, train_top_3_accuracy.avg,
                                                                                                                            train_f1_1_accuracy.avg, train_f1_3_accuracy.avg))
        if best_status['train_loss'] > train_losses.avg:
            best_status['train_loss']= train_losses.avg
            isBest_status['train_loss'] = 1
        if best_status['train_accuracy']<train_top_1_accuracy.avg:
            best_status['train_accuracy'] = train_top_1_accuracy.avg
            isBest_status['train_accuracy'] = 1


        model.eval()
        train_F1_score = train_evaluator.EvaluateTopK(model, use_cuda=use_cuda)
        print "Train F1 Score: {:f}".format(train_F1_score)
        # val_iterator = tqdm(val_dataloader, unit='Batch')
        val_top_1_accuracy = AverageMeter()
        val_top_3_accuracy = AverageMeter()
        val_losses = AverageMeter()
        val_loc_losses = AverageMeter()
        val_cls_losses = AverageMeter()
        val_f1_1_accuracy = AverageMeter()
        val_f1_3_accuracy = AverageMeter()
        pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
        for i_batch, sample_batched in enumerate(val_dataloader):
            pbar.update(i_batch)

            test_batch = Variable(sample_batched[0])
            target_batch = Variable(sample_batched[1])
            target_score_batch =Variable(sample_batched[2])

            if use_cuda:
                test_batch = test_batch.cuda()
                target_batch = target_batch.cuda()
                target_score_batch = target_score_batch.cuda()

            index_vector, segment_score = model(test_batch)
            segment_indices = loss_transforms.torchVT_scores2indices(index_vector)

            overlap = loss_transforms.IoU_OverlapsHardThres(segment_indices.data, target_batch.data, thres=overlap_ratio)

            overlap = Variable(overlap.cpu(), requires_grad=False)
            if use_cuda:
                overlap = overlap.cuda()
            cls_loss = losses.WeightedMSE(segment_score, target_score_batch, overlap)


            index_vector = index_vector.contiguous().view(-1, args.clip_size)
            target_batch = target_batch.view(-1)
            accuracy = Metrics.accuracy_topN(index_vector.data, target_batch.data, topk=[1, 3])
            F1 = Metrics.accuracy_F1(index_vector.data, target_batch.data, topk=[1, 3])
            loc_loss = CCE(index_vector, target_batch)

            # Here adjust the ratio based on loss values...
            total_loss = 0.1* cls_loss + loc_loss

            val_cls_losses.update(cls_loss.data[0], test_batch.size(0))
            val_loc_losses.update(loc_loss.data[0], test_batch.size(0))

            val_losses.update(total_loss.data[0], test_batch.size(0))
            val_top_1_accuracy.update(accuracy[0][0], test_batch.size(0))
            val_top_3_accuracy.update(accuracy[1][0], test_batch.size(0))
            val_f1_1_accuracy.update(F1[0], test_batch.size(0))
            val_f1_3_accuracy.update(F1[1], test_batch.size(0))

        print("Test -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\tcls-loss={:.4f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[
                                                                                                      0]['lr'],
                                                                                                  val_loc_losses.avg, val_cls_losses.avg,
                                                                                                  val_top_1_accuracy.avg,
                                                                                                  val_top_3_accuracy.avg,val_f1_1_accuracy.avg, val_f1_3_accuracy.avg))

        val_F1_score = val_evaluator.EvaluateTopK(model, use_cuda=use_cuda, topK=3)
        print "Val F1 Score: {:f}".format(val_F1_score)
        if best_status['val_loss'] > val_losses.avg:
            best_status['val_loss'] = val_losses.avg
            isBest_status['val_loss']=1
        if best_status['val_accuracy'] < val_top_1_accuracy.avg:
            best_status['val_accuracy'] = val_top_3_accuracy.avg
            isBest_status['val_accuracy']=1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_loss':best_status['val_loss'],
            'val_accuracy': best_status['val_accuracy'],
            'train_loss': best_status['train_loss'],
            'train_accuracy': best_status['train_accuracy']
        }, isBest_status, file_direcotry='GlobalRaw')


        for item in isBest_status.keys():
            isBest_status[item]=0



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
        file_path = os.path.join(file_direcotry, filename.format('val_loss'))
        torch.save(state, file_path)
    if is_best_status['train_loss']:
        file_path = os.path.join(file_direcotry, filename.format('val_loss'))
        torch.save(state, file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()