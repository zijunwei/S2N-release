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
from vsSummDevs.datasets import vsSumLoader as SumMeLoader
import SumMeAugLoader as AugSumMeLoader
import argparse
from PtrNet.PointerNetIndependent import PointerNet
from PtUtils import cuda_model
import PtrNet.Metrics as Metrics
from PyUtils import dir_utils
from PyUtils.AverageMeter import AverageMeter
import progressbar
from Losses import losses, loss_transforms

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--clip_size', default=200, type=int, help='length of each clip')
parser.add_argument('--loss', default='CE', type=str, help='type of loss (ce: cross entropy, emd: earth movers distance)')
# Train
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
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
    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    useEMD = False
    if args.loss == 'emd':
        print "Using EMD Loss"
        useEMD=True
    else:
        print "Using CCE Loss"

    print "clip size: {:d}".format(args.clip_size)
    # this is for training local...
    train_dataset = AugSumMeLoader.Dataset(dataset_name='TVSum', split='train', clip_size=args.clip_size)
    val_dataset = SumMeLoader.Dataset(dataset_name='TVSum', split='val', clip_size=args.clip_size)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)


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
        train_f1_1_accuracy = AverageMeter()
        train_f1_3_accuracy = AverageMeter()

        model.train()
        # train_iterator = tqdm(train_dataloader, unit='Batch')
        pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
        for i_batch, sample_batched in enumerate(train_dataloader):
            pbar.update(i_batch)
            # train_iterator.set_description('Train Batch %i/%i' % (epoch + 1, args.nof_epoch))

            train_batch = Variable(sample_batched[0])
            targetIdx_batch = Variable(sample_batched[1])
            target1Hot_batch = Variable(loss_transforms.torchT_indices2D2scores(sample_batched[1], n_classes=args.clip_size))
            if use_cuda:
                train_batch = train_batch.cuda()
                targetIdx_batch = targetIdx_batch.cuda()
                target1Hot_batch = target1Hot_batch.cuda()

            index_vector, segment_score = model(train_batch)

            index_vector = index_vector.contiguous().view(-1, args.clip_size)
            target1Hot_batch = target1Hot_batch.view(-1, args.clip_size)
            targetIdx_batch = targetIdx_batch.view(-1)
            F1 = Metrics.accuracy_F1(index_vector.data, targetIdx_batch.data, topk=[1, 3])
            accuracy = Metrics.accuracy_topN(index_vector.data, targetIdx_batch.data, topk=[1, 3])

            if useEMD:
                loc_loss = losses.EMD_L2(index_vector, target1Hot_batch, needSoftMax=True)
            else:
                loc_loss = CCE(index_vector, targetIdx_batch)
            model_optim.zero_grad()
            loc_loss.backward()
            model_optim.step()

            train_losses.update(loc_loss.data[0], train_batch.size(0))
            train_top_1_accuracy.update(accuracy[0][0], train_batch.size(0))
            train_top_3_accuracy.update(accuracy[1][0], train_batch.size(0))
            train_f1_1_accuracy.update(F1[0], train_batch.size(0))
            train_f1_3_accuracy.update(F1[1], train_batch.size(0))

        print("Train -- EpochT :{:06d}, LR: {:.6f},\tloss={:.6f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[0]['lr'], train_losses.avg, train_top_1_accuracy.avg, train_top_3_accuracy.avg,
                                                                                                                            train_f1_1_accuracy.avg, train_f1_3_accuracy.avg))
        if best_status['train_loss'] > train_losses.avg:
            best_status['train_loss']= train_losses.avg
            isBest_status['train_loss'] = 1
        if best_status['train_accuracy']<train_top_1_accuracy.avg:
            best_status['train_accuracy'] = train_top_1_accuracy.avg
            isBest_status['train_accuracy'] = 1


        model.eval()
        val_top_1_accuracy = AverageMeter()
        val_top_3_accuracy = AverageMeter()
        val_losses = AverageMeter()
        val_f1_1_accuracy = AverageMeter()
        val_f1_3_accuracy = AverageMeter()
        pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
        for i_batch, sample_batched in enumerate(val_dataloader):
            pbar.update(i_batch)

            test_batch = Variable(sample_batched[0])
            targetIdx_batch = Variable(sample_batched[1])
            target1Hot_batch = Variable(loss_transforms.torchT_indices2D2scores(sample_batched[1], n_classes=args.clip_size))

            if use_cuda:
                test_batch = test_batch.cuda()
                targetIdx_batch = targetIdx_batch.cuda()
                target1Hot_batch = target1Hot_batch.cuda()

            index_vector, segment_score = model(test_batch)


            index_vector = index_vector.contiguous().view(-1, args.clip_size)
            target1Hot_batch = target1Hot_batch.view(-1, args.clip_size)

            targetIdx_batch = targetIdx_batch.view(-1)
            accuracy = Metrics.accuracy_topN(index_vector.data, targetIdx_batch.data, topk=[1, 3])
            F1 = Metrics.accuracy_F1(index_vector.data, targetIdx_batch.data, topk=[1, 3])
            if useEMD:
                loc_loss = losses.EMD_L2(index_vector, target1Hot_batch, needSoftMax=True)
            else:
                loc_loss = CCE(index_vector, targetIdx_batch)

            val_losses.update(loc_loss.data[0], test_batch.size(0))
            val_top_1_accuracy.update(accuracy[0][0], test_batch.size(0))
            val_top_3_accuracy.update(accuracy[1][0], test_batch.size(0))
            val_f1_1_accuracy.update(F1[0], test_batch.size(0))
            val_f1_3_accuracy.update(F1[1], test_batch.size(0))

        print("Test :{:06d}, LR: {:.6f},\tloss={:.6f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(epoch,
                                                                                                  model_optim.param_groups[
                                                                                                      0]['lr'],
                                                                                                  val_losses.avg,
                                                                                                  val_top_1_accuracy.avg,
                                                                                                  val_top_3_accuracy.avg,val_f1_1_accuracy.avg, val_f1_3_accuracy.avg))
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
        }, isBest_status, file_direcotry='vsSum_Segment')


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



if __name__ == '__main__':
    main()