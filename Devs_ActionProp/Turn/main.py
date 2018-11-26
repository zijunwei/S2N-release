import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import thumos14_iccv17
import network
import numpy as np
import argparse
import math
from PtUtils import cuda_model
from PyUtils import dir_utils, argparse_utils
import progressbar
from PyUtils.AverageMeter import AverageMeter
import ActionLocalizationDevs.Metrics_iccv2017 as Metrics
import evaluation


parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--normalize', default=0, type=int, help='normalize')
# Train
parser.add_argument('--start_epoch', default=0, type=int, help='Staring epoch')
parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--resume', '-r', default='/home/zwei/Dev/NetModules/ActionLocalizationDevs/Turn/iccv17_baseline/checkpoint_train_accuracy.pth.tar', type=str, help='resume from previous ')
parser.add_argument('--eval', '-e', default='y', type=argparse_utils.str2bool, help='evaluate only')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_size', type=int, default=2048, help='feature size')
parser.add_argument('--hidden_size', type=int, default=1000, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout value')
# Loss
parser.add_argument('--plambda', type=float, default=2., help='lambda*reg_loss + cls_loss')
# Misc:
parser.add_argument('--branch', type=str, default='iccv17_baseline', help='running name')


def main():
    global args

    args = parser.parse_args()

    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    model = network.TURN(feature_size=args.input_size, mid_layer_size=args.hidden_size, drop=args.dropout)
    print("Number of Params \t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume is not None:
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        # args.start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loading checkpoint '{:s}', epoch: {:d}\n".format(args.resume, args.start_epoch))
    else:
        print("Training from srcatch")

    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

    feature_directory = '/home/zwei/datasets/THUMOS14/features/denseflow'
    train_clip_foreground_path = '/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/val_training_samples.txt'
    train_clip_background_path = '/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/background_samples.txt'
    val_clip_path = '/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/test_swin.txt'

    train_dataset = thumos14_iccv17.TrainDataSet(feature_directory=feature_directory,
                                                 foreground_path=train_clip_foreground_path,
                                                 background_path=train_clip_background_path, n_ctx=4,
                                                 feature_size=args.input_size)
    val_dataset = thumos14_iccv17.EvaluateDataset(feature_directory=feature_directory, clip_path=val_clip_path,
                                                  n_ctx=4, unit_size=16., feature_size=args.input_size)


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    if args.eval:
        evaluator = evaluation.Evaluator(dataloader=val_dataloader, save_directory=args.branch, savename=os.path.basename(args.resume))
        evaluator.evaluate(model, use_cuda=use_cuda)
        sys.exit(0)



    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr)


    best_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': float('inf'), 'val_loss': float('inf')}
    isBest_status = {'train_accuracy': 0, 'val_accuracy': 0, 'train_loss': 0, 'val_loss': 0}


    for epoch in range(args.start_epoch, args.nof_epoch):

        total_losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()
        Accuracy_cls = AverageMeter()
        Accuracy_loc = AverageMeter()


        model.train()
        pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
        for i_batch, sample_batched in enumerate(train_dataloader):
            pbar.update(i_batch)

            feature_batch = Variable(sample_batched[0])
            offset_batch = Variable(sample_batched[1])
            label_batch = Variable(sample_batched[2])
            clip_batch = (sample_batched[3])

            if use_cuda:
                feature_batch = feature_batch.cuda()
                offset_batch = offset_batch.cuda()
                label_batch = label_batch.cuda()
                # clip_batch = clip_batch.cuda()

            if args.normalize>0:
                feature_batch = F.normalize(feature_batch, p=2, dim=1)

            output_v = model(feature_batch)
            cls_logits, loc_logits, _, _ = network.extract_outputs(output_v)
            cls_loss = network.cls_loss(cls_logits, label_batch.long())
            loc_loss = network.loc_loss(loc_logits, offset_batch, label_batch)

            cls_accuracy = Metrics.accuracy_topN(cls_logits.data, label_batch.long().data)
            loc_accuracy, n_valid = Metrics.IoU(clip_batch.numpy(), loc_logits.data.cpu().numpy(), label_batch.data.cpu().numpy())

            total_loss = cls_loss + args.plambda * loc_loss

            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()

            total_losses.update(total_loss.data[0], feature_batch.size(0))
            cls_losses.update(cls_loss.data[0], feature_batch.size(0))
            loc_losses.update(loc_loss.data[0], feature_batch.size(0))
            Accuracy_cls.update(cls_accuracy[0][0], feature_batch.size(0))
            Accuracy_loc.update(loc_accuracy, n_valid)

        print(
            "Train -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\tcls-loss={:.4f}\tCls-Accuracy={:.4f}\tIoU={:.4f}".format(
                epoch,
                model_optim.param_groups[0]['lr'], loc_losses.avg, cls_losses.avg, Accuracy_cls.avg, Accuracy_loc.avg))

        if best_status['train_loss'] > total_losses.avg:
            best_status['train_loss']= total_losses.avg
            isBest_status['train_loss'] = 1
        if best_status['train_accuracy']<Accuracy_cls.avg:
            best_status['train_accuracy'] = Accuracy_cls.avg
            isBest_status['train_accuracy'] = 1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_loss':best_status['val_loss'],
            'val_accuracy': best_status['val_accuracy'],
            'train_loss': best_status['train_loss'],
            'train_accuracy': best_status['train_accuracy']
        }, isBest_status, file_direcotry=args.branch)


        for item in isBest_status.keys():
            isBest_status[item]=0


        # model.eval()
        #
        # total_losses = AverageMeter()
        # loc_losses = AverageMeter()
        # cls_losses = AverageMeter()
        # Accuracy_Top1 = AverageMeter()
        # Accuracy_Top3 = AverageMeter()
        # F1_Top1 = AverageMeter()
        # F1_Top3 = AverageMeter()
        #
        # pbar = progressbar.ProgressBar(max_value=len(val_dataloader))
        # for i_batch, sample_batched in enumerate(val_dataloader):
        #     pbar.update(i_batch)
        #
        #     feature_batch = Variable(sample_batched[0])
        #     offset_batch = Variable(sample_batched[1])
        #     label_batch = Variable(sample_batched[2])
        #
        #     if use_cuda:
        #         feature_batch = feature_batch.cuda()
        #         offset_batch = offset_batch.cuda()
        #         label_batch = label_batch.cuda()
        #
        #     index_vector, segment_score = model(feature_batch)
        #     segment_indices = loss_transforms.torchVT_scores2indices(index_vector)
        #     overlap = loss_transforms.IoU_Overlaps(segment_indices.data, offset_batch.data)
        #
        #     overlap = Variable(overlap, requires_grad=False)
        #     if use_cuda:
        #         overlap = overlap.cuda()
        #
        #     cls_loss = losses.ClsLocLoss_Regression(segment_score, label_batch, overlap, thres=0.5)
        #
        #     index_vector = index_vector.contiguous().view(-1, args.clip_size)
        #     offset_batch = offset_batch.view(-1)
        #     accuracy = Metrics.accuracy_topN(index_vector.data, offset_batch.data, topk=[1, 3])
        #     F1 = Metrics.accuracy_F1(index_vector.data, offset_batch.data, topk=[1, 3])
        #     loc_loss = NLL(torch.log(index_vector), offset_batch)
        #
        #     total_loss = cls_loss + loc_loss
        #
        #     cls_losses.update(cls_loss.data[0], feature_batch.size(0))
        #     loc_losses.update(loc_loss.data[0], feature_batch.size(0))
        #     total_losses.update(total_loss.data[0], feature_batch.size(0))
        #     Accuracy_Top1.update(accuracy[0][0], feature_batch.size(0))
        #     Accuracy_Top3.update(accuracy[1][0], feature_batch.size(0))
        #     F1_Top1.update(F1[0], feature_batch.size(0))
        #     F1_Top3.update(F1[1], feature_batch.size(0))
        #
        # print(
        #     "Test -- Epoch :{:06d}, LR: {:.6f},\tloc-loss={:.4f}\tcls-loss={:.4f}\ttop1={:.4f}\ttop3={:.4f}\tF1_1={:.4f}\tF1_3={:.4f}".format(
        #         epoch,
        #         model_optim.param_groups[
        #             0]['lr'],
        #         loc_losses.avg, cls_losses.avg,
        #         Accuracy_Top1.avg,
        #         Accuracy_Top3.avg, F1_Top1.avg, F1_Top3.avg))
        #
        #
        # # val_F1_score = val_evaluator.EvaluateTop1(model, use_cuda)
        # # print "Val F1 Score: {:f}".format(val_F1_score)
        # if best_status['val_loss'] > total_losses.avg:
        #     best_status['val_loss'] = total_losses.avg
        #     isBest_status['val_loss']=1
        # if best_status['val_accuracy'] < Accuracy_Top1.avg:
        #     best_status['val_accuracy'] = Accuracy_Top1.avg
        #     isBest_status['val_accuracy']=1
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'val_loss':best_status['val_loss'],
        #     'val_accuracy': best_status['val_accuracy'],
        #     'train_loss': best_status['train_loss'],
        #     'train_accuracy': best_status['train_accuracy']
        # }, isBest_status, file_direcotry='MINIST_Dep_Combine')
        #
        #
        # for item in isBest_status.keys():
        #     isBest_status[item]=0



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