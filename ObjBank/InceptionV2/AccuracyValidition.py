import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.datasets as datasets
# import torchvision.models as models
import PtUtils.cuda_model as cuda_model
import json
import model as BNInception
import BNInceptionUtils
from PyUtils.AverageMeter import AverageMeter


parser = argparse.ArgumentParser(description='Validating ImageNet InceptionV2')

# parser.add_argument('--finetune', '-f', action='store_true', help='use pre-trained model to finetune')
# parser.add_argument('--model', default='Orig', help='Type of model [Orig, RRSVM]')
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256 for others, 32 for Inception-V3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--n_epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    model_name = 'InceptionV2'
    model = BNInception.bninception(pretrained=True)

    print("Number of Params in {:s}\t{:d}".format(model_name, sum([p.data.nelement() for p in model.parameters()])))
    model = cuda_model.convertModel2Cuda(model, args.gpu_id, args.multiGpu)
    model.eval()


    criterion = nn.CrossEntropyLoss()

    user_root = os.path.expanduser('~')
    dataset_path = os.path.join(user_root, 'datasets/imagenet12')
    # traindir = os.path.join(dataset_path, 'train')
    valdir = os.path.join(dataset_path, 'val')

    class_idx = json.load( open(os.path.join(dataset_path, "imagenet_class_index.json")))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    # for idx in out[0].sort()[1][-10:]:
    #     print(idx2label[idx])

    # valname = os.path.join(dataset_path, 'imagenet1000_clsid_to_human.txt')
    val_transform = BNInceptionUtils.get_val_transform()



    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, use_cuda, scale256=True)
    return



def validate(val_loader, model, criterion, useCuda, scale256=False, RGBInverse=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if useCuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)

        if RGBInverse:
            input_var = input_var[:, [2, 1, 0], :, :]

        if scale256:
            input_var = input_var * 256
        target_var = torch.autograd.Variable(target)

        # compute output
        output, feature = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))


        # ImageNetUtils.decode_predsngts(output.data.cpu().numpy(), target.cpu().numpy())

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg








def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output: N by D,  target N by 1 long tensor where 0<target_i<D
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()