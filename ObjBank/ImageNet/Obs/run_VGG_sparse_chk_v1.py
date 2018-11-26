import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
# import torch.utils.data.distributed
import torchvision.transforms as transforms

import PtUtils.cuda_model as cuda_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inception V3 Training')
import FeatureBank.ImageNet.Obs.Analyze_Activation

# parser.add_argument('--finetune', '-f', action='store_true', help='use pre-trained model to finetune')
# parser.add_argument('--model', default='Orig', help='Type of model [Orig, RRSVM]')
parser.add_argument("--gpu_id", default=0, type=str)
parser.add_argument('--positive_constraint', '-p', action='store_true', help='positivity constraint')
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch-size', default=1, type=int,
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



class SubModel(nn.Module):
    def __init__(self, orig_model, feat=1):
        super(SubModel, self).__init__()
        self.features = nn.Sequential(*list(orig_model.features.children())[:feat])
    def forward(self, x):
        x = self.features(x)
        return x


def main():
    global args, best_prec1
    args = parser.parse_args()


    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    model_name = 'VGG16_BN'
    model = models.vgg16_bn(pretrained=False)

    sub_model = SubModel(model, feat=1)

    print("Number of Params in {:s}\t{:d}".format(model_name, sum([p.data.nelement() for p in sub_model.parameters()])))
    model = cuda_model.convertModel2Cuda(sub_model, args.gpu_id, args.multiGpu)
    model.eval()

    # centre_crop = transforms.Compose([
    #     transforms.Scale(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # img_name = '12.jpg'
    # if not os.access(img_name, os.W_OK):
    #     img_url = 'http://places.csail.mit.edu/demo/' + img_name
    #     os.system('wget ' + img_url)
    #
    # img = Image.open(img_name)
    # input_img = Variable(centre_crop(img).unsqueeze(0).cuda(), volatile=True)
    #
    # # forward pass
    # logit = model.forward(input_img)
    # s_activation = logit.data.cpu().numpy()
    # Analyze_Activation.CheckActivation(s_activation, 'HAHA')


    #
    #
    # # compute the average response after each of the Conv Layer, could slightly modify to check other layers...
    #
    #
    # for s_name, s_activation in outputs.items():
    #     if 'Conv2d' not in s_name:
    #         continue
    #     else:
    #
    #         s_activation = s_activation.data.cpu().numpy()
    #         Analyze_Activation.CheckActivation(s_activation, s_name)
    #         # mean_activation = np.mean(s_activation, axis=2)
    #         # mean_activation = np.mean(mean_activation, axis=2)
    #
    #         # print "DEBUG"
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,  model.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # p_constraint = False
    # if args.positive_constraint:
    #     p_constraint = True


    # if use_cuda:
    #     if args.multiGpu:
    #         if args.gpu_id is None: # using all the GPUs
    #             device_count = torch.cuda.device_count()
    #             print("Using ALL {:d} GPUs".format(device_count))
    #             model = nn.DataParallel(model, device_ids=[i for i in range(device_count)]).cuda()
    #         else:
    #             print("Using GPUs: {:s}".format(args.gpu_id))
    #             device_ids = [int(x) for x in args.gpu_id]
    #             model = nn.DataParallel(model, device_ids=device_ids).cuda()
    #
    #
    #     else:
    #         torch.cuda.set_device(int(args.gpu_id))
    #         model.cuda()
    #
    #     criterion.cuda()
    #     cudnn.benchmark = True

    # global save_dir
    # save_dir = './snapshots/ImageNet_Inceptionv3_{:s}'.format(args.model.upper())
    # if args.positive_constraint:
    #     save_dir = save_dir + '_p'
    # if args.finetune:
    #     save_dir = save_dir + '_finetune'
    #
    # save_dir = dir_utils.get_dir(save_dir)

    # optionally resume from a checkpoint
    # if args.resume:
    #
    #     # if os.path.isfile(args.resume):
    #     ckpt_filename = 'model_best.ckpt.t7'
    #     assert os.path.isfile(os.path.join(save_dir, ckpt_filename)), 'Error: no checkpoint directory found!'
    #
    #     checkpoint = torch.load(os.path.join(save_dir, ckpt_filename), map_location=lambda storage, loc: storage)
    #     args.start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # TODO: check how to load optimizer correctly
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loading checkpoint '{}', epoch: {:d}".format(ckpt_filename, args.start_epoch))
    #
    # else:
    #     print('==> Training with NO History..')
    #     if os.path.isfile(os.path.join(save_dir, 'log.txt')):
    #         os.remove(os.path.join(save_dir, 'log.txt'))

    user_root = os.path.expanduser('~')
    dataset_path = os.path.join(user_root, 'datasets/imagenet12')
    # traindir = os.path.join(dataset_path, 'train')
    valdir = os.path.join(dataset_path, 'val')

    class_idx = json.load( open(os.path.join(dataset_path, "imagenet_class_index.json")))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    # for idx in out[0].sort()[1][-10:]:
    #     print(idx2label[idx])

    # valname = os.path.join(dataset_path, 'imagenet1000_clsid_to_human.txt')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    validate(val_loader, model, criterion, use_cuda)
    return

    # for epoch in range(args.start_epoch, args.n_epochs):
    #     # if args.distributed:
    #     #     train_sampler.set_epoch(epoch)
    #     adjust_learning_rate(optimizer, epoch, args.finetune)
    #
    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch, p_constraint, use_cuda)
    #
    #     # evaluate on validation set
    #     prec1, prec5 = validate(val_loader, model, criterion, use_cuda)
    #
    #     # remember best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'prec1': prec1,
    #         'prec5': prec5,
    #         'optimizer': optimizer.state_dict(),
    #     }, is_best, filename=os.path.join(save_dir, '{:04d}_checkpoint.pth.tar'.format(epoch)))


# def train(train_loader, model, criterion, optimizer, epoch, p_constraint, use_cuda):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     if p_constraint:
#         positive_clipper = RRSVM.RRSVM_PositiveClipper()
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#         if use_cuda:
#             target = target.cuda()
#             input = input.cuda()
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
#
#         # compute output
#         output, output_aux = model(input_var)
#         loss = criterion(output, target_var)
#         loss_aux = criterion(output, target_var)
#         # TODO: here check how to merge aux loss
#         t_loss = loss + loss_aux
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0]+loss_aux.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         t_loss.backward()
#         optimizer.step()
#
#         if p_constraint and positive_clipper.frequency % (i+1) == 0:
#             model.apply(positive_clipper)
#         # measure elapsed time
#
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    epoch, i, len(train_loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, top1=top1, top5=top5))
#         # #DBUGE
#         # if i % 100 == 0:
#         #     save_checkpoint({
#         #         'epoch': epoch + 1,
#         #         'state_dict': model.state_dict(),
#         #         'prec1': prec1,
#         #         'prec5': prec5,
#         #         'optimizer': optimizer.state_dict(),
#         #     }, False, filename=os.path.join(save_dir, '{:04d}_checkpoint.pth.tar'.format(epoch)))
#

def validate(val_loader, model, criterion, useCuda):
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
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        FeatureBank.ImageNet.Obs.Analyze_Activation.CheckActivation(output.data.cpu().numpy(), 'HAHA')
        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #
    #     if i % args.print_freq == 0:
    #         print('Test: [{0}/{1}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                i, len(val_loader), batch_time=batch_time, loss=losses,
    #                top1=top1, top5=top5))
    #
    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    #
    # return top1.avg, top5.avg


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, isFinetune=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if isFinetune:
        every_n = 5
    else:
        every_n = 30

    lr = args.lr * (0.1 ** (epoch // every_n))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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