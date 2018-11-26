import argparse

import numpy as np
import torch
from torch.autograd import Variable

import I3DUtils
import DatasetUtils
from PtUtils import cuda_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inception V3 Training')

parser.add_argument("--gpu_id", default=0, type=str)
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


if __name__ == '__main__':

    args = parser.parse_args()
    _SAMPLE_PATHS = {
        'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
        'flow': 'data/v_CricketShot_g04_c01_flow.npy',
    }
    useCuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    rgb_sample = torch.from_numpy(np.load(_SAMPLE_PATHS['rgb']))
    rgb_sample = rgb_sample.permute(0, 4, 1, 2, 3)

    model = I3DUtils.getK_I3D_RGBModel(True, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    if useCuda:
        rgb_sample = rgb_sample.cuda()

    rgb_sample = Variable(rgb_sample, volatile=True)

    scores, logits = model(rgb_sample)
    DatasetUtils.decode_predictions(logits.data.cpu().numpy(), verbose=True)



