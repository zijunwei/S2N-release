# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import argparse
import ResNetUtils
import PtUtils.cuda_model as cuda_model
import glob
import DatasetUtils
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

def main():
    # load the class label
    args = parser.parse_args()

    val_transform = ResNetUtils.Res50Places_val_transform()
    useCuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    model = ResNetUtils.getRes50PlacesModel(eval=True, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    image_directory = '/home/zwei/datasets/imagenet12/val/n01440764'
    image_files = glob.glob(os.path.join(image_directory, '*.JPEG'))
    for s_image in image_files:
        image = Image.open(s_image).convert('RGB')

        image = val_transform(image).unsqueeze(0)
        if useCuda:
            image = image.cuda()

        input_image = Variable(image, volatile=True)
        preds = model(input_image)
        DatasetUtils.decode_predictions(preds.data.cpu().numpy(), verbose=True)


if __name__ == '__main__':
    main()