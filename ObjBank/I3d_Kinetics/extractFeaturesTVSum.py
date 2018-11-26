import os
import sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import argparse

import PtUtils.cuda_model as cuda_model
from torch.autograd import Variable
import glob
import I3DUtils
from vsSummDevs.datasets.TVSum import path_vars
from vsSummDevs.datasets import SingleVideoFrameDataset
import torch.utils.data
from PyUtils import dir_utils
import numpy as np
import progressbar
parser = argparse.ArgumentParser(description='PyTorch ImageNet Inception V3 Training')

parser.add_argument("--gpu_id", default=1, type=str)
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
    args = parser.parse_args()

    val_transform = I3DUtils.simple_I3D_transform(224)
    useCuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    model = I3DUtils.getK_I3D_RGBModel(eval=True, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    # Buffer = {}
    # def _fc7_hook(self, input, output):
    #     Buffer['fc7'] = output.data.clone()
    #
    # model.full7.register_forward_hook(_fc7_hook)

    image_root_directory = os.path.join(path_vars.dataset_dir, 'frames')
    save_root_directory = dir_utils.get_dir(os.path.join(path_vars.dataset_dir, 'features/Kinetics/I3D-test'))
    chunk_size = 64
    image_directories = glob.glob(os.path.join(image_root_directory, '*/'))
    for idx_dir, s_image_direcotry in enumerate(image_directories):
        stem_name = s_image_direcotry.split(os.sep)[-2]
        print '[{:02d} | {:02d}] {:s}'.format(idx_dir, len(image_directories), stem_name)
        stem_name = stem_name.replace(' ', '_')
        s_save_file = os.path.join(save_root_directory, '{:s}.npy'.format(stem_name))
        s_dataset = SingleVideoFrameDataset.VideoChunkDenseDataset(s_image_direcotry, chunk_size=chunk_size, transform=val_transform)
        s_dataset_loader = torch.utils.data.DataLoader(s_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.workers, pin_memory=True)
        s_scores = []
        pbar = progressbar.ProgressBar(max_value=len(s_dataset))
        for i, s_image in enumerate(s_dataset_loader):
            pbar.update(i)
            s_image = s_image.permute(0,2,1,3,4)

            if useCuda:
                s_image = s_image.cuda()

            input_image = Variable(s_image)
            _, preds = model(input_image)
            s_score = preds.data.cpu().numpy().squeeze(0)
            for cnt in range(chunk_size):
                s_scores.append(s_score)

        # Padding
        s_scores = np.asarray(s_scores)
        # if s_scores.shape[0] < len(s_dataset.image_path_list):
        #     padding = np.asarray([s_scores[-1, :]] * (- s_scores.shape[0] + len(s_dataset.image_path_list)))
        #     s_scores = np.vstack((s_scores, padding))

        np.save(s_save_file, s_scores)


if __name__ == '__main__':
    main()