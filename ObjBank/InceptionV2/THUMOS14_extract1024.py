import os
import sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import argparse

import PtUtils.cuda_model as cuda_model
from torch.autograd import Variable
import glob
from ActionLocalizationDevs.datasets.THUMOS14.path_vars import PathVars

from vsSummDevs.datasets import ImageDirectoryDataset
import torch.utils.data
from PyUtils import dir_utils
import numpy as np
import progressbar
import torch.nn.functional as F
import model as BNInception
import BNInceptionUtils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inception V3 Training')

parser.add_argument("--gpu_id", default='1', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')



def main():
    args = parser.parse_args()

    val_transform = BNInceptionUtils.get_val_transform()

    Dataset = PathVars()

    useCuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    model = BNInception.bninception(pretrained=True)
    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    model = model.eval()

    model_name = 'BNInception'

    image_root_directory = Dataset.flow_directory
    save_root_directory = dir_utils.get_dir(os.path.join(Dataset.feature_directory, '{:s}'.format(model_name)))

    image_directories = glob.glob(os.path.join(image_root_directory, '*/'))
    for idx_dir, s_image_direcotry in enumerate(image_directories):
        stem_name = s_image_direcotry.split(os.sep)[-2]

        print '[{:02d} | {:02d}] {:s}'.format(idx_dir, len(image_directories), stem_name)
        # if stem_name != 'video_test_0001292':
        #     continue
        stem_name = stem_name.replace(' ', '_')
        s_image_list = glob.glob(os.path.join(s_image_direcotry, 'i_*.jpg'))

        s_save_file = os.path.join(save_root_directory, '{:s}.npy'.format(stem_name))
        s_dataset = ImageDirectoryDataset.ImageListDataset(s_image_list, transform=val_transform)
        s_dataset_loader = torch.utils.data.DataLoader(s_dataset, batch_size=1, shuffle=False, drop_last=False)
        s_scores = []
        pbar = progressbar.ProgressBar(max_value=len(s_dataset))
        for i, s_image in enumerate(s_dataset_loader):
            pbar.update(i)
            if useCuda:
                s_image = s_image.cuda()
            s_image = s_image[:, [2, 1, 0], :, :]
            s_image = s_image * 256

            input_image = Variable(s_image)
            preds, feature = model(input_image)
            feature = F.avg_pool2d(feature, kernel_size=7)

            s_score = feature.data.cpu().numpy().squeeze()
            s_scores.append(s_score)

        s_scores = np.asarray(s_scores)
        np.save(s_save_file, s_scores)


if __name__ == '__main__':
    main()