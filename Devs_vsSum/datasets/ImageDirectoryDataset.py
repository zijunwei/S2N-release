import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import scipy.io
import torch.utils.data as data

import Devs_vsSum.datasets.SumMe.path_vars as SumMePathvars
import glob
import numpy as np
import random
import Devs_vsSum.datasets.SumMe.path_vars
import torchvision.datasets.folder as dataset_utils

class ImageDirectoryDataset(data.Dataset):

    def __init__(self, image_directory, transform=None, file_ext='jpg', default_loader=None):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader

        self.image_path_list = glob.glob(os.path.join(image_directory, '*.{:s}'.format(file_ext)))
        self.image_path_list.sort()
        if len(self.image_path_list) < 1:
            print("Cannot find feature files in {:s}".format(image_directory))
            exit(-1)
        self.transform = transform

    def __getitem__(self, index):
        s_image_path = self.image_path_list[index]
        s_image = self.default_loader(s_image_path)

        if self.transform is not None:
            s_image = self.transform(s_image)

        return s_image

    def __len__(self):
        return len(self.image_path_list)



class ImageListDataset(data.Dataset):

    def __init__(self, image_list, transform=None, file_ext='jpg', default_loader=None):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader
        self.image_path_list = image_list
        self.image_path_list.sort()
        if len(self.image_path_list) < 1:
            print("Cannot find feature files in {:s}".format(image_list))
            exit(-1)
        self.transform = transform

    def __getitem__(self, index):
        s_image_path = self.image_path_list[index]
        s_image = self.default_loader(s_image_path)

        if self.transform is not None:
            s_image = self.transform(s_image)

        return s_image

    def __len__(self):
        return len(self.image_path_list)