import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import scipy.io
import torch.utils.data as data
import torch
import glob
import numpy as np
import random
from PIL import Image
import math
from torchvision import transforms
import torchvision.datasets.folder as dataset_utils
image_size = 224

class Resize(object):
    def __init__(self, w=224, h=224, interpolation=Image.BILINEAR):
        self.w = w
        self.h = h
        self.interpolation = interpolation
    def __call__(self, image):
        return image.resize((self.w, self.h), self.interpolation)


def simple_I3D_transform(image_size=None):
    operations = [] if image_size is None else [Resize(w=image_size, h=image_size)]
    operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]))
    transform = transforms.Compose(operations)
    return transform


def default_loader(path):
    return Image.open(path).convert('RGB')


class VideoChunkDataset(data.Dataset):
    def __init__(self, image_directory, chunk_size=None, sample_rate=None, transform = None, default_loader=None, file_ext='jpg'):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader

        self.image_path_list = glob.glob(os.path.join(image_directory, '*.{:s}'.format(file_ext)))
        self.image_path_list.sort()
        self.sample_rate = 1 if sample_rate is None else sample_rate
        self.chunk_size = 64 if chunk_size is None else chunk_size
        self.transform = transform

    def __getitem__(self, index):
        selected_image_filepaths = self.image_path_list[index*self.sample_rate*self.chunk_size: (index+1)*self.sample_rate*self.chunk_size:self.sample_rate]
        selected_images = []
        for s_image_path in selected_image_filepaths:
            s_image = self.default_loader(s_image_path)
            s_image = self.transform(s_image)
            selected_images.append(s_image)

        selected_images = torch.stack(selected_images)

        return selected_images

    def __len__(self):
        return math.floor(len(self.image_path_list)/self.sample_rate/self.chunk_size)


class VideoChunkDenseDataset(data.Dataset):
    def __init__(self, image_directory, chunk_size=None, sample_rate=None, transform = None, default_loader=None, file_ext='jpg'):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader

        self.image_path_list = glob.glob(os.path.join(image_directory, '*.{:s}'.format(file_ext)))
        self.image_path_list.sort()
        self.sample_rate = 1 if sample_rate is None else sample_rate
        self.chunk_size = 64 if chunk_size is None else chunk_size
        self.transform = transform
        self.half_chunk = self.chunk_size/2
        self.n_frames = len(self.image_path_list)
    def __getitem__(self, index):
        selected_images = []

        for t in range(-self.half_chunk, self.half_chunk):
            indice = index + t + 1
            if indice <= 0 or indice >= self.n_frames:
                s_image = torch.zeros([3, 224, 224])
                selected_images.append(s_image)

            else:
                s_image_path = self.image_path_list[indice]
                s_image = self.default_loader(s_image_path)
                s_image = self.transform(s_image)
                selected_images.append(s_image)
        selected_images = torch.stack(selected_images)

        return selected_images

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    #Test:
    t_transform = simple_I3D_transform(image_size)
    image_directory = '/home/zwei/datasets/SumMe/frames/Air_Force_One'
    # image_list = glob.glob(os.path.join(image_directory, '*.jpg'))
    # image_list.sort()
    videoframedataset = VideoChunkDenseDataset(image_directory, transform=t_transform, chunk_size=64, sample_rate=1)
    videoframedataloader = data.DataLoader(videoframedataset, batch_size=1, shuffle=False,num_workers=1, drop_last=False)
    for idx, s_image_chunk in enumerate(videoframedataloader):
        print "DEBUG"