import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
from vsSummDevs.datasets import SingleVideoFrameDataset

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
from vsSummDevs.datasets.TVSum import path_vars
import glob

import numpy as np

from pytorch_i3d import InceptionI3d
from PyUtils import dir_utils
import progressbar
import ObjBank.I3d_Kinetics.I3DUtils as I3DUtils

# from charades_dataset_full import Charades as Dataset


def run(mode='rgb', batch_size=1,
        load_model='/home/zwei/Dev/NetModules/ObjBank/pytorch-i3d/models/rgb_imagenet.pt'):
    # setup dataset

    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    val_transform = I3DUtils.simple_I3D_transform(224)

    # dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #
    # val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #
    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(400)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    # for phase in ['train', 'val']:
    i3d.train(False)  # Set model to evaluate mode
    image_root_directory = os.path.join(path_vars.dataset_dir, 'frames')
    save_root_directory = dir_utils.get_dir(os.path.join(path_vars.dataset_dir, 'features/I3D/I3D-feature'))
    useCuda=True
    chunk_size=64

    image_directories = glob.glob(os.path.join(image_root_directory, '*/'))
    for idx_dir, s_image_direcotry in enumerate(image_directories):
        stem_name = s_image_direcotry.split(os.sep)[-2]
        print '[{:02d} | {:02d}] {:s}'.format(idx_dir, len(image_directories), stem_name)
        stem_name = stem_name.replace(' ', '_')
        s_save_file = os.path.join(save_root_directory, '{:s}.npy'.format(stem_name))
        s_dataset = SingleVideoFrameDataset.VideoChunkDenseDataset(s_image_direcotry, chunk_size=chunk_size, transform=val_transform)
        s_dataset_loader = torch.utils.data.DataLoader(s_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        s_scores = []
        pbar = progressbar.ProgressBar(max_value=len(s_dataset))
        for i, s_image in enumerate(s_dataset_loader):
            pbar.update(i)
            s_image = s_image.permute(0,2,1,3,4)

            if useCuda:
                s_image = s_image.cuda()

            input_image = Variable(s_image)
            print("InpuImageShape:")
            print(input_image.shape)
            features = i3d.extract_features(input_image)
            s_score = features.data.cpu().numpy().squeeze(0)
            print(s_score.shape)
            for cnt in range(chunk_size):
                s_scores.append(s_score)

        # Padding
        s_scores = np.asarray(s_scores)
        # if s_scores.shape[0] < len(s_dataset.image_path_list):
        #     padding = np.asarray([s_scores[-1, :]] * (- s_scores.shape[0] + len(s_dataset.image_path_list)))
        #     s_scores = np.vstack((s_scores, padding))

        np.save(s_save_file, s_scores)
                    
    #     # Iterate over data.
    # for data in dataloaders[phase]:
    #     # get the inputs
    #     inputs, labels, name = data
    #     if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
    #         continue
    #
    #     b,c,t,h,w = inputs.shape
    #     if t > 1600:
    #         features = []
    #         for start in range(1, t-56, 1600):
    #             end = min(t-1, start+1600+56)
    #             start = max(1, start-48)
    #             ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
    #             features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
    #         np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
    #     else:
    #         # wrap them in Variable
    #         inputs = Variable(inputs.cuda(), volatile=True)
    #         features = i3d.extract_features(inputs)
    #         np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run()
