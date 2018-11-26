from __future__ import print_function
import argparse
import os
import numpy as np
import scipy.io as sio
from PIL import Image
import torchvision.transforms as TT

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.utils.data as DD
import glob
import progressbar
from PtUtils import cuda_model

from C3D import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# data
parser.add_argument('--mRGB', default=[101, 98, 90], type=float, nargs='+', dest='mRGB', help='mean RGB')
# model
parser.add_argument('--nClass', default=1, type=int, help='number of classes')
parser.add_argument('--dropRatio', default=0, type=float, help='dropout ratio')
parser.add_argument('--leak', default=0, type=float, help='leaky relu')
# evaluate
parser.add_argument('--model', default='/home/zwei/.torch/models/C3D@Sport1M/MatFile/params.mat', help='model to test')
parser.add_argument('--saveDir', default='/home/zwei/datasets/THUMOS14/features/c3dd-fc7', help='file to save results')
# misc
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')

def mkdir_p(path):
    if not os.path.isdir(path):
        os.makedirs(path)

print('==> parsing options')
global opts
opts = parser.parse_args()
print(opts)
mkdir_p(opts.saveDir)

print('==> create model')
model = C3D(nClass=opts.nClass, dropRatio=opts.dropRatio, leak=opts.leak)
model = cuda_model.convertModel2Cuda(model, gpu_id=opts.gpu_id, multiGpu=opts.multiGpu)
model.eval()

print('==> initialize model with [%s]'%(opts.model))
if opts.model[-4:]=='.mat':
    model._init_from_mat(opts.model, lastLayer=False)
elif os.path.isfile(opts.model):
    L = torch.load(opts.model)
    model.load_state_dict(L['state_dict'])
else:
    print('[%s] not found'%(opts.model))
    quit()

print('==> register forward hook functions')
Buffer = {}
def _fc7_hook(self, input, output):
    Buffer['fc7'] = output.data.clone()
model.full7.register_forward_hook(_fc7_hook)

print('==> obtain the list of video ids for THUMOS14 dataset')
frame_directories = '/home/zwei/datasets/THUMOS14/frame'
video_frame_list = glob.glob(os.path.join(frame_directories, '*/'))
video_stem_list = [s.split(os.sep)[-2] for s in video_frame_list]
# L = sio.loadmat('anno_instances.mat')
# val_videos = L['val'][0,0][0]
# tst_videos = L['tst'][0,0][0]
# val_videos = [val_videos[i,0][0] for i in range(len(val_videos))]
# tst_videos = [tst_videos[i,0][0] for i in range(len(tst_videos))]
# all_videos = sorted(list(set(val_videos + tst_videos)))

# scale, crop-center
H, W = [128, 176]
H_, W_ = [112, 112]
crp = [H/2-H_/2, H/2+H_/2-1, W/2-W_/2, W/2+W_/2-1]
CLIP = torch.zeros(3, 16, H_, W_)
toTensor = TT.ToTensor()
print('==> start extracting features')

for i in range(len(video_stem_list)):
    print('{:d} | {:d} \t {:s}'.format(i, len(video_stem_list), video_stem_list[i]))
    video = video_stem_list[i]
    matFile = os.path.join(opts.saveDir, '{:s}.mat'.format(video))
    # matFile = '%s/%s.mat'%(opts.saveDir, video)
    if not os.path.isfile(matFile):
        frameDir = os.path.join(frame_directories, video)
        maxFrm = len(glob.glob(os.path.join(frameDir, '*.jpg')))
        # nseg = int(round(maxFrm/16.0))
        fc7 = torch.zeros(maxFrm, 4096)
        pbar = progressbar.ProgressBar(max_value=maxFrm)
        for seg in range(maxFrm):
            pbar.update(seg)
            for t in range(-8,8):
                indice = seg + t + 1
                if indice<=0 or indice>=maxFrm:
                    im = torch.zeros([3, W_, H_])
                else:
                    im = Image.open(os.path.join(frameDir, 'i_{:06d}.jpg'.format(indice)))
                    im = im.resize((W, H), Image.BILINEAR)
                    im = im.crop((crp[2], crp[0], crp[3]+1, crp[1]+1)) # left-inclusive, [a, b)
                    im = toTensor(im)
                CLIP[:,t+8,:,:].copy_(im)
            CLIP = CLIP*255
            CLIP = CLIP - torch.Tensor(opts.mRGB).view(3,1,1,1)
        
            inputs = Variable(CLIP.unsqueeze(0).cuda())
            model.forward(inputs)
            fc7[seg].copy_(Buffer['fc7'][0])

        sio.savemat(matFile, {'fc7':fc7.numpy()})
