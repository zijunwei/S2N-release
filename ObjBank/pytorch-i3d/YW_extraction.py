import os
import sys
import glob
import numpy as np
import cv2
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
except NameError:
    cwd = ''
sys.path.append(cwd + '../pytorch-i3d/') # git clone https://github.com/piergiaj/pytorch-i3d
from pytorch_i3d import InceptionI3d

parser = argparse.ArgumentParser()
parser.add_argument('--clip_root', type=str, default='/nfs/bigeye/yangwang/DataSets/Hollywood2/frameflow/')
parser.add_argument('--feat_root', type=str, default='/nfs/bigeye/yangwang/DataSets/Hollywood2/i3d/rgb/')
parser.add_argument('--model_path', type=str, default='/nfs/bigdisk/yangwang/DataSets/Kinetics/src/extI3D/pytorch-i3d/models/rgb_imagenet.pt')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
        pass

def load_rgb_frames(clipPath):
    frames = []
    nFrm = len(glob.glob(clipPath + '/i_*.jpg'))
    for i in range(1, 1+nFrm):
        frame = cv2.imread('%s/i_%06d.jpg'%(clipPath, i))[:, :, ::-1] # H, W, [RGB]
        frame = cv2.resize(frame, (340, 256))[16:16+224, 58:58+224] # resize + center
        frames.append(frame)
    return np.asarray(frames) # T, H, W, 3

def load_flow_frames(clipPath):
    frames = []
    nFrm = len(glob.glob(clipPath + '/x_*.jpg'))
    for i in range(1, 1+nFrm):
        xflow = cv2.imread('%s/x_%06d.jpg'%(clipPath, i), cv2.IMREAD_GRAYSCALE) # H, W
        xflow = cv2.resize(xflow, (340, 256))[16:16+224, 58:58+224] # resize + center
        yflow = cv2.imread('%s/y_%06d.jpg'%(clipPath, i), cv2.IMREAD_GRAYSCALE)
        yflow = cv2.resize(yflow, (340, 256))[16:16+224, 58:58+224]
        flow = np.asarray([xflow, yflow]).transpose([1,2,0])
        frames.append(flow)
    return np.asarray(frames) # T, H, W, 2

def play_frames(frames):
    for i in range(frames.shape[0]):
        cv2.imshow('frame', frames[i,:,:,0])
        cv2.waitKey(1)
    cv2.destroyAllWindows()

# parse arguments
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
mkdir_p(args.feat_root)

# load i3d-rgb model
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load(args.model_path))
i3d.cuda()
i3d.eval()

# get a list of videos
clipList = glob.glob(args.clip_root + '*')
clipList.sort()
clipList = clipList[args.start:args.end]
for clipPath in clipList:
    # clipPath = clipList[0]
    _, video = os.path.split(clipPath)
    feat_path = os.path.join(args.feat_root, video + '.npy')
    if not os.path.exists(feat_path):
        frames = load_rgb_frames(clipPath) # play_frames(frames)
        nFrm = frames.shape[0]
        step_size = 256
        features = []
        for start in range(0, nFrm, step_size):
            inputs = frames[start:start+step_size] # T, H, W, C
            if inputs.shape[0] >= 16:
                inputs = np.expand_dims(np.moveaxis(inputs, 3, 0), 0) # 1, C, T, H, W
                inputs = torch.FloatTensor(inputs)/255 * 2 - 1
                inputs = Variable(inputs.cuda(), volatile=True)
                features_i = i3d.extract_features(inputs)
                features_i = features_i.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
                features.append(features_i)
        if len(features) > 0:
            features = np.concatenate(features, axis=0)
            np.save(feat_path, features)