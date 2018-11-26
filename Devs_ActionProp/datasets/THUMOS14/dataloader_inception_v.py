import path_vars


import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
from torchvision import datasets, transforms
# compared to t1, refine the location, start and end, always predict to the ceil

dataset_root = os.path.join(os.path.expanduser('~'),'datasets/MNIST')
raw_folder = 'raw'
processed_folder = 'processed'
training_file = 'training.pt'
test_file = 'test.pt'
import random
random.seed(0)
import progressbar
import numpy as np
import scipy.io
np.random.seed(0)
import pickle as pkl
from PyUtils import load_utils
import pandas as pd
import math

miss_name = 'video_validation_0000190'
class THUMOST14(data.Dataset):

    def __init__(self, seq_length=360, overlap=0.9, feature_file_ext='npy'):
        self.PathVars = path_vars.PathVars()
        self.feature_directory = '/home/zwei/datasets/THUMOS14/features/BNInception'
        self.feature_file_ext = feature_file_ext
        self.annotation_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_val_groundtruth.csv'
        print("Reading training data list from {:s}\t clip=len:{:d}".format(self.annotation_file, seq_length))
        self.movie_instances = {}
        ground_truth = pd.read_csv(self.annotation_file, sep=' ')
        n_ground_truth = len(ground_truth)
        for i_pos in range(n_ground_truth):
            s_ground_truth = ground_truth.loc[[i_pos]]
            movie_name = s_ground_truth['video-name'].values[0]
            n_frames = self.PathVars.video_frames[movie_name]
            if movie_name == miss_name:
                # print("DEB")
                continue
            else:
                gt_start = s_ground_truth['f-init'].values[0]
                gt_end = min(s_ground_truth['f-end'].values[0], n_frames)
                if movie_name in self.movie_instances.keys():
                    self.movie_instances[movie_name].append((gt_start, gt_end))
                else:
                    self.movie_instances[movie_name] = [(gt_start, gt_end)]
        n_positive_instances = 0
        total_reps = 0
        #TODO: during training, we can remove the repeats, in the test, the repeats come out as you have to evaluate on different overlapped clipps
        for s_name in self.movie_instances.keys():

            s_action_list = self.movie_instances[s_name]
            orig_len = len(s_action_list)
            s_action_list = list(set(s_action_list))
            s_action_list.sort()
            cur_len = len(s_action_list)
            # print("{:s}\t reps{:d}".format(s_name, orig_len-cur_len))
            total_reps += orig_len - cur_len
            n_positive_instances += len(s_action_list)
            self.movie_instances[s_name] = s_action_list
        print("{:d} reps found".format(total_reps))
        self.instances = []
        self.maximum_outputs = 0
        self.seq_len = seq_length
        for s_movie_name in self.movie_instances.keys():
            s_movie_instance = self.movie_instances[s_movie_name]
            # s_movie_instance = list(set(s_movie_instance))
            n_frames = self.PathVars.video_frames[s_movie_name]
            start_idx = 0
            # end_idx = start_idx+(self.seq_len)*self.unit_size
            # TODO: detail here, not covered the the ones that are expanding out
            get_outbound = False
            while start_idx < n_frames:
                end_idx = start_idx+ self.seq_len
                if end_idx >= n_frames:
                    #TODO: should we add 1 offset?
                    start_idx = start_idx - (end_idx-n_frames)
                    end_idx = n_frames
                    get_outbound = True

                s_instance = {}
                s_instance['name'] = s_movie_name
                s_instance['start'] = start_idx
                s_instance['end'] = end_idx
                s_instance['actions'] = []

                for s_action in s_movie_instance:
                    if s_action[0] >= start_idx and s_action[1] < end_idx:
                        s_instance['actions'].append(s_action)

                if len(s_instance['actions']) > self.maximum_outputs:
                    self.maximum_outputs = len(s_instance['actions'])
                self.instances.append(s_instance)
                if get_outbound:
                    break
                start_idx = int(start_idx + (1-overlap)*self.seq_len)

        print("{:d} video clips, {:d} training instances, {:d} positive examples, max instance per segment:{:d}".
              format(len(self.movie_instances), len(self.instances), n_positive_instances, self.maximum_outputs))

    def __getitem__(self, index):
        s_instance = self.instances[index]
        s_full_feature = np.load(os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_instance['name'], self.feature_file_ext)))
        clip_start_position = s_instance['start']
        clip_end_position = s_instance['end']
        s_clip_feature = s_full_feature[int(clip_start_position):int(clip_end_position)]
        s_start_idxes = np.zeros(self.maximum_outputs)
        s_end_idxes = np.zeros(self.maximum_outputs)
        n_effective_idxes = len(s_instance['actions'])


        for idx in range(n_effective_idxes):
            p_start_position = ((s_instance['actions'][idx][0] - clip_start_position))
            p_end_position = ((s_instance['actions'][idx][1] - clip_start_position))
            s_start_idxes[idx] = p_start_position
            s_end_idxes[idx] = p_end_position
            assert s_start_idxes[idx] >= 0 and s_start_idxes[idx]<self.seq_len, '{:d} start wrong!'
            assert s_end_idxes[idx]>=0 and s_end_idxes[idx] >= s_start_idxes[idx] and s_end_idxes[idx]<self.seq_len,\
                '{:d} end wrong'

        s_clip_feature = torch.FloatTensor(s_clip_feature).transpose(0, 1)
        if s_clip_feature.shape[1] != 360:
            print("DBU")
        s_start_idxes = torch.FloatTensor(s_start_idxes)
        s_end_idxes = torch.FloatTensor(s_end_idxes)
        n_effective_idxes = torch.LongTensor([n_effective_idxes])

        return s_clip_feature, s_start_idxes, s_end_idxes, n_effective_idxes


    def __len__(self):

        return len(self.instances)


if __name__ == '__main__':
    from torch.utils.data import DataLoader


    thumos_dataset = THUMOST14(seq_length=360, overlap=0.9)
    train_dataloader = DataLoader(thumos_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4)

    pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
    offline_dataset = {}
    for idx, data in enumerate(train_dataloader):
        pbar.update(idx)
        clip_feature = data[0]
        start_idxes = data[1]
        end_idxes = data[2]
        effective_idxes = [3]

