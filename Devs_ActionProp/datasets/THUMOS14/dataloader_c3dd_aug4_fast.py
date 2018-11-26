# import path_vars


import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
from torchvision import datasets, transforms
#Update: trained based on generated ground truth from https://github.com/yjxiong/temporal-segment-networks

import random
#Update: derived from v5 but with different augmentations
#Update: a fast version by loading all into RAM
import progressbar
import numpy as np
import scipy.io
import pickle as pkl
from PyUtils import load_utils
import pandas as pd
import math
import sys
np.random.seed(0)
random.seed(0)
# from Losses.h_assign import compute_single_iou
train_selected_perms = np.random.permutation(200)


def compute_intersection(action, window):

    y1 = np.maximum(action[0], window[0])
    y2 = np.minimum(action[1], window[1])

    intersection = np.maximum(y2 - y1+1, 0)
    intersection_rate = intersection*1./(action[1]-action[0]+1)
    return intersection_rate


    # union = box_area + boxes_area[:] - intersection[:]
    # if intersection>0:
    #     return True
    # else:
    #     return False


miss_name = ('video_test_0001292','video_validation_0000190')
user_root = os.path.expanduser('~')
#Update: compared to aug, we add multiple "true" positives that are overlapping with more than 0.75, and return their overlap rate...
#Update: compared to agu2, we added even more true positives
#Update: compared to aug3, we will sort them by outputs
class cDataset(data.Dataset):

    def __init__(self, seq_length=90, overlap=0.9, sample_rate=None, dataset_split='val', feature_file_ext='npy', rdOffset=False, rdDrop=False):
        self.feature_directory = os.path.join(user_root, 'datasets/THUMOS14/features/c3dd-fc7-red500')
        self.feature_file_ext = feature_file_ext
        self.data_split = dataset_split
        if dataset_split == 'train' or dataset_split=='val':
            self.annotation_file = os.path.join(user_root, 'Dev/NetModules/Devs_ActionProp/action_det_prep/thumos14_tag_val_proposal_list_c3dd.csv')
        elif dataset_split == 'test':
            self.annotation_file = os.path.join(user_root, 'Dev/NetModules/Devs_ActionProp/action_det_prep/thumos14_tag_test_proposal_list_c3dd.csv')
        else:
            print("Cannot recognize split: {:s}".format(dataset_split))
            sys.exit(-1)

        if sample_rate is None:
            self.sample_rate = [1, 2, 4]
        else:
            self.sample_rate = sample_rate
        self.seq_len = seq_length
        self.overlap = overlap
        self.rdOffset = rdOffset
        self.rdDrop = rdDrop

        print("{:s}, Reading training data list from {:s}\t clip len:{:d} sample_rate: ".format(self.data_split, self.annotation_file, self.seq_len) + ' '.join(str(self.sample_rate)))

        self.movie_instances = {}

        ground_truth = pd.read_csv(self.annotation_file, sep=' ')
        n_ground_truths = len(ground_truth)

        target_video_frms = ground_truth[['video-name', 'video-frames']].drop_duplicates().values
        self.frm_nums = {}
        self.video_list = []
        for s_target_videofrms in target_video_frms:
            self.frm_nums[s_target_videofrms[0]] = s_target_videofrms[1]
            self.video_list.append(s_target_videofrms[0])

        n_files = len(self.video_list)

        selected_perms = range(n_files)
        if dataset_split == 'train':
            selected_perms = train_selected_perms[:180]
        if dataset_split == 'val':
            selected_perms = train_selected_perms[180:]

        self.selected_video_list = [self.video_list[i] for i in selected_perms]

        self.full_features = {}
        for s_video_name in self.selected_video_list:
            s_full_feature = np.load(
                os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_video_name, self.feature_file_ext)))
            self.full_features[s_video_name] = s_full_feature

        for i_pos in range(n_ground_truths):
            s_ground_truth = ground_truth.loc[[i_pos]]
            movie_name = s_ground_truth['video-name'].values[0]
            if movie_name not in self.selected_video_list:
                continue
            n_frames = self.frm_nums[movie_name]
            if movie_name in miss_name:
                # print("Missing {:s}".format(movie_name))
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
        #Update: during training, we can remove the repeats, in the test, the repeats come out as you have to evaluate on different overlapped clipps
        for s_name in self.movie_instances.keys():

            s_action_list = self.movie_instances[s_name]
            orig_len = len(s_action_list)
            s_action_list = list(set(s_action_list))
            s_action_list.sort() # sort from left to right
            cur_len = len(s_action_list)
            # print("{:s}\t reps{:d}".format(s_name, orig_len-cur_len))
            total_reps += orig_len - cur_len
            n_positive_instances += len(s_action_list)
            # self.movie_instances[s_name] = s_action_list  #Update: no need!
        print("{:d} reps found".format(total_reps))


        self.instances = []
        self.maximum_outputs = 0
        positive_in_trainings = 0
        pbar = progressbar.ProgressBar(max_value=len(self.movie_instances))
        print("Creating training instances")
        for instance_idx, s_movie_name in enumerate(self.movie_instances.keys()):
            pbar.update(instance_idx)
            s_movie_instance = self.movie_instances[s_movie_name]
            n_frames = self.frm_nums[s_movie_name]
            #Update: extract examples as different sample rates.
            for s_sample_rate in self.sample_rate:
                s_seq_len = self.seq_len*s_sample_rate
                start_idx = 0
                isInbound = True
                while start_idx < n_frames and isInbound:
                    end_idx = start_idx+ s_seq_len
                    #UPDATE: cannot set to >, since we want to set isInbound to False this time
                    if end_idx >= n_frames:
                        isInbound = False
                        start_idx = start_idx - (end_idx-n_frames)
                        end_idx = n_frames

                    s_instance = {}
                    s_instance['name'] = s_movie_name
                    s_instance['start'] = start_idx
                    s_instance['end'] = end_idx
                    s_instance['actions'] = []
                    s_instance['overlaps'] = []
                    s_instance['sample_rate'] = s_sample_rate
                    s_instance['n_frames'] = n_frames
                    #TODO: also think about here, perhaps keep the ones that are overlap with the current clip over a threshod?
                    #TODO: in this way, how are we assigning them scores?
                    s_instance_window = [start_idx, end_idx]
                    for s_action in s_movie_instance:
                        #Update: here include the partially overlaps...
                        inWindow = compute_intersection(s_action, s_instance_window)
                        if inWindow > 0.75:
                            s_action_start = max(s_action[0], s_instance_window[0])
                            s_action_end = min(s_action[1], s_instance_window[1]-1) #TODO:check if here should minus 1
                            #TODO: add overlap rate here!
                            s_instance['actions'].append([s_action_start, s_action_end])
                            s_instance['overlaps'].append(inWindow)
                    positive_in_trainings += len(s_instance['actions'])
                    if len(s_instance['actions']) > self.maximum_outputs:
                        self.maximum_outputs = len(s_instance['actions'])
                    self.instances.append(s_instance)

                    start_idx = int(start_idx + (1-self.overlap)*s_seq_len)

        total_sample_in_trainings= len(self.instances)*self.maximum_outputs
        print("{:s}\t{:d} video clips, {:d} training instances, {:d} positive examples non-repeat, max instance per segment:{:d}, {:d} positive training samples(include repeated), {:d} total training examples ".
              format(dataset_split, len(self.movie_instances), len(self.instances), n_positive_instances, self.maximum_outputs, positive_in_trainings, total_sample_in_trainings))

        #TODO: augument more positives by sampling overlap more than 0.75s...
        augmented_positive_samples = 0
        if dataset_split == 'train':
            print("AUGmenting data")
            pbar = progressbar.ProgressBar(max_value=len(self.instances))
            for instance_idx, s_instance in enumerate(self.instances):
                #augument!
                pbar.update(instance_idx)
                if len(s_instance['actions'])<self.maximum_outputs and len(s_instance['actions'])>0:
                    actions = s_instance['actions']
                    gtoverlaps = s_instance['overlaps']
                    # augmented_actions = []
                    # augmented_overlaps = []
                    candidate_augmented_actions = []
                    candidate_augmented_overlaps = []

                    for s_action, s_overlap in zip(actions, gtoverlaps):
                        # augmented_actions.append(s_action)
                        # augmented_overlaps.append(s_overlap)
                        if s_overlap<1:
                            continue
                        else:
                            for counter in range(self.maximum_outputs):# generate 5 each
                                new_candiate_action, new_candidate_overlap = self.augment_action(s_action, [s_instance['start'], s_instance['end']])
                                candidate_augmented_actions.append(new_candiate_action)
                                candidate_augmented_overlaps.append(new_candidate_overlap)

                    selected_n = min(int(self.maximum_outputs) - len(s_instance['actions']), len(candidate_augmented_overlaps))
                    if selected_n>0:
                        sort_idx = np.argsort(np.asarray(candidate_augmented_overlaps))[::-1]
                        sort_idx = sort_idx[:selected_n]
                        for s_idx in sort_idx:
                            if candidate_augmented_overlaps[s_idx]>0.75:
                                actions.append(candidate_augmented_actions[s_idx])
                                gtoverlaps.append(candidate_augmented_overlaps[s_idx])
                augmented_positive_samples += len(s_instance['overlaps'])
                        # s_instance['actions'] = augmented_actions
                        # s_instance['overlaps'] = augmented_overlaps
            print("{:s}\tAfter Augmentation {:d} positive training samples(include repeated),".
              format(dataset_split, augmented_positive_samples))
        #TODO: now think about verifying them...

    def augment_action(self, action, boundary):
        action_length = action[1] - action[0]
        offset1 = np.random.random()*0.3 - 0.15
        offset2 = np.random.random()*0.3 - 0.15
        new_start = max(int(action[0] + offset1*action_length),boundary[0])
        new_end = min(int(action[1]+offset2*action_length), boundary[1])
        overlap=self.compute_single_iou([new_start, new_end], action)
        return [new_start, new_end], overlap

    def compute_single_iou(self, box1, box2):
        """Calculates IoU of the given box with the array of the given boxes.
        simplext iou computation with 2 segments
        """
        # Calculate intersection areas
        y1 = max(box1[0], box2[0])
        y2 = min(box1[1], box2[1])
        box1_area = max(box1[1] - box1[0] + 1, 0)
        box2_area = max(box2[1] - box2[0] + 1, 0)
        intersection = max(y2 - y1 + 1, 0)
        union = box1_area + box2_area - intersection
        iou = intersection / (union + 1e-8)
        return iou

    def random_drop_feats(self, input_feature):
        flip_coin = random.randint(0, 1)
        if flip_coin:
            return input_feature
        else:
            feature_len = input_feature.shape[0]
            rperm = np.random.permutation(self.seq_len-1)[:int(feature_len*0.075)] + 1 # remove the awkward situation of sampling at 0
            rpreplace = rperm-1
            input_feature[rperm,:]=input_feature[rpreplace,:]
            return input_feature


    def __getitem__(self, index):
        # ONLY in this part the sample rate jumps in
        s_instance = self.instances[index]

        # s_full_feature = np.load(os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_instance['name'], self.feature_file_ext)))
        s_full_feature = self.full_features[s_instance['name']].copy()

        clip_start_position = s_instance['start']
        clip_end_position = s_instance['end']
        n_frames = min(s_instance['n_frames'], s_full_feature.shape[0])
        # if self.rdDrop:
        #     s_full_feature = self.random_drop_feats(s_full_feature)

        s_sample_rate = s_instance['sample_rate']

        if self.rdOffset:
            offset = random.randint(-int(self.seq_len*s_sample_rate*(1-self.overlap)*0.5), int(self.seq_len*s_sample_rate*(1-self.overlap)*0.5))
            if clip_end_position+offset>0 and clip_end_position+offset<n_frames and clip_start_position+offset>=0 and clip_start_position+offset<n_frames:
                clip_start_position += offset
                clip_end_position += offset

        s_clip_feature = s_full_feature[int(clip_start_position):int(clip_end_position):s_sample_rate]
        if self.rdDrop:
            s_clip_feature = self.random_drop_feats(s_clip_feature)

        assert s_clip_feature.shape[0] == self.seq_len, 'feature size wrong!'
        s_start_idxes = np.zeros(self.maximum_outputs, dtype=int)
        s_end_idxes = np.zeros(self.maximum_outputs, dtype=int)
        s_action_overlaps = np.zeros(self.maximum_outputs, dtype=float)
        n_effective_idxes = len(s_instance['actions'])
        actionness_scores = -np.ones(self.seq_len)

        counter = 0
        for idx in range(n_effective_idxes):

            p_start_position = ((s_instance['actions'][idx][0] - clip_start_position))
            p_end_position = ((s_instance['actions'][idx][1] - clip_start_position))
            s_overlap = s_instance['overlaps'][idx]

            if p_start_position <0 or p_end_position >= self.seq_len*s_sample_rate:
                n_effective_idxes -= 1
                continue

            s_start_idxes[counter] = int(math.floor(p_start_position*1./s_sample_rate))
            s_end_idxes[counter] = int(math.floor(p_end_position*1./s_sample_rate))
            s_action_overlaps[counter] = s_overlap

            if s_start_idxes[counter] == s_end_idxes[counter]:
                actionness_scores[s_start_idxes[counter]] = 1
            else:
                actionness_scores[s_start_idxes[counter]:s_end_idxes[counter]]=1
            assert s_start_idxes[counter] >= 0 and s_start_idxes[counter]<self.seq_len, '{:d} start wrong!'.format(s_start_idxes[counter])
            assert s_end_idxes[counter]>=0 and s_end_idxes[counter] >= s_start_idxes[counter] and s_end_idxes[counter]<self.seq_len,\
                '{:d} end wrong'.format(s_end_idxes[counter])
            counter += 1

        assert n_effective_idxes == counter
        s_clip_feature = torch.FloatTensor(s_clip_feature).transpose(0, 1)
        s_start_idxes = torch.FloatTensor(s_start_idxes)
        s_end_idxes = torch.FloatTensor(s_end_idxes)
        n_effective_idxes = torch.LongTensor([n_effective_idxes])
        s_action_overlaps = torch.FloatTensor(s_action_overlaps)
        actionness_scores = torch.FloatTensor(actionness_scores)
        return s_clip_feature, s_start_idxes, s_end_idxes, n_effective_idxes, s_action_overlaps, actionness_scores


    def __len__(self):

        return len(self.instances)


if __name__ == '__main__':
    # compute the stats of it
    from torch.utils.data import DataLoader


    thumos_dataset = cDataset(seq_length=90, overlap=0.9, sample_rate=[4], dataset_split='train', rdOffset=True, rdDrop=True)
    train_dataloader = DataLoader(thumos_dataset,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=4)

    pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
    offline_dataset = {}
    effective_example_length = []
    for idx, data in enumerate(train_dataloader):
        pbar.update(idx)
        start_idxes = data[1]
        end_idxes = data[2]
        effective_indices = data[3]
        effective_indices = effective_indices.numpy()
        for instance_idx, s_effective_indice in enumerate(effective_indices):
            s_indice = s_effective_indice[0]
            if s_indice>0:
                for s_id in range(s_indice):
                    s_start = start_idxes[instance_idx,s_id].item()*4
                    s_end = end_idxes[instance_idx, s_id].item()*4
                    effective_example_length.append(s_end - s_start)


    print("Done")

