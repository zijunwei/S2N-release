# load features for each presegmented shots
# the scores for each segment is calculated based on the average 
# get the length of each segment (useful for DP)
# Load each person's annotation separately
# use c3d features
import os
import sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import h5py
import pickle as pkl
import Devs_vsSum.datasets
import progressbar
import torch.utils.data as data
from Devs_vsSum.datasets import KyLoader
import LoaderUtils
import numpy as np
import torch
import scipy.io as sio
import random
import math

from Devs_vsSum.SumEvaluation import rep_conversions, NMS

user_root = os.path.expanduser('~')

KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
# datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
# data_loaders = {'summe': SumMeLoader, 'tvsum': TVSumLoader}

np.random.seed(0)
random.seed(0)

# train_val_perms = {'summe': np.random.permutation(25), 'tvsum': np.random.permutation(50)}


def compute_intersection(action, window):
    y1 = np.maximum(action[0], window[0])
    y2 = np.minimum(action[1], window[1])

    intersection = np.maximum(y2 - y1 + 1, 0)
    intersection_rate = intersection * 1. / (action[1] - action[0] + 1)
    return intersection_rate


class cDataset(data.Dataset):
    def __init__(self, dataset_name='TVSum', split='train', decode_ratio=0.2,
                 feature_file_ext='npy', rdOffset=False, rdDrop=False, train_val_perms=None, data_path=None, max_input_len=130):
        if dataset_name.lower() not in ['summe', 'tvsum']:
            print('Unrecognized dataset {:s}'.format(dataset_name))
            sys.exit(-1)
        self.dataset_name = dataset_name
        self.feature_file_ext = feature_file_ext
        self.split = split
        self.decode_ratio = decode_ratio

        # self.feature_directory = os.path.join(user_root, 'datasets/%s/features/c3dd-red500' % (dataset_name))
        self.feature_directory = os.path.join(data_path, '%s/features/c3dd-red500' % (dataset_name))
        self.filenames = os.listdir(self.feature_directory)
        self.filenames = [f.split('.', 1)[0] for f in self.filenames]
        self.filenames.sort()
        n_files = len(self.filenames)
        
        self.filenames = [self.filenames[i] for i in train_val_perms]
        update_n_files = len(self.filenames)

        self.rdOffset = rdOffset
        self.rdDrop = rdDrop

        print("Processing {:s}\t{:s} data".format(self.dataset_name, self.split))
        print("num_videos:{:d}".format(len(self.filenames)))

        KY_dataset_path = os.path.join(data_path, 'KY_AAAI18/datasets')
        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name.lower())))
        conversion = KyLoader.loadConversion(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, '{:s}_name_conversion.pkl'.format(dataset_name.lower())))
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_data_path = os.path.join(project_root, 'Devs_vsSum/datasets/TVSum/TVSumRaw.pkl')
        raw_annotation_data = pkl.load(open(raw_data_path, 'rb'))

        self.segment_features = {}
        self.segment_scores = {}
        self.instances = []
        # self.maximum_outputs = 0
        self.max_input_len = max_input_len
        print("Creating %s instances"%(split))
        pbar = progressbar.ProgressBar(max_value=len(self.filenames))
        n_users = 0

        for file_dix, s_filename in enumerate(self.filenames):
            pbar.update(file_dix)
            Kykey = self.raw2Ky[s_filename]
            s_usersummaries = Kydataset[Kykey]['user_summary'][...]
            s_usersummaries = s_usersummaries.transpose()
            n_frames = s_usersummaries.shape[0]

            # load raw scores
            # the size of raw_user_summaris is [len, num_users]
            raw_user_summaris = np.array(raw_annotation_data[s_filename]).transpose()

            # change_points, each segment is [cps[i,0]: cps[i,1]+1]
            cps = Kydataset[Kykey]['change_points'][...]
            # total number of frames
            num_frames = Kydataset[Kykey]['n_frames'][()]
            assert n_frames == num_frames, 'frame length should be the same'
            # num of frames per segment
            nfps = Kydataset[Kykey]['n_frame_per_seg'][...]

            num_segments = cps.shape[0]
            # self.max_input_len = max(self.max_input_len, num_segments)

            # load features
            # TODO: if use dimension reduced feature, this need to change to read numpy files
            s_features = np.load(
                os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_filename, self.feature_file_ext)))
            # the size of s_features is: [length, fea_dim]
            s_features_len = len(s_features)
            # the length of c3d feature is larger than annotation, choose middles to match
            assert abs(n_frames - s_features_len) < 6, 'annotation and feature length not equal! {:d}, {:d}'.format(
                n_frames, s_features_len)
            offset = abs(s_features_len - n_frames) / 2
            s_features = s_features[offset:offset + n_frames]

            # get average features
            s_segment_features = LoaderUtils.get_avg_seg_features(s_features, cps, num_segments)
            self.segment_features[s_filename] = s_segment_features

            s_segment_scores = LoaderUtils.get_avg_scores(raw_user_summaris, cps, num_segments)
            self.segment_scores[s_filename] = s_segment_scores

            s_n_users = raw_user_summaris.shape[1]
            n_users += s_n_users

            for s_user in range(s_n_users):
                s_instance = {}
                s_instance['name'] = s_filename
                s_instance['n_frame_per_seg'] = nfps
                s_instance['user_id'] = s_user
                s_instance['n_frames'] = num_segments
                self.instances.append(s_instance)

        self.maximum_outputs = int(self.decode_ratio * self.max_input_len)
        # n_total_train_samples = len(self.instances) * self.maximum_outputs
        self.n_total_train_samples = n_users
        print("{:s}\t{:d}, max input length:{:d}, max instance per segment:{:d}, total number users:{:d}, total:{:d}".
            format(split, update_n_files, self.max_input_len, self.maximum_outputs, n_users, self.n_total_train_samples))


    def random_drop_feats(self, input_feature):
        flip_coin = random.randint(0, 1)
        if flip_coin:
            return input_feature
        else:
            feature_len = input_feature.shape[0]
            rperm = np.random.permutation(self.seq_len - 1)[
                    :int(feature_len * 0.075)] + 1  # remove the awkward situation of sampling at 0
            rpreplace = rperm - 1
            input_feature[rperm, :] = input_feature[rpreplace, :]
            return input_feature


    def __getitem__(self, index):
        # ONLY in this part the sample rate jumps in
        s_instance = self.instances[index]

        s_feature = self.segment_features[s_instance['name']].copy()
        s_score = self.segment_scores[s_instance['name']][:, s_instance['user_id']]

        # pad s_feature to max_input_len
        [fea_len, fea_dim] = s_feature.shape
        s_feature_pad = np.zeros([self.max_input_len, fea_dim])
        s_feature_pad[:fea_len, :] = s_feature

        # sort the s_score from high to low, get positions, only get top scores
        sort_idx = np.argsort(s_score)
        sort_idx = sort_idx[::-1]

        sort_score = s_score[sort_idx]

        n_effective_idxes = self.maximum_outputs

        pointer_idxes = np.zeros([self.maximum_outputs], dtype=int)
        pointer_idxes[:n_effective_idxes] = sort_idx[:n_effective_idxes]
        pointer_scores = np.zeros([self.maximum_outputs], dtype=float)
        pointer_scores[:n_effective_idxes] = sort_score[:n_effective_idxes]

        # s_feature: [fea_dim, max_input_len]
        s_feature_pad = torch.FloatTensor(s_feature_pad).transpose(0, 1)
        pointer_idxes = torch.LongTensor(pointer_idxes)
        n_effective_idxes = torch.LongTensor([n_effective_idxes])
        pointer_scores = torch.FloatTensor(pointer_scores)
        # print(s_score)
        # print('pointer idx')
        # print(pointer_idxes)
        # print('pointer score')
        # print(pointer_scores)
        # print(n_effective_idxes)
        # print(s_feature_pad.size())
        # print(pointer_idxes.size())
        # print(pointer_scores.size())
        # print(n_effective_idxes)
        return s_feature_pad, pointer_idxes, pointer_scores, n_effective_idxes

    def __len__(self):

        return len(self.instances)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    kfold_split = 1
    train_val_perms = np.arange(50)
    random.Random(0).shuffle(train_val_perms)
    train_val_perms = train_val_perms.reshape([5, -1])
    train_perms = np.delete(train_val_perms, kfold_split, 0).reshape([-1])
    val_perms = train_val_perms[kfold_split]
    
    location = 'bigbrain'
    if location == 'home':
        data_path = os.path.join(os.path.expanduser('~'), 'datasets')
    else:
        data_path = os.path.join('/nfs/%s/boyu/SDN'%(location), 'datasets')
    

    sDataset = cDataset(dataset_name='TVSum', split='val', decode_ratio=0.2, train_val_perms=val_perms, data_path=data_path)
    train_dataloader = DataLoader(sDataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)

    pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
    offline_dataset = {}
    effective_example_length = []
    for idx, data in enumerate(train_dataloader):
        pbar.update(idx)
        pointer_idxes = data[1]
        pointer_scores = data[2]
        effective_indices = data[3]
        if idx == 0: break

