from __future__ import print_function
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import os.path as osp
import argparse
import h5py
import math
import numpy as np
import PyUtils.dir_utils as dir_utils

from utils import write_json

parser = argparse.ArgumentParser("Code to create splits in json form")
parser.add_argument('-d', '--dataset', type=str, default='/home/zwei/datasets/KY_AAAI18_v2/datasets/eccv16_dataset_tvsum_google_pool5.h5', help="path to h5 dataset (required)")
parser.add_argument('--save-dir', type=str, default='datasets', help="path to save output json file (default: 'datasets/')")
parser.add_argument('--save-name', type=str, default=None, help="name to save as, excluding extension (default: 'splits')")
parser.add_argument('--num-splits', type=int, default=1, help="how many splits to generate (default: 5)")
parser.add_argument('--train-percent', type=float, default=0.8, help="percentage of training data (default: 0.8)")

args = parser.parse_args()

# def split_random(keys, num_videos, num_train):
#     """Random split"""
#     train_keys, test_keys = [], []
#     rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
#     for key_idx, key in enumerate(keys):
#         if key_idx in rnd_idxs:
#             train_keys.append(key)
#         else:
#             test_keys.append(key)
#
#     assert len(set(train_keys) & set(test_keys)) == 0, "Error: train_keys and test_keys overlap"
#
#     return train_keys, test_keys

def check():
    print("==========\nArgs:{}\n==========".format(args))
    print("Goal: randomly split data for {} times, {:.1%} for training and the rest for testing".format(args.num_splits, args.train_percent))
    print("Loading dataset from {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    keys = dataset.keys()
    num_videos = len(keys)

    #Update: confirm the length is 2seconds
    for s_video_id in range(num_videos):
        s_key = keys[s_video_id]
        s_video_data = dataset[s_key]
        # n_frames = s_video_data['n_frames'].value
        # print("{:s}\t{:d}".format(s_key, n_frames))
        user_summary = s_video_data['user_summary'].value
        for s_user_id, s_user_summary in enumerate(user_summary):
            for s_idx in range(len(s_user_summary)-1):
                if (s_user_summary[s_idx]!= s_user_summary[s_idx+1]):
                    if (s_idx+1) % 64==0:
                        continue
                    else:
                        print("{:s}\t User: {:d} Did not segment every 64 frames @ {:d}".format(s_key, s_user_id,s_idx))

        # print("DB")

    dataset.close()

if __name__ == '__main__':
    check()