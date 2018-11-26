import pickle
import pandas as pd
import path_vars
import numpy as np
import os
# show the distribution of the data
ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
ground_truth = pd.read_csv(ground_truth_file, sep=' ')
# feature_file_directory = '/home/zwei/datasets/THUMOS14/features/BNInception'

video_list = ground_truth['video-name'].unique()
# PathVars = path_vars.PathVars()

for s_video_name in video_list:


    # s_record_frames = frm_nums[s_video_name]
    gt_idx = ground_truth['video-name'] == s_video_name
    this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                    'f-end']].drop_duplicates().values
    # this_video_ground_truth = set(this_video_ground_truth)
    n_frames = ground_truth[gt_idx]['video-frames'].unique()[0]

    print("{:s}, \tFrames:{:d}, \tN-instances: {:d}".format(s_video_name, n_frames,len(this_video_ground_truth)))

print("DEBUG")


