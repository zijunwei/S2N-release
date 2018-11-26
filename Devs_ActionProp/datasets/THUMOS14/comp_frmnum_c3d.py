import pickle
import pandas as pd
import path_vars
import numpy as np
import os
import scipy.io

ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/action_det_prep/thumos14_tag_val_proposal_list.csv'
ground_truth = pd.read_csv(ground_truth_file, sep=' ')
feature_file_directory = '/home/zwei/datasets/THUMOS14/features/c3d'

video_list = ground_truth['video-name'].unique()
# PathVars = path_vars.PathVars()

for s_video_name in video_list:
    s_feature_file = os.path.join(feature_file_directory, '{:s}.mat'.format(s_video_name))
    if not os.path.exists(s_feature_file):
        continue
    s_feature = scipy.io.loadmat(s_feature_file)['relu6']

    s_feature_len = s_feature.shape[0]
    gt_idx = ground_truth['video-name'] == s_video_name
    this_video_ground_truth = ground_truth[gt_idx]['video-frames'].unique()[0]
    # if abs(s_record_frames - s_pathvars_frames)>2:

    print("{:s}, feature-N:{:d}\tpickle-N:{:d}".format(s_video_name, s_feature_len,
                                                                                       this_video_ground_truth))

print("DEBUG")


