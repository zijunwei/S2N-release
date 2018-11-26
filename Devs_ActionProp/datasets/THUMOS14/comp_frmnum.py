import pickle
import pandas as pd
import path_vars
import numpy as np
import os

ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
frm_nums = pickle.load(open("/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/frm_num.pkl"))
ground_truth = pd.read_csv(ground_truth_file, sep=' ')
feature_file_directory = '/home/zwei/datasets/THUMOS14/features/BNInception'

video_list = ground_truth['video-name'].unique()
PathVars = path_vars.PathVars()

for s_video_name in video_list:
    s_feature_file = os.path.join(feature_file_directory, '{:s}.npy'.format(s_video_name))
    if not os.path.exists(s_feature_file):
        continue
    s_feature = np.load(s_feature_file)
    s_feature_len = s_feature.shape[0]
    s_record_frames = frm_nums[s_video_name]
    gt_idx = ground_truth['video-name'] == s_video_name
    this_video_ground_truth = ground_truth[gt_idx]['video-frames'].unique()[0]
    s_pathvars_frames = PathVars.video_frames[s_video_name]
    if abs(s_record_frames - s_pathvars_frames)>2:
        print("{:s}, feature-N:{:d}\tpickle-N:{:d}\tpath-val-N:{:d}\tpd-N:{:d}".format(s_video_name, s_feature_len, s_record_frames,
                                                                                       this_video_ground_truth, s_pathvars_frames))

print("DEBUG")


