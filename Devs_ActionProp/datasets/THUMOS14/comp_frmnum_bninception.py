# import pickle
# import pandas as pd
# import path_vars
import numpy as np
import os
import glob

# ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
# frm_nums = pickle.load(open("/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/frm_num.pkl"))
# ground_truth = pd.read_csv(ground_truth_file, sep=' ')
feature_file_directory = '/home/zwei/datasets/THUMOS14/features/BNInception'

feature_file_list = glob.glob(os.path.join(feature_file_directory, '*.npy'))
# video_list = ground_truth['video-name'].unique()
# PathVars = path_vars.PathVars()
frame_dict = {}
for s_feature_file in feature_file_list:

    s_video_name = os.path.basename(s_feature_file)
    # s_feature_file = os.path.join(feature_file_directory, '{:s}.npy'.format(s_video_name))
    if not os.path.exists(s_feature_file):
        continue
    s_feature = np.load(s_feature_file)
    s_feature_len = s_feature.shape[0]
    frame_dict[s_video_name] = s_feature_len
    print("{:s}\t{:d}".format(s_video_name, s_feature_len))
print("DEBUG")


