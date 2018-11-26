import scipy.io as sio
import sys, os
from PyUtils import dir_utils
import numpy as np
import glob
import progressbar

feature_directory = '/home/zwei/datasets/THUMOS14/features/c3d_feat_dense'
target_directory = dir_utils.get_dir('/home/zwei/datasets/THUMOS14/features/c3d-dense')

feature_lists = glob.glob(os.path.join(feature_directory, '*.mat'))


# video_name = 'video_validation_0000940.mat'
pbar = progressbar.ProgressBar(max_value=len(feature_lists))
for file_id, s_feature_path in enumerate(feature_lists):
    pbar.update(file_id)
    # s_feature_name = os.path.basename(s_feature_path)
    s_feature_stem = dir_utils.get_stem(s_feature_path)
    # feature_path = os.path.join(, video_name)
    s_full_feature = sio.loadmat(s_feature_path)['relu6']
    np.save(os.path.join(target_directory, '{:s}.npy'.format(s_feature_stem)), s_full_feature)
print("DEBUG")
