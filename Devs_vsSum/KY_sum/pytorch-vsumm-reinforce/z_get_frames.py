from __future__ import print_function
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
import glob
import numpy as np


import PyUtils.dir_utils as dir_utils
target_directory = '/home/zwei/datasets/TVSum/frames'
target_feature_directory = '/home/zwei/datasets/TVSum/features/Kinetics/I3D'
video_directories = glob.glob(os.path.join(target_directory, '*/'))
video_directories.sort()
for s_video_directory in video_directories:
    s_video_name = s_video_directory.split(os.sep)[-2]
    target_feature_file = os.path.join(target_feature_directory, '{:s}.npy'.format(s_video_name))
    target_feature = np.load(target_feature_file)
    n_frames = len(glob.glob(os.path.join(s_video_directory,'*.jpg')))
    print("{:s}\t{:d}".format(s_video_name, n_frames))