import os
import sys
import scipy.io
import path_vars
import glob
import shutil




if __name__ == '__main__':
    target_dir = '/home/zwei/datasets/SumMe/frames'
    groundtruth_paths = glob.glob(os.path.join(target_dir, '*/'))
    for s_groundtruth_path in groundtruth_paths:
        # s_basename = os.path.basename(s_groundtruth_path)
        s_basename = s_groundtruth_path.split(os.sep)[-2]
        new_name = os.path.join(target_dir, s_basename.replace(' ', '_'))
        shutil.move(s_groundtruth_path, new_name)
