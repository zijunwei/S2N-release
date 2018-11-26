import os
import sys
import scipy.io
import path_vars
import glob
import shutil




if __name__ == '__main__':
    groundtruth_paths = glob.glob(os.path.join(path_vars.ground_truth_dir, '*.mat'))
    groundtruth_paths.sort()
    print "DEBUG"
