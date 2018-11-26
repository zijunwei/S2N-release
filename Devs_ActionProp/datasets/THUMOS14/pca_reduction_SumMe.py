import time

import h5py
import pickle as pkl
import numpy as np
import glob
import os
import scipy.io
import progressbar
from PyUtils import load_utils,dir_utils


def main():
    dataset_name = 'TVSum'
    PCA_info = pkl.load(open('pca_c3dd_fc7_{:s}.pkl'.format(dataset_name), 'rb'))
    x_mean = PCA_info['x_mean']
    U = PCA_info['U']
    num_red_dim = 500


    feature_directory = '/home/zwei/datasets/{:s}/features/c3d'.format(dataset_name)
    feature_files = glob.glob(os.path.join(feature_directory, '*.mat'))
    target_directory = dir_utils.get_dir('/home/zwei/datasets/{:s}/features/c3dd-red500'.format(dataset_name))

    print time.ctime(), 'start'
    pbar = progressbar.ProgressBar(max_value=len(feature_files))
    for feat_idx, s_feature_file in enumerate(feature_files):
        s_file_stem = dir_utils.get_stem(s_feature_file)
        output_file = os.path.join(target_directory, '{:s}.npy'.format(s_file_stem))
        pbar.update(feat_idx)
        s_feature = scipy.io.loadmat(s_feature_file)['fc7']
        s_feature_red = np.dot(s_feature - x_mean, U[:, :num_red_dim])
        np.save(output_file, s_feature_red)



if __name__ == '__main__':
    main()


