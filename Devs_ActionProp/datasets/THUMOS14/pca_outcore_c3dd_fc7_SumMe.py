#!/bin/bash/env python
"""
PCA done via matrix multiplication out-of-core. It is here just to be
informative i.e. hostile and full of dependencies parsing of inputs.
"""
import time

import h5py
import pickle as pkl
import numpy as np
import glob
import os
import scipy.io
import progressbar
from PyUtils import load_utils


# THUMOS14_VAL = 'data/thumos14/c3d/val_c3d_temporal.hdf5'


def main(feat_dim=4096, log_loop=20):
    dataset_name = 'TVSum'
    feature_directory = '/home/zwei/datasets/{:s}/features/c3d'.format(dataset_name)
    feature_files = glob.glob(os.path.join(feature_directory, '*.mat'))
    x_mean, n = np.zeros((1, feat_dim), dtype=np.float32), 0
    print time.ctime(), 'start: compute mean'
    pbar = progressbar.ProgressBar(max_value=len(feature_files))
    for feat_idx, s_feature_file in enumerate(feature_files):
        pbar.update(feat_idx)
        s_feature = scipy.io.loadmat(s_feature_file)['fc7']
        n += s_feature.shape[0]
        x_mean += s_feature.sum(axis=0)
    x_mean /= n
    print time.ctime(), 'finish: compute mean'


    # Compute A.T A
    print time.ctime(), 'start: out-of-core matrix multiplication'
    j, n_videos = 0, len(feature_files)
    ATA = np.zeros((feat_dim, feat_dim), dtype=np.float32)
    for i, s_feature_file in enumerate(feature_files):
        pbar.update(i)
        s_feature = scipy.io.loadmat(s_feature_file)['fc7']

        feat_ = s_feature - x_mean
        ATA += np.dot(feat_.T, feat_)
        # j += 1
        # if j % log_loop == 0:
        #     print time.ctime(), 'Iteration {}/{}'.format(j, n_videos)
    print time.ctime(), 'finish: out-of-core matrix multiplication'

    # SVD
    print time.ctime(), 'start: SVD in memory'
    U, S, _ = np.linalg.svd(ATA)
    print time.ctime(), 'finish: SVD in memory'

    print time.ctime(), 'serializing ...'

    pkl.dump({'x_mean': x_mean, 'U': U, 'S': S}, open('pca_c3dd_fc7_{:s}.pkl'.format(dataset_name), 'wb'))

if __name__ == '__main__':
    main()