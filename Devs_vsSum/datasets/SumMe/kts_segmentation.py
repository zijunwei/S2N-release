import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/VideoSum')
sys.path.append(project_root)

import glob
import numpy as np
import scipy.io
import PyUtils.dir_utils as dir_utils
import progressbar
import Processing.kts.cpd_auto as cpd_auto


def infer_segmentation_from_scores(user_score):
    segment_list = []
    for i in range(len(user_score)-1):
        if user_score[i] != user_score[i+1]:
            segment_list.append(i)
    return segment_list

feature_directory = '/home/zwei/datasets/SumMe/Annotation_InceptionV3'

feature_filelist = glob.glob(os.path.join(feature_directory, '*.{:s}'.format('mat')))
feature_filelist.sort()
target_directory = dir_utils.get_dir('/home/zwei/datasets/SumMe/Annotation_InceptionV3_SegV1')
pbar = progressbar.ProgressBar(max_value=len(feature_filelist))
for file_idx, s_feaure_file in enumerate(feature_filelist):
    pbar.update(file_idx)
    mat_content = scipy.io.loadmat(s_feaure_file)
    features = mat_content['features']

    # features = features[::5, :]
    K = np.dot(features, features.T)
    # This 1.85 magic number is from  the Summe dataset
    m = K.shape[0]/(1.85*30)
    cps, scores = cpd_auto.cpd_auto(K, int(m), 1, verbose=False)
    mat_content['segs'] = cps
    scipy.io.savemat(os.path.join(target_directory, os.path.basename(s_feaure_file)), mat_content)




