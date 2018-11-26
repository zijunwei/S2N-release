import os
import sys
import scipy.io
import path_vars
import glob



def getRawData(s_filepath):
    mat = scipy.io.loadmat(s_filepath)
    return mat


def getUserScores(s_mat, set1=True):
    userScores = s_mat['user_score']
    if set1:
        userScores[userScores>=1]=1
    return userScores


if __name__ == '__main__':
    groundtruth_paths = glob.glob(os.path.join(path_vars.ground_truth_dir, '*.mat'))
    for s_groundtruth_path in groundtruth_paths:
        annotation = getRawData(s_groundtruth_path)
        userScores = getUserScores(annotation)
        print "DEBUG"