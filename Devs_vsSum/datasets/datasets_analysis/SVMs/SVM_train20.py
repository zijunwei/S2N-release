import os
import sys

import numpy as np

import datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from scipy.stats.stats import pearsonr
from sklearn import svm
import datasets.getShotBoundariesECCV2016 as getSegs
from vsSummDevs.SumEvaluation import rep_conversions, metrics
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()

train_videos = video_file_stem_names[0:20]
val_videos = video_file_stem_names[20:]


train_video_features = []
train_video_labels = []
for video_idx, s_filename in enumerate(train_videos):
    video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    reg_labels = np.mean(user_labels, axis=1)
    clc_labels = np.max(user_labels, axis=1)


    train_video_labels.append(reg_labels)
    train_video_features.append(video_features)

train_video_features = np.vstack(train_video_features)
train_video_labels = np.hstack(train_video_labels)

cs = [1, 1e2, 1e3, 1e4, 1e5]
for c in cs:
    print "--- c: {:f}----".format(c*1.0)
    clf = svm.LinearSVR(C=c)
    clf.fit(train_video_features, train_video_labels)

    F1_scores = 0
    for video_idx, s_filename in enumerate(val_videos):
        video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
        reg_labels = np.mean(user_labels, axis=1)
        clc_labels = np.max(user_labels, axis=1)
        frame_contrib = clf.predict(video_features)
        s_frame01scores= rep_conversions.framescore2frame01score_sort(frame_contrib)
        user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
        s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())

        s_scorr, s_p = pearsonr(frame_contrib, reg_labels)
        print "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}, \t[D * N ]: [{:d}, {:d}]".format(video_idx, len(val_videos), s_filename, s_F1_score, s_scorr, video_features.shape[1], video_features.shape[0])
        F1_scores += s_F1_score


    print "Framewise overall F1 score: {:.04f}".format(F1_scores/len(val_videos))


    F1_scores = 0
    for video_idx, s_filename in enumerate(val_videos):
        s_seg = pdefined_segs[s_filename]

        video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
        reg_labels = np.mean(user_labels, axis=1)
        clc_labels = np.max(user_labels, axis=1)
        frame_contrib = clf.predict(video_features)
        s_frame01scores= rep_conversions.framescore2frame01score(frame_contrib, s_seg)
        user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
        s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())

        s_scorr, s_p = pearsonr(frame_contrib, reg_labels)
        print "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}, \t[D * N ]: [{:d}, {:d}]".format(video_idx, len(val_videos), s_filename, s_F1_score, s_scorr, video_features.shape[1], video_features.shape[0])
        F1_scores += s_F1_score


    print "Segmentwise overall F1 score: {:.04f}".format(F1_scores/len(val_videos))