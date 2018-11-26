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



cs = [1, 1e2, 1e3, 1e4, 1e5]
for C in cs:
    t_acc_sort = 0
    t_acc_seg = 0
    percentage = 0.2
    for video_idx, s_filename in enumerate(video_file_stem_names):
        video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
        reg_labels = np.mean(user_labels, axis=1)
        s_seg = pdefined_segs[s_filename]
        ntrain = int(percentage*video_features.shape[0])

        selected_train_features = video_features[:ntrain, :]
        selected_train_labels = reg_labels[:ntrain]
        # val_features = video_features[ntrain:,:]
        # val_labels = reg_labels[ntrain:]

        # clc_labels = np.max(user_labels, axis=1)
        clf = svm.LinearSVR(C=C)
        clf.fit(selected_train_features, selected_train_labels)
        predicted_video_labels = clf.predict(video_features)
        s_frame01scores = rep_conversions.framescore2frame01score_sort(predicted_video_labels)
        user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
        s_scorr, s_p = pearsonr(predicted_video_labels, reg_labels)

        s_F1_score_sort = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())
        t_acc_sort += s_F1_score_sort

        s_frame01scores = rep_conversions.framescore2frame01score(predicted_video_labels, s_seg)
        s_F1_score_seg = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())

        t_acc_seg += s_F1_score_seg

        print "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}\t{:.04f}".format(video_idx, len(video_file_stem_names),
                                                                                      s_filename, s_F1_score_sort, s_scorr, s_F1_score_seg)
    print "Total Acc: {:.04f}\t{:.04f}".format(t_acc_sort / len(video_file_stem_names), t_acc_seg/(len(video_file_stem_names)))



