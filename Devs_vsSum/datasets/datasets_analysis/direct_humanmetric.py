import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars
from vsSummDevs.SumEvaluation import metrics
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()




t_acc_min = 0
t_acc_max = 0
t_acc_mean = 0
for video_idx, s_filename in enumerate(video_file_stem_names):
    _, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)

    s_F1_scores = []
    for user_idx in range(user_labels.shape[1]):
        selected_labels = user_labels[:, user_idx]
        user_scores_list = [user_labels[:, i] for i in list(set(range(user_labels.shape[1]))-set([user_idx]))]
        s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=selected_labels.tolist())
        s_F1_scores.append(s_F1_score)

    print "[{:02d} | {:02d}]\t{:s}: \tMin:{:.04f}\tMax:{:.04f}, Mean:{:.04f}".format(video_idx, len(video_file_stem_names), s_filename, min(s_F1_scores), max(s_F1_scores), np.mean(np.asarray(s_F1_scores)))
    t_acc_min += min(s_F1_scores)
    t_acc_max += max(s_F1_scores)
    t_acc_mean += np.mean(np.asarray(s_F1_scores))
print "Total MinAcc: {:.04f}\t MaxAcc{:.04f}\t Mean:{:.04f}".format(t_acc_min/len(video_file_stem_names), t_acc_max/len(video_file_stem_names), t_acc_mean/len(video_file_stem_names))



