import os
import sys

import datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import datasets.getShotBoundariesECCV2016 as getSegs
from vsSummDevs.SumEvaluation import rep_conversions, metrics
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()




t_acc_min = 0
t_acc_max = 0

for video_idx, s_filename in enumerate(video_file_stem_names):
    _, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    s_seg = pdefined_segs[s_filename]
    s_F1_scores = []
    for user_idx in range(user_labels.shape[1]):
        selected_labels = user_labels[:, user_idx]
        s_frame01scores = rep_conversions.framescore2frame01score(selected_labels, s_seg)
        user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
        s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())
        s_F1_scores.append(s_F1_score)

    print "[{:02d} | {:02d}]\t{:s}: \tMin:{:.04f}\tMax:{:.04f}".format(video_idx, len(video_file_stem_names), s_filename, min(s_F1_scores), max(s_F1_scores))
    t_acc_min += min(s_F1_scores)
    t_acc_max += max(s_F1_scores)
print "Total MinAcc: {:.04f}\t MaxAcc{:.04f}".format(t_acc_min/len(video_file_stem_names), t_acc_max/len(video_file_stem_names))



