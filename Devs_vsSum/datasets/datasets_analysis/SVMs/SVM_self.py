import os
import sys

import numpy as np

import datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from scipy.stats.stats import pearsonr
from sklearn import svm
import datasets.getShotBoundariesECCV2016 as getSegs
FeatureDirecotry = ['/home/zwei/datasets/SumMe/features/ImageNet/VGG']
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Kinetics/I3D')
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Places/ResNet50')
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Moments/ResNet50')
from vsSummDevs.SumEvaluation import rep_conversions, metrics
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

# s_feature_path = feature_paths[0]

pdefined_segs = getSegs.getSumMeShotBoundaris()
videofile_stems = dataset_pathvars.file_names

F1_scores = 0
for video_idx, s_filename in enumerate(videofile_stems):

    video_features, user_labels, _ = SumMeMultiViewFeatureLoader.load_by_name(s_filename)
    avg_labels = np.mean(user_labels, axis=1)

    clf = svm.LinearSVR()
    clf.fit(video_features, avg_labels)


    frame_contrib = clf.predict(video_features)
    # frame_contrib = (frame_contrib- np.min(frame_contrib))/(np.max(frame_contrib)-np.mean(frame_contrib))
    # s_seg = pdefined_segs[s_filename]
    # s_frame01scores = rep_conversions.framescore2frame01score(frame_contrib, s_seg)
    #
    s_frame01scores= rep_conversions.framescore2frame01score_sort(frame_contrib)
    user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
    s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())

    s_scorr, s_p = pearsonr(frame_contrib, avg_labels)
    print "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}, \t[D * N ]: [{:d}, {:d}]".format(video_idx, len(videofile_stems), s_filename, s_F1_score, s_scorr, video_features.shape[1], video_features.shape[0])
    F1_scores += s_F1_score

print "overall F1 score: {:.04f}".format(F1_scores/len(videofile_stems))




# print "DEBUG"




