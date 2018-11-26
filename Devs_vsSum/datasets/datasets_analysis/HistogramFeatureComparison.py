import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import numpy as np

import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from scipy.stats.stats import pearsonr
import vsSummDevs.datasets.getShotBoundariesECCV2016 as getSegs
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

# s_feature_path = feature_paths[0]

pdefined_segs = getSegs.getSumMeShotBoundaris()
videofile_stems = dataset_pathvars.file_names
videofile_stems.sort()

F1_scores = 0
for video_idx, s_filename in enumerate(videofile_stems):

    video_features, user_labels, _ = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=True)

    HistFeature_video = np.sum(video_features, axis=0)

    for user_idx in range(user_labels.shape[1]):
        selected_label = user_labels[:, user_idx]
        selected_video_features = video_features[selected_label!=0, :]
        HistFeature_s_user = np.sum(selected_video_features, axis=0)

        s_scorr, s_p = pearsonr(HistFeature_video, HistFeature_s_user)

        print "{:s} \t Correlation with: {:02d}\t {:.04f}, \t{:d}".format(s_filename, user_idx, s_scorr, np.sum(selected_label))

    # get a random:
    randome_selected_label = np.random.randn(video_features.shape[0])
    sorted_id = np.argsort(randome_selected_label)
    for i in range(video_features.shape[0]):
        randome_selected_label[sorted_id[i]] = 1 if i < int(0.15*video_features.shape[0]) else 0

    selected_video_features = video_features[randome_selected_label != 0, :]
    HistFeature_s_user = np.sum(selected_video_features, axis=0)
    s_scorr, s_p = pearsonr(HistFeature_video, HistFeature_s_user)
    print "{:s} \t Random Correclation\t {:.04f}\t{:d}".format(s_filename, s_scorr, int(np.sum(randome_selected_label)))


# print "DEBUG"




