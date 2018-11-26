import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
from sklearn.decomposition import PCA
import sklearn
from vsSummDevs.datasets.TVSum import TVSumMultiViewFeatureLoader
import datasets.TVSum.path_vars as dataset_pathvars
from datasets import KyLoader
from vsSummDevs.SumEvaluation import vsum_tools

videofile_stems = dataset_pathvars.file_names
videofile_stems.sort()


eval_dataset = 'TVSum'
dataset = KyLoader.loadKyDataset(eval_dataset)
video_frames = KyLoader.getKyVideoFrames(dataset)
dataset_keys =KyLoader.getKyDatasetKeys(dataset)

F1_scores = 0
frame_rate = 15
feature_set = {'ImageNet': [0, 1], 'Kinetics':[1, 2], 'Places':[2, 3], 'Moments': [3, 4]}

doSoftMax = False
L2NormFeature = False
# feature_type = 'Kinetics'

print "Do SoftMax: " + str(doSoftMax) + "\tL2Norm: " + str(L2NormFeature)
# print "Selected Type: {:s}".format(feature_type)

for video_idx, s_filename in enumerate(videofile_stems):

    # selected_feature_type= feature_set[feature_type]

    video_features, _, feature_sizes = TVSumMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=doSoftMax)
    feature_boundary = [0]
    feature_boundary.extend(np.cumsum(np.asarray(feature_sizes)).tolist())
    # video_features = video_features[:, feature_boundary[selected_feature_type[0]]:feature_boundary[selected_feature_type[1]]]

    if L2NormFeature:
        video_features = sklearn.preprocessing.normalize(video_features)

    n_frames = video_features.shape[0]
    key = KyLoader.searchKeyByFrame(n_frames, video_frames, dataset_keys)
    user_summary = dataset[key]['user_summary'][...]
    nfps = dataset[key]['n_frame_per_seg'][...].tolist()
    cps = dataset[key]['change_points'][...]

    positions = KyLoader.createPositions(n_frames, frame_rate)
    video_features = video_features[positions]
    pca = PCA(whiten=True, svd_solver='auto')

    pca.fit(video_features.transpose())
    matrix = pca.components_
    frame_contrib = np.sum(pca.components_, axis=0)
    probs = (frame_contrib- np.min(frame_contrib))/(np.max(frame_contrib)-np.mean(frame_contrib))
    # probs = probs[positions]

    machine_summary = vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
    s_F1_score, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')

    print "[{:d} | {:d}] \t {:s} \t{:.04f}".format(video_idx, len(videofile_stems), s_filename, s_F1_score)
    F1_scores += s_F1_score

print "overall F1 score: {:.04f}".format(F1_scores/len(videofile_stems))




# print "DEBUG"




