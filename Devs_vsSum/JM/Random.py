import os
import sys

import numpy as np
from sklearn.decomposition import PCA
import sklearn
import JKyLoader as KyLoader
# import KY_vsum_tools as vsum_tools
import sum_tools as vsum_tools

eval_dataset = 'SumMe' # or "SumMe
if eval_dataset == 'TVSum':
    eval_method = 'avg'
else:
    eval_method = 'max'


dataset_path = '/home/zwei/datasets/KY_AAAI18/datasets/eccv16_dataset_{:s}_google_pool5.h5'.format(eval_dataset.lower())
dataset = KyLoader.loadKyDataset(eval_dataset, dataset_path)
video_frames = KyLoader.getKyVideoFrames(dataset)
dataset_keys =KyLoader.getKyDatasetKeys(dataset)

F1_scores = 0
frame_rate = 15

L2NormFeature=False
for video_idx, s_key in enumerate(dataset_keys):

    user_summary = dataset[s_key]['user_summary'][...]
    nfps = dataset[s_key]['n_frame_per_seg'][...].tolist()
    cps = dataset[s_key]['change_points'][...]
    n_frames = dataset[s_key]['n_frames'][()]
    # raw_features = dataset[s_key]['features'][...]
    positions = dataset[s_key]['picks'][...]


    # if L2NormFeature:
    #     raw_features = sklearn.preprocessing.normalize(raw_features)

    # pca = PCA(whiten=True, svd_solver='auto')

    # pca.fit(raw_features.transpose())
    # matrix = pca.components_
    # frame_contrib = np.sum(pca.components_, axis=0)
    # probs = (frame_contrib- np.min(frame_contrib))/(np.max(frame_contrib)-np.mean(frame_contrib))
    probs = np.random.uniform(low=0, high=1, size=[n_frames])
    probs = probs[positions]
    machine_summary = vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
    s_F1_score, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_method)

    print "[{:d} | {:d}] \t{:.04f}".format(video_idx, len(dataset_keys), s_F1_score)
    F1_scores += s_F1_score

print "overall F1 score: {:.04f}".format(F1_scores/len(dataset_keys))




# print "DEBUG"




