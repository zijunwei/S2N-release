import os
import sys

import numpy as np

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from sklearn.decomposition import PCA
from vsSummDevs.datasets.TVSum import TVSumMultiViewFeatureLoader
import vsSummDevs.datasets.TVSum.path_vars as dataset_pathvars
import h5py
import vsSummDevs.SumEvaluation.vsum_tools as vsum_tools

videofile_stems = dataset_pathvars.file_names
# videofile_stems.sort()
videofile_stems.sort()
# pdefined_segs = getSegs.getShotBoundaris()

F1_scores = 0

eval_datasets = ['summe', 'tvsum']
eval_dataset = eval_datasets[1]
KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
h5f_path = os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(eval_dataset))
dataset = h5py.File(h5f_path, 'r')
dataset_keys = dataset.keys()
n_videos = len(dataset_keys)
frame_rate = 15


video_frames = []
for i_video in range(50):
    key = dataset_keys[i_video]

    n_frames = dataset[key]['n_frames'][()]
    video_frames.append(n_frames)

def createPositions(nFrames, framerate):
    positions = range(0, nFrames, framerate)
    positions = np.asarray(positions)

    return positions

for video_idx, s_filename in enumerate(videofile_stems):

    video_features, _, _ = TVSumMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    # sklearn.preprocessing.normalize(video_features)

    n_frames = video_features.shape[0]
    target_idx = video_frames.index(n_frames)
    key = dataset_keys[target_idx]
    # assert n_frames == video_features.shape[0], "Double check"

    cps = dataset[key]['change_points'][...]
    nfps = dataset[key]['n_frame_per_seg'][...].tolist()
    positions = createPositions(n_frames, frame_rate)
    user_summary = dataset[key]['user_summary'][...]


    pca = PCA(whiten=True, svd_solver='auto')
    pca.fit(video_features.transpose())
    matrix = pca.components_
    frame_contrib = np.sum(pca.components_, axis=0)
    probs = frame_contrib[positions]

    frame_contrib = (frame_contrib- np.min(frame_contrib))/(np.max(frame_contrib)-np.mean(frame_contrib))



    machine_summary = vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
    s_F1_score, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, 'avg')

    # s_frame01scores = rep_conversions.framescore2frame01score_sort(frame_contrib)

    # s_F1_score = metrics.max_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())
    # avg_labels = np.mean(user_labels, axis=1)

    # s_scorr, s_p = pearsonr(frame_contrib, avg_labels)
    print  "[{:d} | {:d}] \t {:s} \t{:.04f}".format(video_idx, len(videofile_stems), s_filename, s_F1_score)
    F1_scores += s_F1_score

print "overall F1 score: {:.04f}".format(F1_scores/len(videofile_stems))




# print "DEBUG"




