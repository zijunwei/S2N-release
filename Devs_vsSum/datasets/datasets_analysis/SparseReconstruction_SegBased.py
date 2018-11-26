import os
import sys

import numpy as np

import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars
import vsSummDevs.datasets.getShotBoundariesECCV2016 as getSegs

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from vsSummDevs.SumEvaluation import rep_conversions, metrics
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader
from numpy.linalg import pinv
import math


def selectInitFrameAvgDist(video_features):
    avgFeature = np.mean(video_features, axis=0)
    featDistance = np.sum((video_features - avgFeature)**2, axis=1)
    selected_idx = np.argmin(featDistance)
    return video_features[selected_idx], selected_idx

def selectInitFrameMaxMag(video_features):
    feature_mag = np.sum(video_features**2, axis=1)
    selected_idx = np.argmax(feature_mag)
    return video_features[selected_idx], selected_idx

def computePk(Fk):
    # Fk is d by n where d is the feature dim and n is n samples
    intermediate = pinv(np.matmul(Fk.transpose(), Fk))
    Pk = np.matmul(intermediate, Fk.transpose())
    return Pk

def computePOR(fj, Fk, Pk):
    rj = np.matmul(np.matmul(Fk, Pk), fj)
    norm_rj = np.sum((rj**2).flatten())
    norm_fj = np.sum(fj**2)
    POR = math.sqrt(norm_rj/norm_fj)
    return POR


def checkIdxSegs(selectedIdices, pdefined_segs, nFrames, ratio=1.0, sample_rate=1):
    intevals = rep_conversions.convert_seg2interval(pdefined_segs, nFrames)
    occupied_tags = np.zeros(len(intevals))
    inner_selectedIdics = selectedIdices[:]
    inner_selectedIdics.sort()
    sum_len = 0
    for idx in inner_selectedIdics:
        for int_idx, s_inteval in enumerate(intevals):
            if s_inteval[0] <= idx*sample_rate and s_inteval[1] >= idx*sample_rate:
                if occupied_tags[int_idx] == 1:
                    return False
                else:
                    occupied_tags[int_idx] = 1
                    sum_len+= s_inteval[1] - s_inteval[0]
    if sum_len > ratio*nFrames:
        return False
    return True

def getDenseScores(selectedIdxes, framerate, original_length):
    current_scores = np.zeros(original_length)
    for i in selectedIdxes:
        current_scores[i*framerate:min((i+1)*framerate, original_length)] = 1
    return current_scores



def getSelectedLength(selectedIdices, segment_length):
    length = 0
    for s_idx in selectedIdices:
        length += segment_length[s_idx]

    return length

def segscore2framescore(selectedIndices, s_segments, total_frames):
    frame01scores = np.zeros(total_frames)
    for s_idx in selectedIndices:
        frame01scores[s_segments[s_idx]:s_segments[s_idx+1]] = 1
    return frame01scores

sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
totalF1 = 0
for video_idx, s_filename in enumerate(video_file_stem_names):

    # s_filename = video_file_stem_names[0]
    s_segments = pdefined_segs[s_filename]
    segmental_representations = []
    segment_length = []
    video_features, user_labels, _= SumMeMultiViewFeatureLoader.load_by_name(s_filename)

    for i in range(1, len(s_segments)):
        start_idx = s_segments[i-1]
        end_idx = s_segments[i]
        segment_video_features = (video_features[start_idx:end_idx,:])
        avg_feature, _ = selectInitFrameAvgDist(segment_video_features)
        segmental_representations.append(avg_feature)
        segment_length.append(end_idx - start_idx)


    avg_labels = np.mean(user_labels, axis=1)

    selectedFeatures = []
    selectedIdices = []
    segmental_representations = np.asarray(segmental_representations)
    initFrame, initIdx = selectInitFrameAvgDist(segmental_representations)
    selectedFeatures.append(initFrame)
    selectedIdices.append(initIdx)

    nFrames = video_features.shape[0]
    feature_size = video_features.shape[1]


    while(True):
        if getSelectedLength(selectedIdices, segment_length) > 0.15*nFrames:
            break


        Fk = np.vstack(selectedFeatures)
        Fk = Fk.transpose()
        Pk = computePk(Fk)
        left_selected_indices = list(set(range(segmental_representations.shape[0])) - set(selectedIdices))
        left_POR = []


        # check if there are repeative segments
        for frame_idx in left_selected_indices:
            left_POR.append(computePOR(segmental_representations[frame_idx], Fk, Pk))

        selected_idx = np.argmin(np.asarray(left_POR))

        selectedIdices.append(selected_idx)
        selectedFeatures.append(segmental_representations[selected_idx])




    s_frame01scores =segscore2framescore(selectedIdices,s_segments,nFrames)

    # s_frame01scores = rep_conversions.selectframe2segscores(selectedIdices, s_segments, original_nFrames, sample_rate=sample_rate)
    user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
    s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())
    # print  "{:s} \t{:.04f}\t".format(s_filename, s_F1_score)
    print  "[{:d} | {:d}] \t {:s} \t{:.04f}".format(video_idx, len(video_file_stem_names), s_filename, s_F1_score)
    totalF1 += s_F1_score
    # s_scorr, s_p = pearsonr(frame_contrib, avg_labels)
    # print  "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}".format(video_idx, len(feature_paths), s_video_name_stem, s_F1_score, s_scorr)
#

print "overall F1 score: {:.04f}".format(totalF1/len(video_file_stem_names))




# print "DEBUG"




