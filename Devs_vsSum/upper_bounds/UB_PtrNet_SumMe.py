# the upper bound of pointer net...
import math
import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader
import vsSummDevs.obs_loaders.SumMeDataLoader
import numpy as np
from vsSummDevs.SumEvaluation.knapsack import knapsack_dp
import vsSummDevs.SumEvaluation.vsum_tools as vsum_tools
videofile_stems = dataset_pathvars.file_names
videofile_stems.sort()

feature_set = {'ImageNet': [0, 1], 'Kinetics':[1, 2], 'Places':[2, 3], 'Moments': [3, 4], 'All':[0, 4]}

sum_summary_score = 0
for s_video_stem in videofile_stems:
    _, s_labels, _ = SumMeMultiViewFeatureLoader.load_by_name(s_video_stem, doSoftmax=False,
                                                                             featureset=feature_set['ImageNet'])
    sum_scores = np.sum(s_labels, axis=1)
    s_segments = vsSummDevs.obs_loaders.SumMeDataLoader.convertlabels2NonoverlappedSegs(s_labels)
    s_values = []

    updated_segments = []
    updated_values = []
    for s_segment in s_segments:
        s_len = s_segment[1] - s_segment[0]
        if s_len > 0:
            s_value = np.sum(sum_scores[s_segment[0]: s_segment[1]])*1./s_len
            if s_value > 0:
                updated_values.append(s_value)
                updated_segments.append(s_segment)
    # cut:

    # for (s_segment, s_value) in zip(s_segments, s_values):
    #     if s_value > 0 and s_segment[1]-s_segment[0]>0:
    #         updated_segments.append(s_segment)
    #         updated_values.append(s_value)


    # updated_segments, updated_values = ArrangeSegs.deoverlap_segs(s_segments, s_values, smooth_weight=0.5)
    NFPS = []
    for s_segment in updated_segments:
        NFPS.append(s_segment[1]-s_segment[0])

    n_frames = len(sum_scores)
    # positions = range(n_frames)
    updated_segments = np.asarray(updated_segments).tolist()
    updated_values = np.asarray(updated_values).tolist()
    limits = int(math.floor(n_frames * 0.15))

    picks = knapsack_dp(updated_values, NFPS, len(NFPS), limits)
    machine_summary = np.zeros(n_frames)
    for s_pick in picks:
        machine_summary[updated_segments[s_pick][0]:updated_segments[s_pick][1]] = 1


    # positions = np.asarray(positions)
    # summary = vsum_tools.generate_summary(updated_values, updated_segments, n_frames, NFPS, positions)
    summary_score, _, _ = vsum_tools.evaluate_summary(machine_summary, s_labels.transpose(), eval_metric='max')
    print("{:s}\t{:.5f}".format(s_video_stem, summary_score))
    sum_summary_score += summary_score

print "Average F1 score: {:.4f}".format(sum_summary_score/len(videofile_stems))

