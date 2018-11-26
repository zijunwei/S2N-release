# the upper bound of pointer net...
import math
import vsSummDevs.obs_loaders.SumMeDataLoader
import numpy as np
from vsSummDevs.SumEvaluation.knapsack import knapsack_dp
import vsSummDevs.JM.JKyLoader as KyLoader
from vsSummDevs import JM as vsum_tools

eval_dataset = 'TVSum' # or "SumMe
if eval_dataset == 'TVSum':
    eval_method = 'avg'
else:
    eval_method = 'max'


dataset_path = '/home/zwei/datasets/KY_AAAI18/datasets/eccv16_dataset_{:s}_google_pool5.h5'.format(eval_dataset.lower())
dataset = KyLoader.loadKyDataset(eval_dataset, dataset_path)
video_frames = KyLoader.getKyVideoFrames(dataset)
dataset_keys =KyLoader.getKyDatasetKeys(dataset)


sum_summary_score = 0
for video_idx, s_key in enumerate(dataset_keys):
    s_labels = dataset[s_key]['user_summary'][...]
    s_labels = s_labels.transpose()

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
    print("{:s}\t{:.5f}".format(s_key, summary_score))
    sum_summary_score += summary_score

print "Average F1 score: {:.4f}".format(sum_summary_score/len(dataset_keys))

