import numpy as np

def upsample_temporal_scores(scores, sample_rate=5, groundtruth_scores=None):
    avg_scores = np.mean(scores)
    upsampled_scores = []
    for s_score in scores:
        upsampled_scores.append([s_score] * sample_rate)
    upsampled_scores = np.concatenate(upsampled_scores, axis=0)
    upsampled_scores = upsampled_scores.tolist()
    if len(upsampled_scores) > len(groundtruth_scores):
        upsampled_scores = upsampled_scores[:len(groundtruth_scores)]

    if len(upsampled_scores) < len(groundtruth_scores):
        padding_len = len(groundtruth_scores)-len(upsampled_scores)
        for i in range(padding_len):
            upsampled_scores.append(avg_scores)

    upsampled_scores = np.asarray(upsampled_scores).squeeze()
    return upsampled_scores

