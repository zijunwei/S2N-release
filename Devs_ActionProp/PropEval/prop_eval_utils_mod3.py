import  numpy as np
import progressbar
from Utils import non_maxima_supression
# added NMS or non-repeat pruning
# faster than mod2 because repeatitive computations are removed!



def common_pruning_nms(video_lst, proposals, ground_truth, value=0.99):
    score_lst = []
    score_name = []
    effective_proposals = 0
    print("Processing...")
    pbar = progressbar.ProgressBar(max_value=len(video_lst))
    for idx, videoid in enumerate(video_lst):
        pbar.update(idx)
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init',
                                                    'f-end']].values
        # scores = proposals[prop_idx]['score'].values
        pred_scores = proposals[prop_idx]['score'].values
        this_video_proposals, _ = non_maxima_supression(this_video_proposals, pred_scores, overlap=value)

        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        effective_proposals += len(this_video_proposals)
        score_lst.append(tiou)
        score_name.append(videoid)
    return score_lst, score_name, effective_proposals


def average_recall_vs_nr_proposals(video_lst, score_lst, score_name, n_effective_proposals,
                                   tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    """ Computes the average recall given an average number
        of proposals per video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    # video_lst_prop = proposals['video-name'].unique()
    # video_lst_gt = ground_truth['video-name'].unique()
    # video_lst = np.intersect1d(video_lst_prop, video_lst_gt)
    # score_lst, score_name = common_pruning_nms(video_lst, proposals, ground_truth)

    pcn_lst = np.arange(1, 101) / 100.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        #TODO add progress bar.
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(n_effective_proposals) / video_lst.shape[0])

    return recall, proposals_per_video


def average_recall_vs_freq(video_lst, score_lst, score_name, n_effective_proposals, frm_nums,
                           tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    # Get list of videos.
    # video_lst = proposals['video-name'].unique()
    # video_lst_prop = proposals['video-name'].unique()
    # video_lst_gt = ground_truth['video-name'].unique()
    # video_lst = np.intersect1d(video_lst_prop, video_lst_gt)
    # score_lst, score_name = common_pruning_nms(video_lst, proposals, ground_truth)
    # For each video, computes tiou scores among the retrieved proposals.
    # score_lst = []
    # score_name = []
    # for videoid in video_lst:
    #     # Get proposals for this video.
    #     prop_idx = proposals['video-name'] == videoid
    #     this_video_proposals = proposals[prop_idx][['f-init',
    #                                                 'f-end']].values
    #     # Sort proposals by score.
    #     # sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
    #     # this_video_proposals = this_video_proposals[sort_idx, :]
    #     scores = proposals[prop_idx]['score'].values
    #     this_video_proposals, _ = non_maxima_supression(this_video_proposals, scores, overlap=0.8)
    #     # Get ground-truth instances associated to this video.
    #     gt_idx = ground_truth['video-name'] == videoid
    #     this_video_ground_truth = ground_truth[gt_idx][['f-init',
    #                                                     'f-end']].values
    #
    #     # Compute tiou scores.
    #     tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
    #     score_lst.append(tiou)
    #     score_name.append(videoid)

    # Computes average recall.
    freq_lst = np.array([float(number) for number in 10 ** (np.arange(-1, 0.9, 0.1))])
    matches = np.empty((video_lst.shape[0], freq_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], freq_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            frm_num = frm_nums[score_name[i]]
            # Total positives per video.
            positives[i] = score.shape[0]

            for j, freq in enumerate(freq_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = min(score.shape[1], int(freq * frm_num / 30.0))
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    recall = recall.mean(axis=0)

    return recall, freq_lst


def recall_vs_tiou_thresholds(video_lst, score_lst, score_name, n_effective_proposals, nr_proposals=1000,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    # Get list of videos.
    # video_lst = proposals['video-name'].unique()
    # video_lst_prop = proposals['video-name'].unique()
    # video_lst_gt = ground_truth['video-name'].unique()
    # video_lst = np.intersect1d(video_lst_prop, video_lst_gt)
    # score_lst, score_name = common_pruning_nms(video_lst, proposals, ground_truth)

    # For each video, computes tiou scores among the retrieved proposals.
    # score_lst = []
    # for videoid in video_lst:
    #     # Get proposals for this video.
    #     prop_idx = proposals['video-name'] == videoid
    #     this_video_proposals = proposals[prop_idx][['f-init',
    #                                                 'f-end']].values
    #     scores = proposals[prop_idx]['score'].values
    #     this_video_proposals, keep_indices = np.unique(this_video_proposals)
    #     scores = scores[keep_indices]
    #     sort_idx = scores.argsort()[::-1]
    #     this_video_proposals = this_video_proposals[sort_idx, :]
    #
    #     # this_video_proposals, _ = non_maxima_supression(this_video_proposals, scores, overlap=0.8)
    #     # Get ground-truth instances associated to this video.
    #     gt_idx = ground_truth['video-name'] == videoid
    #     this_video_ground_truth = ground_truth[gt_idx][['f-init',
    #                                                     'f-end']].values
    #
    #     # Compute tiou scores.
    #     tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
    #     score_lst.append(tiou)

    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    pcn = (video_lst.shape[0] * float(nr_proposals)) / n_effective_proposals

    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            # Get number of proposals at the fixed percentage of total retrieved.
            nr_proposals = int(score.shape[1] * pcn)
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()

    return recall, tiou_thresholds




def recall_freq_vs_tiou_thresholds(video_lst, score_lst, score_name, n_effective_proposals, frm_nums, pdefined_freq=1.0,
                                   tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    # Get list of videos.
    # video_lst = proposals['video-name'].unique()
    # video_lst_prop = proposals['video-name'].unique()
    # video_lst_gt = ground_truth['video-name'].unique()
    # video_lst = np.intersect1d(video_lst_prop, video_lst_gt)
    # score_lst, score_name = common_pruning_nms(video_lst, proposals, ground_truth)

    # For each video, computes tiou scores among the retrieved proposals.
    # score_lst = []
    # score_name = []
    # for videoid in video_lst:
    #     # Get proposals for this video.
    #     prop_idx = proposals['video-name'] == videoid
    #     this_video_proposals = proposals[prop_idx][['f-init',
    #                                                 'f-end']].values
    #     # Sort proposals by score.
    #     # sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
    #     # this_video_proposals = this_video_proposals[sort_idx, :]
    #     scores = proposals[prop_idx]['score'].values
    #     this_video_proposals, _ = non_maxima_supression(this_video_proposals, scores, overlap=0.8)
    #     # Get ground-truth instances associated to this video.
    #     gt_idx = ground_truth['video-name'] == videoid
    #     this_video_ground_truth = ground_truth[gt_idx][['f-init',
    #                                                     'f-end']].values
    #
    #     # Compute tiou scores.
    #     tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
    #     score_lst.append(tiou)
    #     score_name.append(videoid)

    freq = pdefined_freq

    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        for i, (score, s_name) in enumerate(zip(score_lst, score_name)):
            # Total positives per video.
            positives[i] = score.shape[0]
            # frm_num = frm_nums[score_name[i]]
            frm_num = frm_nums[s_name]

            nr_proposals = int(min(score.shape[1], freq * frm_num / 30.0))
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()

    return recall, tiou_thresholds



def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou