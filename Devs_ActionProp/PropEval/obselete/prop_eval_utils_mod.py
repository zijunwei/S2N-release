import  numpy as np
import progressbar
#compare to the original version:
# 1. removed the negatives categories
# 2. removed the repeatatives
# 3.

def average_recall_vs_nr_proposals(proposals, ground_truth,
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
    video_lst = proposals['video-name'].unique()
    positive_ids = ground_truth['label-idx'] > -1
    ground_truth = ground_truth[positive_ids]
    gt_video_list = ground_truth['video-name'].unique()
    video_lst = list(set(video_lst.tolist()).intersection(gt_video_list.tolist()))

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init',
                                                    'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        this_video_ground_truth = np.unique(this_video_ground_truth, axis=0)
        # this_video_groud_truth_categories = ground_truth[gt_idx]['label-idx']==-1


        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 101) / 100.0
    matches = np.empty((len(video_lst), pcn_lst.shape[0]))
    positives = np.empty(len(video_lst))
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
    proposals_per_video = pcn_lst * (float(proposals.shape[0]) / len(video_lst))

    return recall, proposals_per_video


def average_recall_vs_freq(proposals, ground_truth, frm_nums,
                           tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    # Get list of videos.
    video_lst = proposals['video-name'].unique()
    positive_ids = ground_truth['label-idx'] > -1
    ground_truth = ground_truth[positive_ids]
    gt_video_list = ground_truth['video-name'].unique()
    video_lst = list(set(video_lst.tolist()).intersection(gt_video_list.tolist()))

    # gt_video_list = ground_truth['video-name'].unique()
    # video_lst = list(set(video_lst.tolist()).intersection(gt_video_list.tolist()))
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    score_name = []
    for videoid in video_lst:
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init',
                                                    'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        this_video_ground_truth = np.unique(this_video_ground_truth, axis=0)

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)
        score_name.append(videoid)

    # Computes average recall.
    freq_lst = np.array([float(number) for number in 10 ** (np.arange(-1, 0.9, 0.1))])
    matches = np.empty((len(video_lst), freq_lst.shape[0]))
    positives = np.empty(len(video_lst))
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


def recall_vs_tiou_thresholds(proposals, ground_truth, nr_proposals=1000,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    # Get list of videos.
    video_lst = proposals['video-name'].unique()
    positive_ids = ground_truth['label-idx'] > -1
    ground_truth = ground_truth[positive_ids]
    gt_video_list = ground_truth['video-name'].unique()
    video_lst = list(set(video_lst.tolist()).intersection(gt_video_list.tolist()))
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init',
                                                    'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        this_video_ground_truth = np.unique(this_video_ground_truth, axis=0)

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    pcn = (len(video_lst) * float(nr_proposals)) / proposals.shape[0]

    # Computes recall at different tiou thresholds.
    matches = np.empty((len(video_lst), tiou_thresholds.shape[0]))
    positives = np.empty(len(video_lst))
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


def recall_freq_vs_tiou_thresholds(proposals, ground_truth, frm_nums,
                                   tiou_thresholds=np.arange(0.05, 1.05, 0.05), pdefined_freq=1.0):
    # Get list of videos.
    video_lst = proposals['video-name'].unique()
    positive_ids = ground_truth['label-idx'] > -1
    ground_truth = ground_truth[positive_ids]
    gt_video_list = ground_truth['video-name'].unique()
    video_lst = list(set(video_lst.tolist()).intersection(gt_video_list.tolist()))
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    score_name = []
    for videoid in video_lst:
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init',
                                                    'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        this_video_ground_truth = np.unique(this_video_ground_truth, axis=0)

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)
        score_name.append(videoid)

    freq = pdefined_freq

    # Computes recall at different tiou thresholds.
    matches = np.empty((len(video_lst), tiou_thresholds.shape[0]))
    positives = np.empty(len(video_lst))
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            frm_num = frm_nums[score_name[i]]
            nr_proposals = int(min(score.shape[1], freq * frm_num/30.0))
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