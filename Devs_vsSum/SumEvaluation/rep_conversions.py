#convertion between different outputs
import numpy as np
import knapsack

def convert_seg2interval(segs, n_frames=None):
    if segs[0] !=0:
        segs.insert(0, 0)
    if n_frames is not None and segs[-1]!=n_frames:
        segs.append(n_frames)
    intervals = []
    for i in range(len(segs)-1):
        if segs[i]==segs[i+1]:
            segs[i+1] = segs[i]+1
        intervals.append([segs[i], segs[i+1]])
    return intervals




def framescore2keyshots(framescore, segments, lratio=0.15):
    # segementations is a list indicating starting and ending
    n_frames = len(framescore)
    assert segments[-1][-1] == n_frames
    seg_values = []
    seg_weights = []
    for s_seg in segments:
        # TODO: These two gets similar results
        seg_values.append(sum(framescore[s_seg[0]:s_seg[1]])/float(s_seg[1] - s_seg[0]))
        # seg_values.append(sum(framescore[s_seg[0]:s_seg[1]])*1.0)

        seg_weights.append(max(s_seg[1]-s_seg[0], 1))


    # rank the intervals:
    # sorted_idx = np.argsort(-seg_values)
    n_selected_frames = int(lratio * n_frames)

    picks = knapsack.knapsack_dp(seg_values, seg_weights, len(seg_weights), n_selected_frames)
    selected_segments = [segments[i] for i in picks]
    return selected_segments


def keyshots2keyframe(keyshots, framescore=None):
    keyframes = []
    for s_shot in keyshots:
        starting_idx = s_shot[0]
        if framescore is not None:
            seg_frames = np.asarray(framescore[s_shot[0]:s_shot[1]])
            selected_frame_idx = np.argmax(seg_frames)
            keyframes.append(starting_idx + selected_frame_idx)
        else:
            ending_idx = s_shot[1]
            keyframes.append(int((starting_idx + ending_idx)/2))

    return keyframes

def keyshots2frame01scores(keyshots, n_frames=None):
    if n_frames is None:
        n_frames = keyshots[-1][1]
    framescore = np.zeros(n_frames)
    for s_shot in keyshots:
        framescore[s_shot[0]:s_shot[1]] = 1

    return framescore

def framescore2frame01score_segmentbased_greedy(framescores, segments, lratio=0.15):
    nframes = len(framescores)
    segment_scores = []
    for i_seg in segments:
        seg_len = i_seg[1] - i_seg[0]
        segscore = sum(framescores[i_seg[0]:i_seg[1]])
        segment_scores.append(segscore*1./seg_len)
    segment_scores = np.asarray(segment_scores)
    sorted_idx = np.argsort(-segment_scores)
    n_len = 0
    counter = 0
    framescores01 = np.zeros_like(framescores)
    while n_len<lratio*nframes:
        curIdx = sorted_idx[counter]
        framescores01[segments[curIdx][0]:segments[curIdx][1]]=1
        n_len += -segments[curIdx][0]+segments[curIdx][1]
        counter += 1
    return framescores01


def keyframes2keyshots():
    #check zhang's supp
    raise  NotImplementedError


# def smooth_segmentscores(framescores, ,n_frames=None):
#     if n_frames is None:
#         n_frames = keyshots[-1][1]
#     framescore = np.zeros(n_frames)
#     for s_shot in keyshots:
#         framescore[s_shot[0]:s_shot[1]] = 1
#
#     return framescore

# for F1 score
def get_max_interval_size(intevals):
    max_len = 0
    for s_segs in intevals:
        if s_segs[1]-s_segs[0]>max_len:
            max_len = s_segs[1] - s_segs[0]
    return max_len

def framescore2frame01score_sort(framescore, lratio=0.15):
    nframes = len(framescore)
    frame01score = np.zeros(nframes)
    frameindices = np.argsort(-framescore)
    endpoints = int(lratio*nframes)
    for i in range(nframes):
        frame01score[frameindices[i]] = 1 if i <= endpoints else 0
    return frame01score

def framescore2frame01score(framescore, segments, lratio=0.15):
    n_frames = len(framescore)
    intevals = convert_seg2interval(segments, n_frames)
    # max_len = get_max_interval_size(intevals)
    # if max_len >= n_frames * lratio:
    #     lratio = max_len*1.1 / n_frames
    selected_segmentations = framescore2keyshots(framescore, intevals, lratio)
    frame01score = keyshots2frame01scores(selected_segmentations, n_frames)
    if sum(frame01score) == 0:

        qtl = np.percentile(framescore, (1-lratio)*100)
        frame01score[framescore>qtl]=1

    if sum(frame01score) == 0:
         # print "DEBUG"
        i = 0
        indices = np.argsort(-framescore)
        while sum(frame01score) < lratio*n_frames:
            frame01score[indices[i]]= 1
            i+=1

    return frame01score

def framescore2frame01score_inteval(framescore, segments, lratio=0.15):
    n_frames = len(framescore)
    # max_len = get_max_interval_size(intevals)
    # if max_len >= n_frames * lratio:
    #     lratio = max_len*1.1 / n_frames
    selected_segmentations = framescore2keyshots(framescore, segments, lratio)
    frame01score = keyshots2frame01scores(selected_segmentations, n_frames)
    if sum(frame01score) == 0:

        qtl = np.percentile(framescore, (1-lratio)*100)
        frame01score[framescore>qtl]=1

    if sum(frame01score) == 0:
         # print "DEBUG"
        i = 0
        indices = np.argsort(-framescore)
        while sum(frame01score) < lratio*n_frames:
            frame01score[indices[i]]= 1
            i+=1

    return frame01score


# TODO: for AP score, check Panda's paper
def framescore2smthframescore(framescore, segments, lratio=0.15):
    n_frames = len(framescore)
    intevals = convert_seg2interval(segments, n_frames)
    selected_segmentations = framescore2keyshots(framescore, intevals, lratio)
    smoothed_framescore = keyshots2frame01scores(intevals, n_frames)
    return smoothed_framescore


def selectframe2segscores(selectedIndices, segments, nframes, sample_rate=1):
    # convert the selected Indices to segmental scores
    intevals = convert_seg2interval(segments, nframes)
    framescores = np.zeros(nframes)
    for s_idx in selectedIndices:
        for s_inteval in intevals:
            if s_inteval[0]<=s_idx*sample_rate and s_inteval[1] >= s_idx*sample_rate:
                framescores[s_inteval[0]:s_inteval[1]] = 1

    return framescores

def selecteTopSegments(segments, segment_scores, n_frames, lratio=0.15):

    seg_values = []
    seg_weights = []
    for (s_value, s_seg) in zip(segment_scores, segments):
        # TODO: These two gets similar results
        seg_weights.append(max(s_seg[1]-s_seg[0], 1))
        seg_values.append(s_value*(s_seg[1] - s_seg[0])*1.)
    # debug:
    # for i, (s_weight, s_value) in enumerate(zip(seg_weights, seg_values)):
    #     print('{:d}\t{:d}\t{:d}'.format(i, s_value, s_weight))
    n_selected_frames = int(lratio * n_frames)

    picks = knapsack.knapsack_dp(seg_values, seg_weights, len(seg_weights), n_selected_frames)
    selected_segments = [segments[i] for i in picks]
    return selected_segments


def keyshots2framescores(segments, segment_scores, n_frames):
    framescores = np.zeros(n_frames)
    for (s_segment, s_score) in zip(segments, segment_scores):
        framescores[s_segment[0]:s_segment[1]] = s_score
    return framescores

# def convertSegment2ScoreswithConfidences(segments, segment_scores, segment_conficences, nFrames):
#     # convert segment scores and confidences to list scores
#     segments = np.asarray(segments)
#     segment_scores = np.asarray(segment_scores)
#     segment_conficences = np.asarray(segment_conficences)
#     framescreos = np.zeros(nFrames)
#     frameconfidences = np.zeros(nFrames)
#
#     updated_segments = []
#     updated_scores = []
#     sorted_idx = np.argsort(-segment_conficences)
#     for s_idx in sorted_idx:
#         s_segment = segments[s_idx]
#         s_confidence = segment_conficences[s_idx]
#         s_score = segment_scores[s_idx]
#         starting_idx = s_segment[0]
#         endind_idx = s_segment[1]
#
#         for position_idx in range(s_segment[0], s_segment[1]):
#             if frameconfidences[s_idx] < s_confidence:
#                 starting_idx = position_idx
#                 break
#
#
#
#     for (s_segment, s_score, s_confidence) in zip(segments, segment_scores, segment_conficences):
#
#         for idx in range(s_segment[0], s_segment[1]):


if __name__ == '__main__':
    #TODO: Not tested!
    pass








