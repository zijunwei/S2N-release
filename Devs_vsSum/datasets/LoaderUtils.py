import numpy as np
import random
random.seed(0)

class Segment():
    def __init__(self, video_name):
        self.video_stem = video_name
    def initVector(self, indicatorVector):
        self.indicatorVector = indicatorVector
        indices = np.argwhere(indicatorVector>0)[0]
        self.startIdx = indices[0]
        self.endIdx = indices[-1]
        self.seq_length = len(indicatorVector)
        self.score = self.indicatorVector[self.startIdx]
    def initId(self, startId, endId, length, score):
        self.indicatorVector = np.zeros(length)
        self.indicatorVector[startId:endId]=1
        self.startIdx = startId
        self.endIdx = endId
        self.seq_length = length
        self.score = score


class TrainSegment():
    def __init__(self, video_name):
        self.video_stem = video_name
    def initVector(self, indicatorVector):
        self.indicatorVector = indicatorVector
        indices = np.argwhere(indicatorVector>0)[0]
        self.startIdx = indices[0]
        self.endIdx = indices[-1]
        self.seq_length = len(indicatorVector)
        self.score = self.indicatorVector[self.startIdx]
    def initId(self, startId, endId, length, score):
        self.indicatorVector = np.zeros(length)
        self.indicatorVector[startId:endId]=1
        self.startIdx = startId
        self.endIdx = endId
        self.seq_length = length
        self.score = score
    def addCandidate(self, p_startIdx=None, p_endIdx=None):
        self.p_startIdx = p_startIdx or self.startIdx
        self.p_endIdx = p_endIdx or self.endIdx


class DataInstance():
    def __init__(self, video_name):
        self.video_name = video_name

    def initInstance(self, position_vector, startId, endId, gt_score, seq_length):
        self.position_vector = position_vector
        self.startId = startId
        self.endId = endId
        self.seq_len = seq_length
        self.gt_score = gt_score

    def addRdIdx(self, rd_startId, rd_endId, rd_score):
        self.rd_startId = rd_startId
        self.rd_endId = rd_endId
        self.rd_score = rd_score



def unify_ftlbl(feature, label):
    feature_length = feature.shape[0]
    label_length = label.shape[0]
    min_length = min(feature_length, label_length)
    video_features = feature[0:min_length, :]
    labels = label[0:min_length, :]
    return video_features, labels


def convertlabels2segs(labels):

    n_users = labels.shape[1] if len(labels.shape)>1 else 1
    n_frames = labels.shape[0]
    label_indicators = []
    for user_idx in range(n_users):
        s_labels = labels[:, user_idx]
        startIdx = 0
        endIdx = n_frames
        for idx in range(1, n_frames):
            if s_labels[idx] == 1 and s_labels[idx - 1] == 0:
                startIdx = idx
            if (s_labels[idx] == 0 and s_labels[idx - 1] == 1) or (s_labels[idx] == 1 and idx+1 == n_frames):
                endIdx = idx
                if [startIdx, endIdx] not in label_indicators:
                    label_indicators.append([startIdx, endIdx])

    return label_indicators


def convertscores2segs(scores):
    n_users = scores.shape[1] if len(scores.shape)>1 else 1
    assert n_users == 1, 'number of users should be 1'
    n_frames = scores.shape[0]
    segment_indicators = []
    segment_scores = []
    for user_idx in range(n_users):
        s_scores = scores[:, user_idx]
        startIdx = 0
        endIdx = 1
        for idx in range(1, n_frames):
            if s_scores[idx] == s_scores[idx - 1]:
                endIdx += 1
                if idx == n_frames-1:
                    segment_indicators.append([startIdx, n_frames-1])   
                    segment_scores.append(s_scores[startIdx])
            else:
                segment_indicators.append([startIdx, endIdx])
                segment_scores.append(s_scores[startIdx])
                startIdx = idx
                endIdx = idx+1

    return segment_indicators, segment_scores


def createRdIdxes(start_idx, end_idx, clip_size):
        rstart_idx = 0
        rend_idx = clip_size
        r_situation = random.randint(0, 4)
        if r_situation == 0:
            rstart_idx = random.randint(0, clip_size - 2)
            rend_idx = random.randint(rstart_idx + 1, clip_size - 1)
        elif r_situation == 1:
            rstart_idx = random.randint(0, start_idx)
            rend_idx = random.randint(start_idx + 1, end_idx)
        elif r_situation == 2:
            rstart_idx = random.randint(start_idx, end_idx - 1)
            rend_idx = random.randint(end_idx , clip_size - 1)
        elif r_situation == 3:
            rstart_idx = random.randint(0, start_idx)
            rend_idx = random.randint(end_idx - 1, clip_size - 1)
        elif r_situation == 4:
            rstart_idx = random.randint(start_idx, end_idx - 1)
            rend_idx = random.randint(rstart_idx+1, end_idx)

        return rstart_idx, rend_idx


def get_avg_seg_features(features, cpts, num_segments):
    [fea_len, fea_dim] = features.shape
    avged_features = np.zeros([num_segments, fea_dim])
    assert cpts[-1][-1] == fea_len - 1, 'last change point should be feature len - 1 %d / %d'%(cpts[-1][-1], fea_len)
    for seg_id in range(num_segments):
        avged_features[seg_id] = np.mean(features[cpts[seg_id, 0]: cpts[seg_id, 1]+1, :], 0)
    return avged_features


def get_avg_scores(scores, cpts, num_segments):
    [seq_len, num_users] = scores.shape
    avged_scores = np.zeros([num_segments, num_users])
    scores = scores.astype('float')
    for seg_id in range(num_segments):
        avged_scores[seg_id] = np.mean(scores[cpts[seg_id, 0]: cpts[seg_id, 1]+1, :], 0)
    return avged_scores    