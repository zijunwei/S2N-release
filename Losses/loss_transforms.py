import torch
import torch.nn.functional as F

def torchVT_scores2indices(scores, topK=1, reduce_dim=True):
    _, indices = scores.topk(topK, -1, largest=True, sorted=True)
    if topK == 1 and reduce_dim:
        indices = indices.squeeze(-1)
    return indices


def torchT_indices1D2scores(indices, n_classes):
    n_samples = indices.size(0)
    scores = torch.zeros(n_samples, n_classes)
    for idx in range(n_samples):
        scores[idx, indices[idx]]=1
    return scores



def torchT_indices2D2scores(indices, n_classes):

    n_samples = indices.size(0)
    n_idxes = indices.size(1)

    scores = torch.zeros(n_samples, n_idxes, n_classes)
    for sample_idx in range(n_samples):
        for id_idx in range(n_idxes):
            scores[sample_idx, id_idx, indices[sample_idx, id_idx]]=1
    return scores


def torchT_indices2D2class3(indices, seq_len):
    n_samples = indices.size(0)

    class3 = torch.zeros(n_samples, seq_len).long()
    for sample_idx in range(n_samples):

        class3[sample_idx][indices[sample_idx,0]] = 1# starting idxes set to 1
        class3[sample_idx][indices[sample_idx,1]] = 2# ending idxes set to 2, others are set to 0
    return class3


def torchVT_classProb2indices(class_probs):
    # convert class probabilities to indices probabilitis
    start_probs = class_probs[:,:,1]
    end_probs = class_probs[:,:,2]
    start_probs = F.softmax(start_probs, dim=1).unsqueeze(-1)
    end_probs = F.softmax(end_probs, dim=1).unsqueeze(-1)
    return torch.cat([start_probs, end_probs], dim=1)


def torchVT_scores2indicesTopK(scores, topK=1):
    _, indices = scores.topk(topK, -1, largest=True, sorted=True)

    return indices


def IoU_Overlaps(pred_segments, groundtruth_segments):

    # correct_order = pred_segments[:,1]>pred_segments[:,0]

    startIndices = torch.stack([pred_segments[:, 0], groundtruth_segments[:, 0]], dim=1)
    endIndices = torch.stack([pred_segments[:,1], groundtruth_segments[:,1]], dim=1)
    UnionStart, _ = torch.min(startIndices, dim=1)
    UnionEnd, _ = torch.max(endIndices, dim=1)

    IntersectStart,_ = torch.max(startIndices, dim=1)
    IntersectEnd,_ = torch.min(endIndices, dim=1)
    intersect = torch.clamp(IntersectEnd - IntersectStart, min=0).float()
    union = (UnionEnd - UnionStart).float()

    overlap = intersect / union
    # overlap = overlap*correct_order
    if overlap.is_cuda:
        overlap = overlap.cpu()
    return overlap.float()


def IoU_OverlapsHardThres(pred_segments, groundtruth_segments, thres=0.65):

    correct_order = pred_segments[:,1]>pred_segments[:,0]

    startIndices = torch.stack([pred_segments[:, 0], groundtruth_segments[:, 0]], dim=1)
    endIndices = torch.stack([pred_segments[:,1], groundtruth_segments[:,1]], dim=1)
    IntersectStart,_ = torch.max(startIndices, dim=1)
    IntersectEnd,_ = torch.min(endIndices, dim=1)
    overlap = torch.clamp(IntersectEnd - IntersectStart, min=0).float()
    #TODO: update here!
    pred_sizes = torch.clamp((pred_segments[:,1]-pred_segments[:,0]).float(), min=1e-4)
    groundtruth_sizes = torch.clamp((groundtruth_segments[:,1]-groundtruth_segments[:,0]).float(), min=1e-4)
    precision = overlap/groundtruth_sizes
    recall = overlap*1./pred_sizes
    f1_scores = precision*recall*2/(precision+recall)
    overlap = f1_scores>thres
    overlap = overlap*correct_order


    return overlap.float()

def F1_Overlaps(pred_segments, groundtruth_segments, thres=0.65):

    correct_order = pred_segments[:,1]>pred_segments[:,0]

    startIndices = torch.stack([pred_segments[:, 0], groundtruth_segments[:, 0]], dim=1)
    endIndices = torch.stack([pred_segments[:,1], groundtruth_segments[:,1]], dim=1)
    IntersectStart,_ = torch.max(startIndices, dim=1)
    IntersectEnd,_ = torch.min(endIndices, dim=1)
    overlap = torch.clamp(IntersectEnd - IntersectStart, min=0).float()
    #TODO: update here!
    pred_sizes = torch.clamp((pred_segments[:,1]-pred_segments[:,0]).float(), min=1e-4)
    groundtruth_sizes = torch.clamp((groundtruth_segments[:,1]-groundtruth_segments[:,0]).float(), min=1e-4)
    precision = overlap/groundtruth_sizes
    recall = overlap*1./pred_sizes
    f1_scores = precision*recall*2/(precision+recall)
    overlap = f1_scores>thres
    overlap = overlap*correct_order


    # if overlap.is_cuda:
    #     overlap = overlap.cpu()

    return overlap.float()
