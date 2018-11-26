import numpy as np

def accuracy_topN(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output = output.view(-1, output.size(2))
    # target = target.view(-1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def IoU(ground_truths, pred_offsets, labels):
    preds_start = ground_truths[:, 0] + pred_offsets[:, 0]
    preds_end = ground_truths[:, 1] + pred_offsets[:, 1]

    preds = np.stack((preds_start, preds_end), axis=1)
    total_ious = 0

    n_valid = np.sum(labels)

    if n_valid == 0:
        return total_ious, 1e-10
    else:
        for idx in range(len(labels)):
            if labels[idx] >0:
                s_ious = calculate_IoU(preds[idx], ground_truths[idx])
                if s_ious>0:
                    total_ious += s_ious


        return total_ious/n_valid, n_valid


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / ((union[1] - union[0])+1e-12)
    return iou


