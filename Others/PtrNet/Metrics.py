import numpy as np

def match_accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    seq_len = target.size(1)
    batch_size = target.size(0)
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    # for s_element in range(batch_size):
    n_correct = np.sum((target == output).astype(float))
    n_total = batch_size*seq_len

    res = n_correct * 1. / n_total
    return res


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



def accuracy_F1(output, target, topk=(1,)):
    #TODO: check the correctness!
    """Computes the precision@k for the specified values of k"""
    # output = output.view(-1, output.size(2))
    # target = target.view(-1)
    maxk = max(topk)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    pred = pred.view(-1,  2, maxk)
    target = target.view(-1, 2)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    total_f1s = []
    for k in topk:
        s_total_f1s = []
        for idx in range(batch_size/2):
            target_start = target[idx][0]
            target_end = target[idx][1]
            len_target = target_end - target_start
            s_f1s = []
            for item_start in range(k):
                for item_end in range(k):
                    pred_start = pred[idx][0][item_start]
                    pred_end = pred[idx][1][item_end]
                    len_pred =  pred_end - pred_start
                    if len_pred >0:
                        # if target_end <= pred_start or pred_end <= target_start:
                        s_diff = min(pred_end, target_end) - max(pred_start, target_start)

                        len_overlap = s_diff if s_diff>0 else 0
                        if len_overlap == 0:
                            s_f1s.append(0)
                        else:
                            s_p = len_overlap*1. / len_target
                            s_r = len_overlap*1. / len_pred
                            s_f1 = 2* s_p * s_r / (s_p + s_r)
                            s_f1s.append(s_f1)
                    else:
                        s_f1s.append(0)
            s_f1 = max(s_f1s)
            s_total_f1s.append(s_f1)

        total_f1s.append(sum(s_total_f1s)*2./batch_size)

    return total_f1s








