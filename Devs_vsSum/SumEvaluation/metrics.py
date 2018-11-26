import sklearn.metrics as sk_metrics
import numpy as np
from sklearn.metrics import f1_score

def average_precision(y_true, y_score):
    # ytrue is a list of 1 and 0
    # y score is a list of scores
    assert len(y_true) == len(y_score), 'average precision score need the two lists tobe the same size'

    return sk_metrics.average_precision_score(y_true, y_score)


def F1_score(y_true, y_score):
    # ytrue is a list of 1 and 0
    # y score is a list of 1 and 0
    eps = 1e-5
    assert len(y_true) == len(y_score), 'F1 score need the two lists tobe the same size'
    n_true = y_true.count(1)
    n_score = y_score.count(1)
    overlap = 0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_score[i]==1:
            overlap += 1
    #todo: revert the precision and recall
    precision = overlap * 1. / n_score
    recall = overlap * 1. / n_true
    # precision = overlap*1./n_true
    # recall = overlap*1./n_score
    return 2.*recall*precision/(precision+recall+eps)

# def F1_score2(y_true, y_score):
#     # ytrue is a list of 1 and 0
#     # y score is a list of 1 and 0
#     eps = 1e-5
#     assert len(y_true) == len(y_score), 'F1 score need the two lists tobe the same size'
#     n_true = y_true.count(1)
#     n_score = y_score.count(1)
#     overlap = 0
#     for i in range(len(y_true)):
#         if y_true[i]==y_score[i]:
#             overlap += 1
#     #todo: revert the precision and recall
#     precision = overlap * 1. / len(y_score)
#     recall = overlap * 1. / len(y_score)
#     # precision = overlap*1./n_true
#     # recall = overlap*1./n_score
#     return 2.*recall*precision/(precision+recall+eps)



def averaged_F1_score(y_trues, y_score):
    F1_scores = []
    for s_y_true in y_trues:
        if isinstance(s_y_true, np.ndarray):
            s_y_true = s_y_true.tolist()
        F1_scores.append(F1_score(s_y_true, y_score))

    return sum(F1_scores)/float(len(F1_scores))


def max_F1_score(y_trues, y_score):
    F1_scores = []
    for s_y_true in y_trues:
        if isinstance(s_y_true, np.ndarray):
            s_y_true = s_y_true.tolist()
        F1_scores.append(F1_score(s_y_true, y_score))

    return max(F1_scores)

if __name__ == '__main__':
    # test on f1 score
    y_true = [0, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    my_F1Score = F1_score(y_true, y_pred)
    sk_F1Score = f1_score(y_true, y_pred)
    print "DEBUG"