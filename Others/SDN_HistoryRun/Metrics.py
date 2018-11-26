import numpy as np

def get_avg_iou(pred_positions, assigned_positions, assigned_scores=None):
    if assigned_scores is None:
        selected_pred_positions = pred_positions
        selected_assigned_positions = assigned_positions
    else:
        selected_pred_positions = pred_positions[assigned_scores==1]
        selected_assigned_positions = assigned_positions[assigned_scores == 1]

    invalid_iou_idx = (selected_pred_positions[:, 1] - selected_pred_positions[:, 0] < 0)

    y1 = np.maximum(selected_assigned_positions[:, 0], selected_pred_positions[:, 0])
    y2 = np.minimum(selected_assigned_positions[:, 1], selected_pred_positions[:, 1])

    pred_areas = np.maximum(selected_pred_positions[:, 1] - selected_pred_positions[:, 0]+1, 0)
    assigned_areas = np.maximum(selected_assigned_positions[:, 1] - selected_assigned_positions[:, 0]+1, 0)

    intersection =  np.maximum(y2 - y1+1, 0)
    union = pred_areas + assigned_areas - intersection
    iou = intersection*1. / (union + 1e-8)
    iou[invalid_iou_idx]=0
    return np.sum(iou)


def get_avg_iou2(pred_positions, assigned_positions, assigned_scores=None):
    if assigned_scores is None:
        selected_pred_positions = pred_positions
        selected_assigned_positions = assigned_positions
    else:
        selected_pred_positions = pred_positions[assigned_scores==1]
        selected_assigned_positions = assigned_positions[assigned_scores == 1]

    effective_pairs = selected_assigned_positions.shape[0]
    invalid_iou_idx = (selected_pred_positions[:, 1] - selected_pred_positions[:, 0] <= 0)

    y1 = np.maximum(selected_assigned_positions[:, 0], selected_pred_positions[:, 0])
    y2 = np.minimum(selected_assigned_positions[:, 1], selected_pred_positions[:, 1])

    pred_areas = np.maximum(selected_pred_positions[:, 1] - selected_pred_positions[:, 0]+1, 0)
    assigned_areas = np.maximum(selected_assigned_positions[:, 1] - selected_assigned_positions[:, 0]+1, 0)

    intersection =  np.maximum(y2 - y1+1, 0)
    union = pred_areas + assigned_areas - intersection
    iou = intersection*1. / (union + 1e-8)
    iou[invalid_iou_idx]=0
    return np.sum(iou), effective_pairs

def get_avg_iou3(pred_positions, assigned_positions, assigned_scores=None):
    if assigned_scores is None:
        selected_pred_positions = pred_positions
        selected_assigned_positions = assigned_positions
    else:
        selected_pred_positions = pred_positions[assigned_scores>0]
        selected_assigned_positions = assigned_positions[assigned_scores >0 ]

    effective_pairs = selected_assigned_positions.shape[0]
    invalid_iou_idx = (selected_pred_positions[:, 1] - selected_pred_positions[:, 0] <= 0)

    y1 = np.maximum(selected_assigned_positions[:, 0], selected_pred_positions[:, 0])
    y2 = np.minimum(selected_assigned_positions[:, 1], selected_pred_positions[:, 1])

    pred_areas = np.maximum(selected_pred_positions[:, 1] - selected_pred_positions[:, 0]+1, 0)
    assigned_areas = np.maximum(selected_assigned_positions[:, 1] - selected_assigned_positions[:, 0]+1, 0)

    intersection =  np.maximum(y2 - y1+1, 0)
    union = pred_areas + assigned_areas - intersection
    iou = intersection*1. / (union + 1e-8)
    iou[invalid_iou_idx]=0
    return np.sum(iou), effective_pairs