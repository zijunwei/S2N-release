from scipy.optimize import linear_sum_assignment
import numpy as np
# this uses fixed matching...
# the order is following
from h_assign import compute_single_iou

def Assign_Batch(gt_positions, pred_positions, gt_valid, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()

    max_predictions = pred_positions.shape[1]
    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            topK = min(max_predictions, int(gt_valid[batch_idx]))
            target_scores[batch_idx, :topK] = 1
            corresponding_positions[batch_idx, :topK] = gt_positions[batch_idx, :topK]

    return target_scores, corresponding_positions


def Assign_Batch_v2(gt_positions, pred_positions, gt_valid, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()

    max_predictions = pred_positions.shape[1]
    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            s_valid = gt_valid[batch_idx, 0]
            total_valid += s_valid
            topK = min(max_predictions, int(gt_valid[batch_idx]))
            target_scores[batch_idx, :topK] = 1
            corresponding_positions[batch_idx, :topK] = gt_positions[batch_idx, :topK]
            for s_idx in range(s_valid):
                total_iou += compute_single_iou(gt_positions[batch_idx, s_idx,:], pred_positions[batch_idx, s_idx, :])

    return target_scores, corresponding_positions, total_valid, total_iou


def Assign_Batch_v2_repression(gt_positions, pred_positions, gt_valid, gt_overlaps, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    if not isinstance(gt_overlaps, (np.ndarray, np.generic)):
        gt_overlaps = gt_overlaps.data.cpu().numpy()

    max_predictions = pred_positions.shape[1]
    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    target_overlaps = np.zeros(pred_positions.shape[0:2])

    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            s_valid = gt_valid[batch_idx, 0]
            total_valid += s_valid
            topK = min(max_predictions, int(gt_valid[batch_idx]))
            target_scores[batch_idx, :topK] = 1
            target_overlaps[batch_idx, :topK] = gt_overlaps[batch_idx, :topK]
            corresponding_positions[batch_idx, :topK] = gt_positions[batch_idx, :topK]
            for s_idx in range(s_valid):
                total_iou += compute_single_iou(gt_positions[batch_idx, s_idx,:], pred_positions[batch_idx, s_idx, :])

    return target_scores, corresponding_positions, target_overlaps, total_valid, total_iou


def Assign_Batch_all_valid(gt_positions, pred_positions, n_predictions=6, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    # if not isinstance(gt_valid, (np.ndarray, np.generic)):
    #     gt_valid = gt_valid.data.cpu().numpy()

    max_predictions = pred_positions.shape[1]

    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    for batch_idx in range(batch_size):
        # if gt_valid[batch_idx]>0:
            topK = min(max_predictions, n_predictions)
            target_scores[batch_idx, :topK] = 1
            corresponding_positions[batch_idx, :topK] = gt_positions[batch_idx, :topK]

    return target_scores, corresponding_positions


if __name__ == '__main__':


    print("DB")



