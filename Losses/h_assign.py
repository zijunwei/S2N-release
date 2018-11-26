from scipy.optimize import linear_sum_assignment
import numpy as np
import Devs_ActionProp.PropEval.Utils as PropUtils

def toCostMatrix(profit_matrix):
    """
    Converts a profit matrix into a cost matrix.
    Expects NumPy objects as input.
    """
    # subtract profit matrix from a matrix made of the max value of the profit matrix
    matrix_shape = profit_matrix.shape
    offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
    cost_matrix = offset_matrix - profit_matrix
    return cost_matrix

def toProfitMatrix(cost_matrix):
    """
    Converts a profit matrix into a cost matrix.
    Expects NumPy objects as input.
    """
    # subtract profit matrix from a matrix made of the max value of the profit matrix
    matrix_shape = cost_matrix.shape
    offset_matrix = np.ones(matrix_shape, dtype=int) * cost_matrix.max()
    profit_matrix = offset_matrix - cost_matrix
    return profit_matrix

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[1], boxes[:, 1])

    intersection = np.maximum(y2 - y1 + 1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / (union + 1e-8)
    return iou


def compute_overlaps(grtruth_bboxes, predict_bboxes):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    grtruth_areas = (grtruth_bboxes[:, 1] - grtruth_bboxes[:, 0]) + 1
    predict_areas = np.maximum((predict_bboxes[:, 1] - predict_bboxes[:, 0]) + 1, 0)

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((grtruth_bboxes.shape[0], predict_bboxes.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = predict_bboxes[i]
        if predict_areas[i] == 0:
            overlaps[:, i] = 0
        else:
            overlaps[:, i] = compute_iou(box2, grtruth_bboxes, predict_areas[i], grtruth_areas)
    return overlaps

def compute_l1distance(box, bboxes):
    return np.abs(box[0] - bboxes[:, 0]) + np.abs(box[1]-bboxes[:, 1])

def compute_l1s(grtruth_bboxes, predict_bboxes):
    # predict_areas = np.maximum((predict_bboxes[:, 1] - predict_bboxes[:, 0]) + 1, 0)
    l1s = np.zeros((grtruth_bboxes.shape[0], predict_bboxes.shape[0]))
    for i in range(l1s.shape[1]):
        box2 = predict_bboxes[i]
        # if predict_areas[i] == 0:
        #     l1s[:, i] = 0
        # else:
        l1s[:, i] = compute_l1distance(box2, grtruth_bboxes)
    return l1s


def Assign(gt_intevals, pred_intevals, thres=0.25):
    # gt_intevals M by 2
    # pred_intevals N by 2
    # for the cost, the smaller, the better

    n_pred = pred_intevals.shape[0]
    n_gt = gt_intevals.shape[0]
    r_j = range(n_pred, 0, -1)

    # o_ij cost
    o_ij = compute_overlaps(gt_intevals, pred_intevals)
    d_ij = compute_l1s(gt_intevals, pred_intevals)

    o_ij[o_ij >= thres] = 1
    o_ij[o_ij < thres] = 0

    r_mat = np.expand_dims(np.array(r_j), axis=0).repeat(n_gt, axis=0)
    d_ij = toProfitMatrix(d_ij)

    r_max = max(np.max(r_mat),1)*10
    d_max = max(np.max(d_ij),1)*10

    match_profit = o_ij * d_max * r_max + r_mat * d_max + d_ij
    match_cost = toCostMatrix(match_profit)

    row_inds, col_inds = linear_sum_assignment(match_cost)

    return row_inds, col_inds,  o_ij


def Assign_various(gt_intevals, pred_intevals, thres=0.25, r_max=None, d_max=None):
    # gt_intevals M by 2
    # pred_intevals N by 2
    # for the cost, the smaller, the better

    n_pred = pred_intevals.shape[0]
    n_gt = gt_intevals.shape[0]
    r_j = range(n_pred, 0, -1)

    # o_ij cost
    o_ij = compute_overlaps(gt_intevals, pred_intevals)
    d_ij = compute_l1s(gt_intevals, pred_intevals)

    o_ij[o_ij >= thres] = 1
    o_ij[o_ij < thres] = 0

    r_mat = np.expand_dims(np.array(r_j), axis=0).repeat(n_gt, axis=0)
    d_ij = toProfitMatrix(d_ij)

    if r_max is None:
        r_max = max(np.max(r_mat),1)*10
    if d_max is None:
        d_max = max(np.max(d_ij), 1)*10


    match_profit = o_ij * d_max * r_max + r_mat * d_max + d_ij
    match_cost = toCostMatrix(match_profit)

    row_inds, col_inds = linear_sum_assignment(match_cost)

    return row_inds, col_inds,  o_ij


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

    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx,0]>0:
            row_inds, col_inds, o_ij = Assign(gt_positions[batch_idx, :int(gt_valid[batch_idx,0]),:], pred_positions[batch_idx], thres)

            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]

    return target_scores, corresponding_positions



def compute_single_iou(box1, box2):
    """Calculates IoU of the given box with the array of the given boxes.
    simplext iou computation with 2 segments
    """
    # Calculate intersection areas
    y1 = np.maximum(box1[0], box2[0])
    y2 = np.minimum(box1[1], box2[1])
    box1_area = np.maximum(box1[1] - box1[0] + 1, 0)
    box2_area = np.maximum(box2[1] - box2[0] + 1, 0)
    intersection = np.maximum(y2 - y1 + 1, 0)
    union = box1_area + box2_area - intersection
    iou = intersection / (union + 1e-8)
    return iou


def Assign_simple(gt_intevals, pred_intevals, thres=0.25):
    # gt_intevals M by 2
    # pred_intevals N by 2
    # for the cost, the smaller, the better

    n_pred = pred_intevals.shape[0]
    n_gt = gt_intevals.shape[0]
    r_j = range(n_pred, 0, -1)

    # o_ij cost
    o_ij = compute_overlaps(gt_intevals, pred_intevals)
    d_ij = compute_l1s(gt_intevals, pred_intevals)

    o_ij[o_ij >= thres] = 1
    o_ij[o_ij < thres] = 0

    r_mat = np.expand_dims(np.array(r_j), axis=0).repeat(n_gt, axis=0)
    d_ij = toProfitMatrix(d_ij)

    r_max = 0
    d_max = 0

    match_profit = o_ij * d_max * r_max + r_mat * d_max + d_ij
    match_cost = toCostMatrix(match_profit)

    row_inds, col_inds = linear_sum_assignment(match_cost)

    return row_inds, col_inds,  o_ij

def IOUMatch(gt_intevals, pred_intevals):
    # gt_intevals M by 2
    # pred_intevals N by 2
    # simplext matching using IOU matching!

    # r_j = range(n_pred, 0, -1)

    # o_ij cost
    o_ij = compute_overlaps(gt_intevals, pred_intevals)

    match_cost = toCostMatrix(o_ij)

    row_inds, col_inds = linear_sum_assignment(match_cost)

    return row_inds, col_inds,  o_ij




def Assign_Batch_eval(gt_positions, pred_positions, gt_valid, thres=1.0):
    #Update: note that this cannot be used for training since only a subset is used!
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()


    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            s_valid = gt_valid[batch_idx, 0]
            total_valid += s_valid
            s_pred_positions = pred_positions[batch_idx, :s_valid, :]
            row_inds, col_inds, o_ij = Assign_simple(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions, thres)



            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]
                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return target_scores, corresponding_positions, total_valid, total_iou


def Assign_Batch_v2(gt_positions, pred_positions, gt_valid, thres=1.0):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()


    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            total_valid += gt_valid[batch_idx,0]
            s_pred_positions = pred_positions[batch_idx]
            # s_pred_scores = pred_scores[batch_idx]
            # s_pred_positions_nms, _ = PropUtils.non_maxima_supression(s_pred_positions, s_pred_scores, overlap=0.99)
            # s_pred_positions_nms = s_pred_positions_nms[:gt_valid[batch_idx, 0]]

            #TODO: check if also need to keep topN
            row_inds, col_inds, o_ij = Assign(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions, thres)



            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]
                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return target_scores, corresponding_positions, total_valid, total_iou


def Assign_Batch_v3(gt_positions, pred_positions, gt_valid, thres=1.0, assign_type='full'):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()
    assgin_wegiths = {'full': [None, None], 'rank':[0, None], 'greedy': [0, 0]}

    this_assign_weights = assgin_wegiths[assign_type]

    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            total_valid += gt_valid[batch_idx,0]
            s_pred_positions = pred_positions[batch_idx]
            # s_pred_scores = pred_scores[batch_idx]
            # s_pred_positions_nms, _ = PropUtils.non_maxima_supression(s_pred_positions, s_pred_scores, overlap=0.99)
            # s_pred_positions_nms = s_pred_positions_nms[:gt_valid[batch_idx, 0]]

            #TODO: check if also need to keep topN
            row_inds, col_inds, o_ij = Assign_various(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions, thres, r_max=this_assign_weights[0], d_max=this_assign_weights[1])

            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]
                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return target_scores, corresponding_positions, total_valid, total_iou



def Assign_Batch_v2_regression(gt_positions, pred_positions, gt_valid, gt_overlaps, thres=1.0):
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


    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    target_overlaps = np.ones(pred_positions.shape[0:2])

    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            total_valid += gt_valid[batch_idx,0]
            s_pred_positions = pred_positions[batch_idx]
            # s_pred_scores = pred_scores[batch_idx]
            # s_pred_positions_nms, _ = PropUtils.non_maxima_supression(s_pred_positions, s_pred_scores, overlap=0.99)
            # s_pred_positions_nms = s_pred_positions_nms[:gt_valid[batch_idx, 0]]

            #TODO: check if also need to keep topN
            row_inds, col_inds, o_ij = Assign(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions, thres)



            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]
                        target_overlaps[batch_idx, col_ind]=gt_overlaps[batch_idx, row_ind]
                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return target_scores, target_overlaps, corresponding_positions, total_valid, total_iou



def totalMatch_Batch(gt_positions, pred_positions, gt_valid):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()


    batch_size = gt_positions.shape[0]
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            total_valid += gt_valid[batch_idx,0]
            s_pred_positions = pred_positions[batch_idx]
            #FIXME: you can do NMS on this later
            # s_pred_scores = pred_scores[batch_idx]
            # s_pred_positions_nms, _ = PropUtils.non_maxima_supression(s_pred_positions, s_pred_scores, overlap=0.99)
            # s_pred_positions_nms = s_pred_positions_nms[:gt_valid[batch_idx, 0]]

            #TODO: check if also need to keep topN
            row_inds, col_inds, o_ij = IOUMatch(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions)



            for row_ind, col_ind in zip(row_inds, col_inds):

                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return  total_valid, total_iou


def totalMatch(gt_positions, pred_positions, gt_valid=None):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    # if not isinstance(gt_valid, (np.ndarray, np.generic)):
    #     gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()
    if gt_valid is None:
        gt_valid = pred_positions.shape[0]

    # batch_size = gt_positions.shape[0]
    total_valid = 0
    total_iou = 0
    # for batch_idx in range(batch_size):
    if gt_valid >0:
        total_valid += gt_valid
        s_pred_positions = pred_positions
        #FIXME: you can do NMS on this later
        # s_pred_scores = pred_scores[batch_idx]
        # s_pred_positions_nms, _ = PropUtils.non_maxima_supression(s_pred_positions, s_pred_scores, overlap=0.99)
        # s_pred_positions_nms = s_pred_positions_nms[:gt_valid[batch_idx, 0]]

        #TODO: check if also need to keep topN
        row_inds, col_inds, o_ij = IOUMatch(gt_positions[:int(gt_valid),:], s_pred_positions)



        for row_ind, col_ind in zip(row_inds, col_inds):

                    total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[row_ind])


    return  total_valid, total_iou


def totalMatch_Batch_extra(gt_positions, pred_positions, gt_valid, allowed_extra=None):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()
    # if not isinstance(pred_scores, (np.ndarray, np.generic)):
    #     pred_scores = pred_scores.data.cpu().numpy()


    batch_size = gt_positions.shape[0]
    n_pred_per_batch = gt_positions.shape[1]
    total_valid = 0
    total_iou = 0
    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            total_valid += gt_valid[batch_idx,0]
            s_pred_positions = pred_positions[batch_idx]
            if allowed_extra is not None:
                allowed_indexes = gt_valid[batch_idx, 0]+allowed_extra
                if allowed_indexes < n_pred_per_batch:
                    s_pred_positions = s_pred_positions[:allowed_indexes]

            row_inds, col_inds, o_ij = IOUMatch(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], s_pred_positions)

            for row_ind, col_ind in zip(row_inds, col_inds):

                        total_iou += compute_single_iou(s_pred_positions[col_ind], gt_positions[batch_idx, row_ind])


    return  total_valid, total_iou



def Assign_Batch_MClass(gt_positions, gt_classes, pred_positions, gt_valid, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(gt_classes, (np.ndarray, np.generic)):
        gt_classes = gt_classes.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    if not isinstance(gt_valid, (np.ndarray, np.generic)):
        gt_valid = gt_valid.data.cpu().numpy()

    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])

    for batch_idx in range(batch_size):
        if gt_valid[batch_idx]>0:
            row_inds, col_inds, o_ij = Assign(gt_positions[batch_idx, :int(gt_valid[batch_idx]),:], pred_positions[batch_idx], thres)

            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = gt_classes[batch_idx, row_ind]
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]

    return target_scores, corresponding_positions

def Assign_Batch_all_valid(gt_positions, pred_positions, thres=0.5):
    # gt_intevals batch_size X M X 2
    # pred_intevals batch_size X N b X 2
    # for the cost, the smaller, the better
    if not isinstance(gt_positions, (np.ndarray, np.generic)):
        gt_positions = gt_positions.data.cpu().numpy()
    if not isinstance(pred_positions,(np.ndarray, np.generic)):
        pred_positions = pred_positions.data.cpu().numpy()
    # if not isinstance(gt_valid, (np.ndarray, np.generic)):
    #     gt_valid = gt_valid.data.cpu().numpy()

    batch_size = gt_positions.shape[0]
    corresponding_positions = np.zeros_like(pred_positions)
    target_scores = np.zeros(pred_positions.shape[0:2])
    for batch_idx in range(batch_size):
        # if gt_valid[batch_idx]>0:
            row_inds, col_inds, o_ij = Assign(gt_positions[batch_idx], pred_positions[batch_idx], thres)

            for row_ind, col_ind in zip(row_inds, col_inds):
                        target_scores[batch_idx, col_ind] = 1
                        corresponding_positions[batch_idx, col_ind] = gt_positions[batch_idx, row_ind]

    return target_scores, corresponding_positions




if __name__ == '__main__':

    # profit_matrix = np.load('/home/zwei/Dev/NetModules/PtrNet2/debug_invalid_h_profit.npy')
    profit_matrix = np.array([[4, 1, 3], [2, 0, 5]])
    cost_matrix = toCostMatrix(profit_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print("DB")



