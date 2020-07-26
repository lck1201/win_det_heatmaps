import math
import numpy as np
import logging

import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon

# For Evaluation
def computeIOU(quad1, quad2):
    # quadrangle, order: left-top, left-bottom, right-bottom, right-top
    # the Polygon will construct the shape
    poly1 = Polygon(np.array(quad1)).convex_hull
    poly2 = Polygon(np.array(quad2)).convex_hull

    iou = 0
    if poly1.intersects(poly2):
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            union_area = poly1.area + poly2.area - inter_area  # union area
            if union_area == 0:
                iou = 0
            else:
                iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    return iou


# match windows
def match_gt_and_pred(ground_truth, predict, threshold=0.5):
    '''
    :param ground_truth:
    :param predict:
    :param threshold:
    :return: match list, index of mismatched ground truth, index of mismatched predict
    '''
    match_list = []
    non_match_gt = []
    matched_pred = []

    gt_num = ground_truth.shape[0]
    pred_num = predict.shape[0]

    for gt_idx in range(gt_num):
        gt_window = ground_truth[gt_idx].copy()
        max_iou = 0
        match_pred_index = 0
        for p_idx in range(pred_num):
            if p_idx in matched_pred:  # if this prection has been matched, skip it
                continue

            pred_window = predict[p_idx][:, :2].copy()
            iou = computeIOU(gt_window, pred_window)
            if iou >= max_iou:
                max_iou = iou
                match_pred_index = p_idx

        if max_iou >= threshold:
            match_list.append([gt_idx, match_pred_index])
            matched_pred.append(match_pred_index)
        else:
            non_match_gt.append(gt_idx)

    # find index of non-matched predicts
    non_match_pred = set(range(pred_num)) - set(matched_pred)

    return match_list, non_match_gt, non_match_pred


def apArea(recall, precision):
    """
    Calculate the area under PR curve

    :param recall: array of recall
    :param precision: array of precision

    :return: average precision
    """

    m_precision = np.concatenate(([0.], precision, [0.]))
    m_recall = np.concatenate(([0.], recall, [1.]))

    # replace each precision value with the maximum precision value to the right
    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])

    # show PR curve
    # plt.figure()
    # plt.plot(recall, precision, label='ROC Curve')
    # plt.plot(m_recall, m_precision, label='PR Curve')
    # plt.legend()
    # plt.show()

    # position where value changed
    c = np.where(m_recall[1:] != m_recall[:-1])[0]

    # calculate area under PR curve
    ap = np.sum((m_recall[c + 1] - m_recall[c]) * m_precision[c + 1])

    return ap


def evaluateAP(ground_truth, predict, threshold):
    """
    Calculate the AP of the entire dataset

    :param ground_truth: {img_id1:{{'position': 4x2 array, 'is_matched': 0 or 1}, {...}, ...}, img_id2:{...}, ...}
    :param predict:      [{'position':4x2 array, 'img_id': image Id, 'confident': confident}, {...}, ...]
    :param threshold:    iou threshold

    :return: average precision
    """

    # sort predict by confidence
    sorted_pre = sorted(predict, key=lambda e: e.__getitem__('score'), reverse=True)

    num_pre = len(predict)
    num_gt = sum([len(ground_truth[k]) for k in ground_truth])

    tp = np.zeros(num_pre)
    fp = np.ones(num_pre)

    for img_id in ground_truth.keys():
        for i in range(len(ground_truth[img_id])):
            win_gt = ground_truth[img_id][i]['position']
            max_iou = float('-inf')
            match_index = -1

            for j in range(num_pre):
                if img_id != sorted_pre[j]['img_id']:
                    continue
                win_pred = sorted_pre[j]['position']
                iou = computeIOU(win_pred, win_gt)
                if iou > max_iou:
                    max_iou = iou
                    match_index = j

            if max_iou >= threshold:
                assert ground_truth[img_id][i]['is_matched'] == 0
                tp[match_index] = 1
                fp[match_index] = 0
                ground_truth[img_id][i]['is_matched'] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    precision = tp / (tp + fp)
    recall = tp / float(num_gt)

    ap = apArea(recall, precision)

    return ap, tp[-1] / float(num_gt), tp[-1] / float(num_pre)


def evaluateAPV2(ground_truth, predict, threshold):
    """
    Calculate the AP of the entire dataset

    :param ground_truth: {img_id1:{{'position': 4x2 array, 'is_matched': 0 or 1}, {...}, ...}, img_id2:{...}, ...}
    :param predict:      [{'position':4x2 array, 'img_id': image Id, 'confident': confident}, {...}, ...]
    :param threshold:    iou threshold

    :return: average precision
    """

    # sort predict by confidence
    sorted_pre = sorted(predict, key=lambda e: e.__getitem__('score'), reverse=True)

    num_pre = len(predict)
    num_gt = sum([len(ground_truth[k]) for k in ground_truth])

    tp = np.zeros(num_pre)
    fp = np.zeros(num_pre)

    # get TPs and FPs
    for i in range(num_pre):
        win_pred = sorted_pre[i]['position']
        img_id = sorted_pre[i]['img_id']

        max_iou = float('-inf')
        match_index = 0
        for j in range(len(ground_truth[img_id])):
            win_gt = ground_truth[img_id][j]['position']
            iou = computeIOU(win_pred, win_gt)
            if iou > max_iou:
                max_iou = iou
                match_index = j

        if max_iou >= threshold:
            if ground_truth[img_id][match_index]['is_matched'] == 0:
                tp[i] = 1
                ground_truth[img_id][match_index]['is_matched'] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    precision = tp / (tp + fp)
    recall = tp / float(num_gt)

    ap = apArea(recall, precision)

    return ap, tp[-1] / float(num_gt), tp[-1] / float(num_pre)


def getAp(ground_truth, predict, fullEval=False):
    """
    Calculate AP at IOU=.50:.05:.95, AP at IOU=.50, AP at IOU=.75

    :param ground_truth: {img_id1:{{'position': 4x2 array, 'is_matched': 0 or 1}, {...}, ...}, img_id2:{...}, ...}
    :param predict:      [{'position':4x2 array, 'img_id': image Id, 'confident': confident}, {...}, ...]

    :return: AP, AP at IOU=.50, AP at IOU=.75
    """

    is_match = {'is_matched': 0}

    ap_050_095 = 0.
    ap_050 = 0.
    ap_075 = 0.

    prec_050_095 = 0.
    prec_050 = 0.
    prec_075 = 0.

    recall_050_095 = 0.
    recall_050 = 0.
    recall_075 = 0.

    if fullEval:
        for i in np.arange(0.50, 1.0, 0.05):
            for key in ground_truth:
                for win_idx in range(len(ground_truth[key])):
                    ground_truth[key][win_idx].update(is_match)  # reset 'is_matched' for all windows

            ap, recall, precision = evaluateAP(ground_truth, predict, threshold=i)

            if math.isclose(round(i, 2), 0.5):
                ap_050 = ap
                prec_050 = precision
                recall_050 = recall
            if math.isclose(round(i, 2), 0.75):
                ap_075 = ap
                prec_075 = precision
                recall_075 = recall

            ap_050_095 += ap
            prec_050_095 += precision
            recall_050_095 += recall

            logging.info("threshold:%.2f"%i + " precsion:%.2f"%(precision*100) + " recall:%.2f"%(recall*100))
    else:
        ap_050, recall_050, prec_050 = evaluateAP(ground_truth, predict, threshold=0.5)

    ap_050_095 = ap_050_095 / 10
    prec_050_095 = prec_050_095 / 10
    recall_050_095 = recall_050_095 / 10

    return [ap_050_095, ap_050, ap_075], \
           [prec_050_095, prec_050, prec_075], \
           [recall_050_095, recall_050, recall_075]