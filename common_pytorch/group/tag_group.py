# Functions for grouping tags
from torch import nn
import numpy as np
from munkres import Munkres

from common_pytorch.common_loss.heatmap_label import get_coords_from_heatmaps_with_NMS

def convert_to_dict(window, rectify=False):
    lt, lb, rb, rt = window
    score = np.mean([lt[2], lb[2], rb[2], rt[2]])
    if rectify:
        x1 = min([lt[0], lb[0]])
        y1 = min([lt[1], rt[1]])
        x2 = max([rb[0], rt[0]])
        y2 = max([lb[1], rb[1]])

        lt = np.array([x1, y1], dtype=lt.dtype)
        lb = np.array([x1, y2], dtype=lt.dtype)
        rb = np.array([x2, y2], dtype=lt.dtype)
        rt = np.array([x2, y1], dtype=lt.dtype)

    return {'position': np.array([lt, lb, rb, rt]), 'score': score}

class Params:
    def __init__(self):
        self.num_parts = 4
        self.detection_threshold = 0.2
        self.tag_threshold = 1.0
        self.partOrder = [i - 1 for i in [1, 2, 3, 4]]
        self.max_num_corner = 120
        self.use_detection_val = False
        self.ignore_too_much = False

def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:, :, :, None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis=0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index(np.argmax(tmp2), tmp.shape)
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy + 1, det.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
            y += 0.25
        else:
            y -= 0.25

        if tmp[min(xx + 1, det.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
            x += 0.25
        else:
            x -= 0.25

        x, y = np.array([y, x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(4):
            if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1

    return keypoints

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

def match_by_tag(inp, params):
    tag_k, loc_k, val_k = inp
    assert type(params) is Params
    default_ = np.zeros((params.num_parts, 3 + tag_k[0].shape[1]))  # locy,locx,score,tag_dimension=2 if flip test

    dic = {}
    dic2 = {}
    for i in range(params.num_parts):
        ptIdx = params.partOrder[i]

        tags = tag_k[ptIdx]
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]
        if i == 0 or len(dic) == 0:
            for tag, joint in zip(tags, joints):
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint
                dic2[tag[0]] = [tag] #flip-test or maybe multi-dimension FEAT
        else:
            actualTags_key = list(dic.keys())[:params.max_num_corner]
            actualTags = [np.mean(dic2[k], axis=0) for k in actualTags_key]

            if params.ignore_too_much and len(actualTags) == params.max_num_corner:
                continue
            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :]) ** 2).mean(axis=2) ** 0.5
            if diff.shape[0] == 0:
                continue

            diff2 = np.copy(diff)

            if params.use_detection_val:
                diff = np.round(diff) * 100 - joints[:, 2:3]

            if diff.shape[0] > diff.shape[1]:
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10), axis=1)

            pairs = py_max_match(-diff)  ##get minimal matching
            for row, col in pairs:
                if row < diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < params.tag_threshold:
                    dic[actualTags_key[col]][ptIdx] = joints[row]
                    dic2[actualTags_key[col]].append(tags[row])
                else:
                    key = tags[row][0]
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row]
                    dic2[key] = [tags[row]]

    # delete matching with location(0, 0)
    to_remove = list()
    for k in dic:
        for loc in dic[k]:
            if np.isclose(loc[0], 0) and np.isclose(loc[1], 0):
                to_remove.append(k)
                break

    for k in to_remove:
        del dic[k]

    if len(dic) == 0:
        ans = np.zeros((1, 4, 3))
    else:
        ans = np.array(list(dic.values()))

    return ans.astype(np.float32)

class HeatmapParser():
    def __init__(self, loss_config, useCenter, centerThreshold, imdb_list=None):
        self.param = Params()
        self.pool = nn.MaxPool2d(3, 1, 1)

        self.imdb_list = imdb_list
        self.useCenter = useCenter
        self.param.tag_threshold = loss_config.ae_expect_dist
        self.centerThreshold = centerThreshold
        self.ae_feat_dim = loss_config.ae_feat_dim

        print("Param Tag_threshold", self.param.tag_threshold)
        print("Param detection_threshold", self.param.detection_threshold)

    def match(self, tag_k, loc_k, val_k):
        return match_by_tag([tag_k, loc_k, val_k], self.param)

    def calc(self, det, tag, idx=0):
        '''
        Get topK keypoint score/tag/location
        '''
        coords_in_patch_with_score_id = get_coords_from_heatmaps_with_NMS(det[:4])

        val_k = [c_pts[:, 2, np.newaxis] for c_pts in coords_in_patch_with_score_id]
        ind_k = [(c_pts[:, 0:2] + 0.5).astype(int) for c_pts in coords_in_patch_with_score_id]

        # NOTE:exprimental, each corner one step towards center
        indForExtractTag = [item.copy() for item in ind_k]
        indForExtractTag[0][:, 0] += 3
        indForExtractTag[0][:, 1] += 3

        indForExtractTag[1][:, 0] += 3
        indForExtractTag[1][:, 1] -= 3

        indForExtractTag[2][:, 0] -= 3
        indForExtractTag[2][:, 1] -= 3

        indForExtractTag[3][:, 0] -= 3
        indForExtractTag[3][:, 1] += 3
        patch_size = 384
        for i in range(4):
            mask = (indForExtractTag[i][:, 0:2] >= patch_size)
            indForExtractTag[i][:, 0:2][mask] = patch_size-1
        # NOTE:exprimental, each corner one step towards center

        n, patch_h, patch_w = tag.shape
        tag = tag.reshape((n//self.ae_feat_dim, self.ae_feat_dim, patch_h, patch_w))
        tag_k = [tag[idx, :, loc[:, 1], loc[:, 0]] for idx, loc in enumerate(indForExtractTag)]

        if tag.shape[0] == 8: # flip-test
            tag_k_flip = [tag[idx + 4, :, loc[:, 1], loc[:, 0]] for idx, loc in enumerate(indForExtractTag)]
            tag_k = [np.concatenate((tg, tg_fp), axis=1) for tg, tg_fp in zip(tag_k, tag_k_flip)]

        return {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}

    def rectify(self, windows):
        for i in range(len(windows)):
            lt, lb, rb, rt = windows[i]

            x1 = min([lt[0], lb[0]])
            y1 = min([lt[1], rt[1]])
            x2 = max([rb[0], rt[0]])
            y2 = max([lb[1], rb[1]])

            lt = np.array([x1, y1], dtype=lt.dtype)
            lb = np.array([x1, y2], dtype=lt.dtype)
            rb = np.array([x2, y2], dtype=lt.dtype)
            rt = np.array([x2, y1], dtype=lt.dtype)

            windows[i, 0, 0:2] = lt
            windows[i, 1, 0:2] = lb
            windows[i, 2, 0:2] = rb
            windows[i, 3, 0:2] = rt

        return windows

    def removeInvalidGivenCenter(self, preds, centerMap):
        centerPostition = preds[:, :, :2].mean(axis=1)
        toKeep = list()
        for i in range(len(centerPostition)):
            x, y = centerPostition[i]
            if centerMap[int(y+0.5)][int(x+0.5)] >= self.centerThreshold:
                toKeep.append(i)

        return preds[toKeep]

    def parse(self, det, tag, idx, ratio, rectify=False):
        re = self.calc(det, tag, idx)
        ans = self.match(**re)
        if rectify:
            ans = self.rectify(ans)
        if self.useCenter:
            ans = self.removeInvalidGivenCenter(ans, det[4])

        ans = ans[:, :, :3]
        ans[:, :, 0:2] *= ratio

        return ans

def group_corners_on_tags(idx, parser, dets, tags, patch_width, patch_height, im_width, im_height, rectify,
                          winScoreThres):
    '''
    :param preds_in_patch:
    :param tagmap:
    :param rectify:
    :return:
    '''
     # rescale ratio from patch size to image size
    ratio = max(im_width, im_height) / patch_height
    ratio = np.array([ratio, ratio])

    grouped = parser.parse(np.float32(dets), np.float32(tags), idx, ratio, rectify)  # shape=(num_of_windows, 4, 4)

    linked_window = list() # convert format & do rectify
    for n_w in range(len(grouped)):
        score = np.mean(grouped[n_w][:,2])
        if score < winScoreThres: #skip low-score windows
            continue
        linked_window.append(convert_to_dict(grouped[n_w]))

    return linked_window