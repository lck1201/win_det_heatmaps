import torch
import torch.nn as nn

def weighted_mse_loss(pred, target, weights=1, size_average=True):
    out = (pred - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(pred)
    else:
        return out.sum()

def weighted_l1_loss(pred, target, weights=1, size_average=True):
    out = torch.abs(pred - target)
    out = out * weights
    if size_average:
        return out.sum() / len(pred)
    else:
        return out.sum()

def weighted_ae_loss(tagmap, keypoint, expected_dist, ae_feat_dim, size_average=True):
    batch_size, max_num_winow, num_corner, _ = keypoint.size()
    epsilon = 1e-6

    push_loss_list = list()
    pull_loss_list = list()
    for n_b in range(batch_size):
        pull_loss = 0
        push_loss = 0

        # get tagval from tagmap
        s_kpt_loc = keypoint[n_b, :, :, 0:ae_feat_dim].reshape(-1)
        s_kpt_vis = keypoint[n_b, :, :, ae_feat_dim]
        s_kpt_pos = keypoint[n_b][:, :, 2].type(torch.cuda.FloatTensor).mean(dim=1)
        tag_vals = tagmap[n_b, s_kpt_loc].reshape((max_num_winow, num_corner, ae_feat_dim))

        # PULL LOSS
        center_list = list()
        pos_list = list()
        maxPos = float('-inf')
        minPos = float('inf')
        moreValidKpt = True
        for s_win, s_vis, s_w in zip(tag_vals, s_kpt_vis, s_kpt_pos):
            if moreValidKpt:
                validNum = s_vis.sum()
            else:
                break

            if validNum > 0:  # have visible keypoints
                window_tag_mean = s_win[:validNum].mean(dim=0) # N-D feature vector
                if ae_feat_dim== 1:
                    pull_loss += torch.sum(torch.pow(s_win[:validNum] - window_tag_mean, 2))
                else:
                    pull_loss += torch.sum(torch.norm(s_win[:validNum] - window_tag_mean, dim=1))
                center_list.append(window_tag_mean)

                # distance weight
                pos_list.append(s_w)
                if s_w > maxPos:
                    maxPos = s_w
                if s_w < minPos:
                    minPos = s_w
            else:
                moreValidKpt=False

        maxDist = maxPos - minPos + epsilon
        valid_sample_num = len(center_list)

        # PUSH LOSS
        pushWeight = 3.0
        for src_id in range(valid_sample_num):
            for dst_id in range(valid_sample_num):
                if src_id != dst_id:
                    weight = abs(pos_list[src_id] - pos_list[dst_id]) / maxDist * pushWeight + 1.0
                    if ae_feat_dim == 1:
                        dist = expected_dist - torch.abs(center_list[src_id] - center_list[dst_id])
                    else:
                        dist = expected_dist - torch.norm(center_list[src_id] - center_list[dst_id])

                    dist = weight * nn.functional.relu(dist, inplace=True)
                    push_loss += dist

        pull_loss /= valid_sample_num + epsilon
        push_loss /= valid_sample_num * (valid_sample_num - 1) + epsilon

        push_loss_list.append(push_loss)
        pull_loss_list.append(pull_loss)

    if size_average:
        return (sum(push_loss_list) + sum(pull_loss_list)) / batch_size
    else:
        return sum(push_loss_list) + sum(pull_loss_list)