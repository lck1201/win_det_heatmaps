import math
from easydict import EasyDict as edict
import torch
import torch.nn as nn

from common.utility.image_processing_cv import flip
from common_pytorch.common_loss.heatmap_label import generate_gaussian_heatmap_label
from common_pytorch.common_loss.weighted_mse import weighted_mse_loss, weighted_l1_loss, weighted_ae_loss

# config
def get_default_loss_config():
    config = edict()
    # gaussian
    config.heatmap_type = 'gaussian'
    config.loss_type = 'L2'
    config.sigma = 2
    config.feat_stride = 4

    # ae
    config.useAE = True
    config.ae_weight = 1.0
    config.ae_expect_dist = 12.0
    config.ae_feat_dim = 1

    # centerNet
    config.useCenterNet = False

    return config
# config

# define loss
def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def heatmap_loss(loss_type, size_average):
    if loss_type == 'L1':
        return weighted_l1_loss
    elif loss_type == 'L2':
        return weighted_mse_loss
    elif loss_type == 'MSE':
        return nn.MSELoss(size_average=size_average)
    else:
        assert 0, 'Wrong loss type, current %s' % loss_type

class GaussianAEHeatmapLoss(nn.Module):
    def __init__(self, loss_config, size_average=True):
        super(GaussianAEHeatmapLoss, self).__init__()
        self.size_average = size_average
        self.loss_type = loss_config.loss_type.upper()
        self.ae_loss_weight = loss_config.ae_weight
        self.ae_expect_dist = loss_config.ae_expect_dist
        self.ae_feat_dim = loss_config.ae_feat_dim
        self.useAE = loss_config.useAE

        self.hm_loss = heatmap_loss(self.loss_type, size_average)

    def forward(self, heatmaps, tagmaps, gt_heatmaps, gt_loc):
        keypoints = gt_loc.type(dtype=torch.int64)
        _assert_no_grad(gt_heatmaps)
        _assert_no_grad(keypoints)
        batchsize, num_points, _, _ = gt_heatmaps.shape

        gt_heatmaps = gt_heatmaps.reshape(batchsize, num_points, -1)
        heatmaps = heatmaps.reshape(batchsize, num_points, -1)
        hm_loss = self.hm_loss(heatmaps, gt_heatmaps)

        if self.useAE and not math.isclose(self.ae_loss_weight, 0):
            tagmaps = tagmaps.reshape(batchsize, -1)
            ae_loss = weighted_ae_loss(tagmaps, keypoints, self.ae_expect_dist, self.ae_feat_dim)
            return hm_loss + self.ae_loss_weight * ae_loss

        return hm_loss

# define label
def generate_heatmap_label(config, patch_width, patch_height, window):
    type = config.heatmap_type
    sigma = config.sigma
    feat_stride = config.feat_stride

    if 'gaussian' in type:
        return generate_gaussian_heatmap_label(feat_stride, patch_width, patch_height, window, sigma)
    else:
        assert 0, 'Unknown heatmap type {0}'.format(type)

# define flip merge
def merge_hm_flip_func(orgin, pFliped, flip_pair):
    output_flip = flip(pFliped, dims=3)

    for pair in flip_pair:
        tmp = torch.zeros(output_flip[:, pair[0], :, :].shape)
        tmp.copy_(output_flip[:, pair[0], :, :])
        output_flip[:, pair[0], :, :].copy_(output_flip[:, pair[1], :, :])
        output_flip[:, pair[1], :, :].copy_(tmp)

    return (orgin + output_flip) * 0.5

def merge_tag_flip_func(orgin, pFliped, flip_pair):
    output_flip = flip(pFliped, dims=3)

    #todo: flip-test for multi-ae-feat-dim
    for pair in flip_pair:
        tmp = torch.zeros(output_flip[:, pair[0]].shape)
        tmp.copy_(output_flip[:, pair[0]])
        output_flip[:, pair[0]].copy_(output_flip[:, pair[1]])
        output_flip[:, pair[1]].copy_(tmp)

    return torch.cat((orgin, pFliped), dim=1)

# API
def get_loss_func(loss_config):
    return GaussianAEHeatmapLoss(loss_config)

def get_label_func(loss_config):
    return generate_heatmap_label

def get_merge_func(loss_config):
    return merge_hm_flip_func, merge_tag_flip_func
# API