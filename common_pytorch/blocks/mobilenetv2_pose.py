import os
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.mobilenet import model_urls
from easydict import EasyDict as edict

from common_pytorch.base_modules.mobilenetv2 import MobileNetV2_Backbone
from common_pytorch.base_modules.deconv_head import DeconvHead
from common_pytorch.base_modules.architecture import PoseNet_1branch, PoseNet_2branch

def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.name = 'mobilenet_v2'
    # default head setting
    config.num_deconv_layers = 3
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    # output
    config.depth_dim = 1
    #AE
    config.head_branch = 2

    return config

def init_pose_net(pose_net, cfg):
    if cfg.from_model_zoo:
        org_net = model_zoo.load_url(model_urls[cfg.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_net.pop('classifier.1.weight', None)
        org_net.pop('classifier.1.bias', None)
        pose_net.backbone.load_state_dict(org_net)
        print("Init Network from model zoo")
    else:
        if os.path.exists(cfg.pretrained):
            model = torch.load(cfg.pretrained)
            pose_net.load_state_dict(model['network'])
            print("Init Network from pretrained", cfg.pretrained)

def get_pose_net(network_cfg, ae_feat_dim, num_point_types):
    backbone_net = MobileNetV2_Backbone()

    # one branch, double output channel
    out_channel = num_point_types * 2
    if network_cfg.head_branch == 2:
        out_channel = num_point_types
    backbone_output_dim = 1280
    heatmap_head = DeconvHead(backbone_output_dim, network_cfg.num_deconv_layers, network_cfg.num_deconv_filters,
                              network_cfg.num_deconv_kernel, network_cfg.final_conv_kernel, out_channel, network_cfg.depth_dim)
    # NOTE: to specify 4 to avoid center tag
    tagmap_head = DeconvHead(backbone_output_dim, network_cfg.num_deconv_layers, network_cfg.num_deconv_filters,
                             network_cfg.num_deconv_kernel, network_cfg.final_conv_kernel, 4, ae_feat_dim)

    if network_cfg.head_branch == 1:
        return PoseNet_1branch(backbone_net, heatmap_head)
    elif network_cfg.head_branch == 2:
        return PoseNet_2branch(backbone_net, heatmap_head, tagmap_head)
    else:
        assert 0