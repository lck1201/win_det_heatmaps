import os
import copy
import time
import glob
import torch
import logging

from torch.utils.data import DataLoader

# define project dependency
import _init_paths

# project dependence
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, \
    s_args, s_config, s_config_file
from common_pytorch.common_loss.balanced_parallel import DataParallelModel
from common_pytorch.net_modules import inferNet

from blocks.resnet_pose import get_default_network_config
from loss.heatmap import get_default_loss_config, get_merge_func

from core.loader import infer_facade_Dataset
exec('from common_pytorch.blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')

def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)

    if not os.path.exists(s_args.infer):
        print("invalid infer path")
        exit(-1)

    # create log and path
    output_path = os.path.join(s_args.infer, "infer_result")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # label, loss, metric and result
    logger.info("Defining result & flip func")
    merge_hm_flip_func, merge_tag_flip_func = get_merge_func(config.loss)

    # dataset, basic imdb
    logger.info("Creating dataset")

    infer_imdbs = glob.glob(s_args.infer + '/*.jpg')
    infer_imdbs += glob.glob(s_args.infer + '/*.png')
    infer_imdbs.sort()

    dataset_infer = infer_facade_Dataset(infer_imdbs, config.train.patch_width, config.train.patch_height, config.aug)

    # here disable multi-process num_workers, because limit of GPU server
    batch_size = len(devices) * config.dataiter.batch_images_per_ctx
    infer_data_loader = DataLoader(dataset = dataset_infer, batch_size = batch_size)

    # prepare network
    assert os.path.exists(s_args.model), 'Cannot find model!'
    logger.info("Loading model from %s"%s_args.model)
    net = get_pose_net(config.network, config.loss.ae_feat_dim,
                       num_corners if not config.loss.useCenterNet else num_corners + 1)
    net = DataParallelModel(net).cuda()
    ckpt = torch.load(s_args.model)  # or other path/to/model
    net.load_state_dict(ckpt['network'])
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # train and valid
    logger.info("Test DB size: {}.".format(len(infer_imdbs)))

    beginT = time.time()
    inferNet(infer_data_loader, net, merge_hm_flip_func, merge_tag_flip_func, flip_pairs,
             config.train.patch_width, config.train.patch_height, config.loss, config.test, output_path)
    endt = time.time() - beginT
    logger.info('Speed: %.3f second per image' % (endt / len(infer_imdbs)))
    logger.info("Save inference results into %s"%output_path)

if __name__ == "__main__":
    main()