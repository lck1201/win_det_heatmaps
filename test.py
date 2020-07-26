import os
import copy
import time
import torch
import logging
import pprint

from torch.utils.data import DataLoader

# define project dependency
import _init_paths

# project dependence
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, \
    s_args, s_config, s_config_file
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from common_pytorch.net_modules import validNet, evalNet

from blocks.resnet_pose import get_default_network_config
from loss.heatmap import get_default_loss_config, get_loss_func, get_label_func, get_merge_func

from core.loader import facade_Dataset
exec('from common_pytorch.blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')

def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)
    et = config.dataset.eval_target

    # create log and path
    output_path = os.path.dirname(s_config_file)
    log_name = os.path.basename(s_args.model)
    logging.basicConfig(filename=os.path.join(output_path, '{}_test.log'.format(log_name)),
                        format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info('Test config:{}\n'.format(pprint.pformat(config)))

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # label, loss, metric and result
    logger.info("Defining lable, loss, metric and result")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)
    merge_hm_flip_func, merge_tag_flip_func = get_merge_func(config.loss)

    # dataset, basic imdb
    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    logger.info("Creating dataset")
    valid_imdbs = [facade(config.dataset.benchmark[et], 'valid', config.dataset.path[et])]

    dataset_valid = facade_Dataset(valid_imdbs, False, config.train.patch_width,config.train.patch_height,
                                   label_func, config.aug, config.loss)

    # here disable multi-process num_workers, because limit of GPU server
    valid_data_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size)

    # prepare network
    assert os.path.exists(s_args.model), 'Cannot find model!'
    logger.info("Loading model from %s"%s_args.model)
    net = get_pose_net(config.network,  config.loss.ae_feat_dim,
                       num_corners if not config.loss.useCenterNet else num_corners + 1)
    net = DataParallelModel(net).cuda()
    ckpt = torch.load(s_args.model)  # or other path/to/model
    net.load_state_dict(ckpt['network'])
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # T^est
    logger.info("Test DB size: {}.".format(len(dataset_valid)))
    print("------TestUseCenter:%s, centerT:%.1f, windowT:%.1f ----------"%
          (config.test.useCenter, config.test.centerT, config.test.windowT))

    beginT = time.time()
    heatmaps, tagmaps, vloss = \
        validNet(valid_data_loader, net, loss_func, merge_hm_flip_func, merge_tag_flip_func,
                 devices, flip_pairs, flip_test=True)
    endt1 = time.time() - beginT
    logger.info('Valid Loss:%.4f' % vloss)

    beginT = time.time()
    evalNet(0, heatmaps, tagmaps, valid_data_loader, config.loss, config.test,
            config.train.patch_width, config.train.patch_height, output_path)
    endt2 = time.time() - beginT
    logger.info('This Epoch Valid %.3fs, Eval %.3fs ' % (endt1, endt2))

if __name__ == "__main__":
    main()
