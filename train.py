import os
import copy
import time
import torch
import shutil
import pprint
from torch.utils.data import DataLoader

# define project dependency
import _init_paths

# common
from common.speedometer import Speedometer
from common.utility.logger import create_logger
from common.utility.visualization import plot_LearningCurve

# project dependence
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, \
    s_args, s_config, s_config_file
from common_pytorch.optimizer import get_optimizer
from common_pytorch.io_pytorch import save_all_model
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from common_pytorch.net_modules import trainNet, validNet, evalNet

from blocks.resnet_pose import get_default_network_config, init_pose_net
from loss.heatmap import get_default_loss_config, get_loss_func, get_label_func, get_merge_func

from core.loader import facade_Dataset
exec('from common_pytorch.blocks.' + s_config.pytorch.block + ' import get_default_network_config, get_pose_net, init_pose_net')

def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)
    et = config.dataset.eval_target

    # create log and path
    final_output_path, final_log_path, logger = create_logger(s_config_file, config.dataset.benchmark[et],
                                                              config.pytorch.output_path, config.pytorch.log_path)
    logger.info('Train config:{}\n'.format(pprint.pformat(config)))
    shutil.copy2(s_args.cfg, final_output_path)

    # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # label, loss, metric and result
    logger.info("Defining lable, loss, metric and result")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    merge_hm_flip_func, merge_tag_flip_func = get_merge_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)  # advanced parallel

    # dataset, basic imdb
    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    logger.info("Creating dataset")
    train_imdbs = []
    for bmk_name in ['JSON', 'XML']:
        train_imdbs += [facade(bmk_name, 'TRAIN', config.dataset.path)]
    test_imdbs = [facade('TEST', 'TEST', config.dataset.path)]

    # basic data_loader unit
    dataset_train = facade_Dataset(train_imdbs, True, config.train.patch_width, config.train.patch_height,
                                  label_func, config.aug, config.loss)

    dataset_test = facade_Dataset(test_imdbs, False, config.train.patch_width, config.train.patch_height,
                                  label_func, config.aug, config.loss)

    train_data_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                   num_workers=config.dataiter.threads)
    valid_data_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                                   num_workers=config.dataiter.threads)

    # prepare network
    logger.info("Creating network")
    net = get_pose_net(config.network, config.loss.ae_feat_dim,
                       num_corners if not config.loss.useCenterNet else num_corners + 1)
    init_pose_net(net, config.network)
    net = DataParallelModel(net).cuda() # advanced parallel
    model_prefix = os.path.join(final_output_path, config.train.model_prefix)
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # Optimizer
    logger.info("Creating optimizer")
    optimizer, scheduler = get_optimizer(config.optimizer, net)

    # resume from model
    train_loss = []
    valid_loss = []
    latest_model = '{}_latest.pth.tar'.format(model_prefix)
    if s_args.autoresume and os.path.exists(latest_model):
        model_path = latest_model if os.path.exists(latest_model) else s_args.model
        assert os.path.exists(model_path), 'Cannot find model!'
        logger.info('Load checkpoint from {}'.format(model_path))

        # load state from ckpt
        ckpt = torch.load(model_path)
        config.train.begin_epoch = ckpt['epoch'] + 1
        net.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        train_loss.extend(ckpt['train_loss'])
        valid_loss.extend(ckpt['valid_loss'])

        assert config.train.begin_epoch >= 2, 'resume error. begin_epoch should no less than 2'
        logger.info('continue training the {0}th epoch, init from the {1}th epoch'.
                    format(config.train.begin_epoch,config.train.begin_epoch - 1))

    # train and valid
    logger.info("Train DB size: {}; Valid DB size: {}.".format(int(len(dataset_train)), int(len(dataset_test))))
    for epoch in range(config.train.begin_epoch, config.train.end_epoch + 1):
        logger.info("\nWorking on {}/{} epoch || LearningRate:{} ".format(epoch, config.train.end_epoch, scheduler.get_lr()[0]))
        speedometer = Speedometer(train_data_loader.batch_size, config.pytorch.frequent, auto_reset=False)

        beginT = time.time()
        tloss = trainNet(epoch, train_data_loader, net, optimizer, config.loss, loss_func, speedometer)
        endt1 = time.time() - beginT

        beginT = time.time()
        heatmaps, tagmaps, vloss = validNet(valid_data_loader, net, loss_func, merge_hm_flip_func,
                                            merge_tag_flip_func, devices, flip_pairs, flip_test=False)
        endt2 = time.time() - beginT

        beginT = time.time()
        if epoch > config.train.end_epoch - 3: #only eval late model, because evaluation takes too much time
            evalNet(epoch, heatmaps, tagmaps, valid_data_loader, config.loss, config.test,
                    config.train.patch_width, config.train.patch_height, final_output_path)
        endt3 = time.time() - beginT

        logger.info('This Epoch Train %.1fs, Valid %.1fs, Eval %.1fs ' % (endt1, endt2, endt3))
        logger.info('Train Loss:%.4f, Valid Loss:%.4f' % (tloss, vloss))

        train_loss.append(tloss)
        valid_loss.append(vloss)
        scheduler.step()

        # save model
        state = {
            'epoch': epoch,
            'network': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }
        save_all_model(epoch, model_prefix, state, vloss, config, logger)
        plot_LearningCurve(train_loss, valid_loss, final_log_path, "LearningCurve")

if __name__ == "__main__":
    main()
