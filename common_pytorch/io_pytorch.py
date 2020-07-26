import torch

vloss_min = float('inf')
def save_all_model(epoch, model_prefix, state, vloss, config, logger):
    # global vloss_min

    # if vloss < vloss_min:
    #     vloss_min = vloss
    #     save_lowest_vloss_model(state, model_prefix, logger)

    save_latest_model(state, model_prefix, logger)

    if (epoch % (config.train.end_epoch // 2) == 0) \
            or epoch == config.train.end_epoch:
        save_model(state, model_prefix, logger, epoch)

def save_model(state, model_prefix, logger, epoch):
    file_path = '{}_epoch{}.pth.tar'.format(model_prefix, str(epoch))
    torch.save(state, file_path)
    logger.info("Write Model into {}".format(file_path))

def save_latest_model(state, model_prefix, logger):
    '''
    Save a latest ckpt to automatically resume on Philly
    '''
    file_path = '{}_latest.pth.tar'.format(model_prefix)
    torch.save(state, file_path)
    logger.info('Write Model into {}'.format(file_path))


def save_lowest_vloss_model(state, model_prefix, logger):
    '''
    Save a latest ckpt to automatically resume on Philly
    '''
    file_path = '{}_lowest_vloss.pth.tar'.format(model_prefix)
    torch.save(state, file_path)
    logger.info('Write Model into {}'.format(file_path))

