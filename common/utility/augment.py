import random
import numpy as np
from easydict import EasyDict as edict

def get_default_augment_config():
    config = edict()
    config.do_aug = True

    config.scale_factor = 0.25
    config.rot_factor = 15
    config.center_factor = 0.10 # 15% relative to the patch size
    config.color_factor = 0.2
    config.do_flip_aug = True

    config.rot_aug_rate = 0.6  #possibility to rot aug
    config.flip_aug_rate = 0.5 #possibility to flip aug

    config.use_color_normalize = True
    config.mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    config.std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    config.multiplier = [0.5, 1., 1.5, 2, 2.5]
    return config


def do_augmentation(aug_config):
    scale = np.clip(np.random.randn(), -0.5, 1.0) * aug_config.scale_factor + 1.0
    rot = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.rot_factor \
        if random.random() <= aug_config.rot_aug_rate else 0
    center = np.abs(np.clip(np.random.randn(2), -1.0, 1.0)) * aug_config.center_factor

    do_flip = aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + aug_config.color_factor
    c_low = 1.0 - aug_config.color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return scale, rot, center, do_flip, color_scale


def get_multiplier(img_size, scale_search, patch_size):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    return [x * patch_size / float(img_size) for x in scale_search]