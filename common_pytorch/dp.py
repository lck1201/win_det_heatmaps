import torch.utils.data as data
from common_pytorch.dataset.facade import flip_pairs

class data_provider(data.Dataset):
    def __init__(self, db_list, is_train, patch_width, patch_height, label_func, aug_config, loss_config):
        # gather various dataset
        self.db = list()
        for db in db_list:
            self.db.extend(db.gt_db(is_train))

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.mean = aug_config.mean
        self.std = aug_config.std
        self.flip_pair = flip_pairs

        self.label_func = label_func
        self.aug_config = aug_config
        self.loss_config = loss_config
        self.useCenterNet = loss_config.useCenterNet

        if self.is_train:
            self.do_augment = aug_config.do_aug
        else:
            self.do_augment = False

    def __len__(self):
        return len(self.db)