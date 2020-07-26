import cv2
import numpy as np
from common.utility.image_processing_cv import get_single_patch_sample, get_single_patch_sample_inference
from common_pytorch.dp import data_provider

class facade_Dataset(data_provider):
    def __init__(self, db_list, is_train, patch_width, patch_height, label_func, aug_config, label_config):
        super(facade_Dataset, self).__init__(db_list, is_train, patch_width, patch_height, label_func, aug_config, label_config)

    def __getitem__(self, index):
        the_db = self.db[index]
        windows = [the_db['left_top'].copy(), the_db['left_bottom'].copy(),
                  the_db['right_bottom'].copy(), the_db['right_top'].copy()]
        if self.useCenterNet:
            windows += [the_db['center'].copy()]

        img_patch, label, gt_loc = get_single_patch_sample(the_db['image'], windows, self.flip_pair,
                                    self.patch_width, self.patch_height, self.mean, self.std,
                                    self.do_augment, self.aug_config, self.label_func, self.loss_config)

        return img_patch.astype(np.float32), label.astype(np.float32), gt_loc.astype(np.float32)

class infer_facade_Dataset:
    def __init__(self, image_list, patch_width, patch_height, aug_config):
        # gather various dataset
        self.db = list()
        for im_path in image_list:
            im = cv2.imread(im_path)
            try:
                height, width, channel = im.shape
            except:
                assert 0, "Mistake loading image:%s" % im_path

            self.db.append({
                'image': im_path,
                'im_width': width,
                'im_height': height
            })

        self.patch_width = patch_width
        self.patch_height = patch_height
        self.mean = aug_config.mean
        self.std = aug_config.std
        self.aug_config = aug_config

    def __getitem__(self, index):
        img_patch = get_single_patch_sample_inference(
            self.db[index]['image'], self.patch_width, self.patch_height, self.mean, self.std, self.aug_config)

        return img_patch.astype(np.float32)

    def __len__(self):
        return len(self.db)