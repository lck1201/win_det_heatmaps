import os
import cv2
import torch
import numpy as np

from common.utility.utils import float2int
from common.utility.augment import do_augmentation
from common.utility.visualization import debug_vis

def KeypointsRef(keypoints, output_res):
    max_num_windows = 120

    visible_nodes = np.zeros((max_num_windows, 4, 3)) # pos1, visibility, winPos
    for n_p in range(len(keypoints)):
        tot = 0
        for n_j, pt in enumerate(keypoints[n_p]):
            x, y = pt[0], pt[1]
            if x >= 0 and y >= 0 and x < output_res and y < output_res:
                visible_nodes[n_p][tot] = (n_j * output_res * output_res + y * output_res + x,
                                           1, x + y)  # encode heatmap index in value
                tot += 1
    return visible_nodes

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)

def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def fliplr_label(window, im_width, matched_parts):
    """
    flip coords
    window: [lt, lb, rb, rt]
    width: image width
    matched_parts: list of pairs
    """
    for idx in range(len(window)):
        corner = window[idx]
        # Flip horizontal
        corner[:, 0] = im_width - corner[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        window[pair[0]], window[pair[1]] = window[pair[1]].copy(), window[pair[0]].copy()

    return window

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_aug_on_center(c_x, c_y, src_width, src_height, dst_width, dst_height, aug_param, inv=False):
    scaleAug, rotAug, centerAug = aug_param

    # augment size with scale
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rotAug / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    if src_h > src_w:
        dst_w = 1.0 * dst_height * src_w / src_h
        dst_h = dst_height
    else:
        dst_w = dst_width
        dst_h = 1.0 * dst_width * src_h / src_w

    center_aug = centerAug * np.array([dst_width, dst_height])
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32) + center_aug
    dst_downdir = np.array([0, dst_h * 0.5 * scaleAug], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5 * scaleAug, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def gen_trans_from_aug_on_origin(src_width, src_height, dst_width, dst_height, aug_param, inv=False):
    scaleAug, rotAug, centerAug = aug_param

    # augment size with scale
    src_w = src_width
    src_h = src_height

    # augment rotation
    rot_rad = np.pi * rotAug / 180
    src_downdir = rotate_2d(np.array([0, src_h], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w, 0], dtype=np.float32), rot_rad)

    if src_h > src_w:
        dst_w = src_w * dst_height / src_h
        dst_h = dst_height
    else:
        dst_w = dst_width
        dst_h = src_h * dst_width / src_w

    center_aug = centerAug * np.array([dst_width, dst_height])
    dst_center = np.zeros(2) + center_aug
    dst_downdir = np.array([0, dst_h * scaleAug], dtype=np.float32)
    dst_rightdir = np.array([dst_w * scaleAug, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[1, :] = src_downdir
    src[2, :] = src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def generate_patch_image_cv(img, img_width, img_height, patch_width, patch_height, do_flip, aug_param):
    if do_flip:
        img = img[:, ::-1, :]

    # not to modify original function to much, just pass into center point
    cx = img_width // 2
    cy = img_height // 2

    # trans = gen_trans_from_aug_on_center(cx, cy, img_width, img_height, patch_width, patch_height, aug_param)
    trans = gen_trans_from_aug_on_origin(img_width, img_height, patch_width, patch_height, aug_param)
    img_patch = cv2.warpAffine(img, trans, float2int((patch_width, patch_height)), flags=cv2.INTER_CUBIC)

    return img_patch, trans

def convert_cvimg_to_tensor(cvimg):
    # from h,w,c(OpenCV) to c,h,w
    tensor = cvimg.copy()
    tensor = np.transpose(tensor, (2, 0, 1))
    # from BGR(OpenCV) to RGB
    tensor = tensor[::-1, :, :]
    # from int to float
    tensor = tensor.astype(np.float32)
    return tensor

def coord_from_patch_to_image(coords_in_patch, img_width, img_height, patch_size):
    coords_in_img = list()

    ratio = max(img_width, img_height) / patch_size
    ratio = np.array((ratio, ratio))

    for n_c in range(len(coords_in_patch)):
        # in the early epoch of training
        # heatmap may not give valid coords, so
        if coords_in_patch[n_c].size == 0:
            coords_in_img.append(np.zeros((0, 4)))
            continue
        rescale_coords = coords_in_patch[n_c].copy() # include extra info(score, id)
        rescale_coords[:, 0:2] *= ratio
        coords_in_img.append(rescale_coords)

    return coords_in_img

def get_single_patch_sample(img_path, windows, flip_pairs, patch_width, patch_height, mean, std,
                            do_augment, aug_config, label_func, label_config):
    # 1. load image
    cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)

    img_height, img_width, img_channels = cvimg.shape #original shape

    # 3. get augmentation params
    if do_augment:
        scale, rot, center, do_flip, color_scale = do_augmentation(aug_config)
    else:
        scale, rot, center, do_flip, color_scale = 1.0, 0, np.zeros(2), False, [1.0, 1.0, 1.0]

    # 4. generate image patch
    aug_param = [scale, rot, center]
    img_patch_cv, trans = generate_patch_image_cv(cvimg.copy(), img_width, img_height,
                                                  patch_width, patch_height, do_flip, aug_param)
    img_patch_tensor = convert_cvimg_to_tensor(img_patch_cv)

    # apply normalization
    for n_c in range(img_channels):
        img_patch_tensor[n_c, :, :] = np.clip(img_patch_tensor[n_c, :, :] * color_scale[n_c], 0, 255)
        if aug_config.use_color_normalize and mean is not None and std is not None:
            img_patch_tensor[n_c, :, :] = (img_patch_tensor[n_c, :, :] - mean[n_c]) / std[n_c]

    # 5. generate patch joint ground truth,flip joints
    if do_flip:
        windows = fliplr_label(windows, img_width, flip_pairs)

    # 6. Apply Affine Transform on joints
    for idx in range(len(windows)):
        for n_jt in range(len(windows[idx])):
            windows[idx][n_jt, :] = trans_point2d(windows[idx][n_jt, :], trans)

    # 7. get label of some type according to certain need
    label = label_func(label_config, patch_width, patch_height, windows)

    # 8. get gt loc for AE method
    gt_loc = np.zeros((10, 4, 2))
    if label_config.useAE:
        gt_loc = np.transpose(np.array(windows[0: 4]), (1, 0, 2))
        gt_loc = (gt_loc / label_config.feat_stride + 0.5).astype(int)
        hm_size = patch_height // label_config.feat_stride
        gt_loc = KeypointsRef(gt_loc, hm_size)

    VIS = False
    if VIS:
        debug_vis(img_patch_cv, windows, label=label, raw_img = cvimg)

    return img_patch_tensor, label, gt_loc

def get_single_patch_sample_inference(img_path, patch_width, patch_height, mean, std, aug_config):
    # 1. load image
    cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)

    img_height, img_width, img_channels = cvimg.shape #original shape

    # 3. get augmentation params
    scale, rot, center, do_flip, color_scale = 1.0, 0, np.zeros(2), False, [1.0, 1.0, 1.0]

    # 4. generate image patch
    aug_param = [scale, rot, center]
    img_patch_cv, trans = generate_patch_image_cv(cvimg.copy(), img_width, img_height,
                                                  patch_width, patch_height, do_flip, aug_param)
    img_patch_tensor = convert_cvimg_to_tensor(img_patch_cv)

    # apply normalization
    for n_c in range(img_channels):
        img_patch_tensor[n_c, :, :] = np.clip(img_patch_tensor[n_c, :, :] * color_scale[n_c], 0, 255)
        if aug_config.use_color_normalize and mean is not None and std is not None:
            img_patch_tensor[n_c, :, :] = (img_patch_tensor[n_c, :, :] - mean[n_c]) / std[n_c]

    return img_patch_tensor

