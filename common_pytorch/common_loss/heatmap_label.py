import cv2
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

def find_peaks(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param['thre1'])
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

def compute_resized_coords(coords, resizeFactor):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5

def get_coords_from_heatmaps_with_NMS(heatmap, sigma=2, param_thre1=0.1):
    assert isinstance(heatmap, np.ndarray)

    all_peaks = []
    peak_counter = 0

    for part in range(heatmap.shape[0]):
        map_ori = heatmap[part, :, :]
        map_gau = gaussian_filter(map_ori, sigma=sigma)

        map_left = np.zeros(map_gau.shape)
        map_left[1:, :] = map_gau[:-1, :]
        map_right = np.zeros(map_gau.shape)
        map_right[:-1, :] = map_gau[1:, :]
        map_up = np.zeros(map_gau.shape)
        map_up[:, 1:] = map_gau[:, :-1]
        map_down = np.zeros(map_gau.shape)
        map_down[:, :-1] = map_gau[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
             map_gau >= map_down, map_gau > param_thre1))

        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
        peaks = list(peaks)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        ids = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

        all_peaks.append(np.array(peaks_with_score_and_id))
        peak_counter += len(peaks)

    return all_peaks

def generate_gaussian_heatmap_label(feat_stride, patch_width, patch_height, window, sigma):
    bound = 3
    num_type_points = len(window)

    # get a downsampled heatmap
    hm_width = patch_width // feat_stride
    hm_height = patch_height // feat_stride

    # init heatmap as all zeros
    label = np.zeros((num_type_points, hm_height, hm_width), dtype=np.float)

    # gaussian radius
    tmp_size = sigma * bound

    # generate gaussian shape
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, dtype=float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2  # mean coordinate
    # The gaussian is not normalized, we want the center(peak) value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Check that any part of the gaussian is in-bounds
    for corner_id in range(num_type_points):
        corner = window[corner_id]
        mu_x_list = (corner[:, 0] / feat_stride + 0.5).astype(int)
        mu_y_list = (corner[:, 1] / feat_stride + 0.5).astype(int)

        for mu_x, mu_y in zip(mu_x_list,mu_y_list):
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  # ul:upper left, br:bottom right
            br = [int(mu_x + tmp_size) + 1, int(mu_y + tmp_size + 1)]

            if ul[0] >= hm_width or ul[1] >= hm_height or br[0] < 0 or br[1] < 0:
                continue

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], hm_width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm_height) - ul[1]

            # Image range
            img_x = max(0, ul[0]), min(br[0], hm_width)
            img_y = max(0, ul[1]), min(br[1], hm_height)

            label[corner_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                np.maximum(g[g_y[0]:g_y[1], g_x[0]:g_x[1]], label[corner_id][img_y[0]:img_y[1], img_x[0]:img_x[1]])

    return label