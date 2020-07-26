import os
import cv2
import glob
import json
import logging

import numpy as np
import pickle
import xml.etree.ElementTree as et
from collections import namedtuple
from functools import cmp_to_key

from .imdb import IMDB
from common.utility.visualization import debug_vis, vis_eval_result, vis_eval_result_with_gt
from common.utility.utils import float2int
from .evaluation import getAp

num_corners = 4
max_num_windows = 100
point = namedtuple('point', ['x', 'y'])
flip_pairs = np.array([[0, 3], [1, 2]], dtype=np.int)

# For data parsing
def notBG(im, x, y):
    flag = (im[int(y) - 1, int(x) - 1] != np.array([130, 130, 130]))[0] and \
           (im[int(y) - 1, int(x) - 1] != np.array([7, 7, 7]))[0]
    return flag

def isValid(shape, pts):
    im_height = shape[0]
    im_width  = shape[1]

    return pts.x >= 0 and pts.x <= im_width - 1 and pts.y >= 0 and pts.y <= im_height - 1

def sortWindow(win1, win2):
    if abs(win1[1] - win2[1]) < 3:
        return win1[0] - win2[0]
    else:
        return win1[1] - win2[1]

def calcPixelAcc(windows, winBoolMap, totalGTPixel):
    '''
    :param windows: N x 4 x 2
    :param winBoolMap: H x W
    :param totalGTPixel: int
    :return:
    '''
    predBoolMap = np.zeros(winBoolMap.shape, dtype=np.uint8)
    for winKpts in windows:
        try:
            cv2.fillConvexPoly(predBoolMap, winKpts, 1)
        except Exception as e:
            assert 0, (e, winKpts)

    predBoolMap = (predBoolMap == 1)
    interArea = np.logical_and(predBoolMap, winBoolMap)
    correctPixelCount = interArea.sum()

    return 1.0 * correctPixelCount / totalGTPixel

class facade(IMDB):
    def __init__(self, benchmark_name, image_subset_name, dataset_path):
        super(facade, self).__init__(benchmark_name, image_subset_name, dataset_path)
        self.use_xml = False
        if 'XML' in self.benchmark_name:
            self.use_xml = True

        self.flip_pairs = np.array([[0, 3], [1, 2]], dtype=np.int)

    def parse_json_annotation(self, annotation_path, im_path, shape):
        window_list = list()

        with open(annotation_path, 'r') as fid:
            label = json.load(fid)

        assert label['imagePath'] == os.path.basename(im_path), \
            "Wrong Annotation<->Image correspond, %s" % annotation_path

        for item in label['shapes']:
            if item['label'] == 'window':
                try:
                    left_top, left_bottom, right_bottom, right_top = item['points']
                except:
                    assert 0, 'Non-Four-Points in %s' % annotation_path

                left_top = point(*left_top)
                left_bottom = point(*left_bottom)
                right_bottom = point(*right_bottom)
                right_top = point(*right_top)

                # examine pts relation correctness
                if not left_top.x < right_top.x or not left_bottom.x < right_bottom.x or \
                        not left_top.y < left_bottom.y or not right_top.y < right_bottom.y:
                    assert 0, 'wrong relation %s' % annotation_path

                assert isValid(shape, left_top) and isValid(shape, left_bottom) and \
                       isValid(shape, right_bottom) and isValid(shape, right_top), \
                    "invalid point position in %s" % annotation_path

                window_list.append([left_top.x, left_top.y, left_bottom.x, left_bottom.y,
                                    right_bottom.x, right_bottom.y, right_top.x, right_top.y])
            else:
                print("Unknow Label Type %s in %s" % (item['label'], annotation_path))

        window_list = sorted(window_list, key=cmp_to_key(sortWindow))
        return window_list


    def parse_xml_annotation(self, annotation_path, im_path, shape):
        im_height, im_width, im_chn = shape

        window_list = list()

        root = et.parse(annotation_path).getroot()
        for obj in root:
            label_name = obj[2].text.strip()
            if label_name == 'window':
                y1 = float(obj[0][0].text.strip()) * im_height - 1
                y2 = float(obj[0][1].text.strip()) * im_height - 1
                x1 = float(obj[0][2].text.strip()) * im_width - 1
                x2 = float(obj[0][3].text.strip()) * im_width - 1

                left_top = point(*float2int((x1, y1)))
                left_bottom = point(*float2int((x1, y2)))
                right_bottom = point(*float2int((x2, y2)))
                right_top = point(*float2int((x2, y1)))

                assert isValid(shape, left_top) and isValid(shape, left_bottom) and \
                       isValid(shape, right_bottom) and isValid(shape, right_top), \
                    "invalid point position in %s" % annotation_path

                window_list.append([left_top.x, left_top.y, left_bottom.x, left_bottom.y,
                                    right_bottom.x, right_bottom.y, right_top.x, right_top.y])

        # window_list = sorted(window_list, key=itemgetter(1, 0))
        window_list = sorted(window_list, key=cmp_to_key(sortWindow))
        return window_list


    def parse_gt_file(self, data_root_path):
        img_list = list()
        window_list = list()
        shape_list = list()

        images_path_list = glob.glob(data_root_path + '/images/*.jpg')
        images_path_list += glob.glob(data_root_path + '/images/*.png')
        if not images_path_list:
            assert 0, "Zero Image Path"

        images_path_list.sort()

        for s_img_path in images_path_list:
            img_list.append(s_img_path)
            im = cv2.imread(s_img_path)
            try:
                shape = im.shape
            except:
                assert 0, "Mistake loading image %s shape"%s_img_path

            if self.use_xml:
                annotation_filename = os.path.basename(s_img_path).rsplit('.', 1)[0] + '.xml'
                annotation_path = os.path.join(data_root_path, 'annotation', annotation_filename)
                the_window_anno = self.parse_xml_annotation(annotation_path, s_img_path, shape)
            else:
                annotation_filename = os.path.basename(s_img_path).rsplit('.', 1)[0] + '.json'
                annotation_path = os.path.join(data_root_path, 'annotation', annotation_filename)
                the_window_anno = self.parse_json_annotation(annotation_path, s_img_path, shape)

            window_list.append(the_window_anno)
            shape_list.append(shape)

        return img_list, window_list, shape_list


    def gt_db(self, is_train):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_db.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pickle.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            return db

        if is_train:
            img_list, window_anno_list, shape_list = self.parse_gt_file(
                os.path.join(self.dataset_path, 'TRAIN', self.benchmark_name))
        else:
            img_list, window_anno_list, shape_list = self.parse_gt_file(
                os.path.join(self.dataset_path, 'TEST'))

        gt_db = list()
        for n_img in range(len(img_list)):
            image_path = img_list[n_img]
            the_sample_window = np.array(window_anno_list[n_img], dtype=np.float)
            im_height, im_width, _ = shape_list[n_img]

            if len(the_sample_window) > max_num_windows: #exclude windows whose num exceeds max_num_windows
                continue

            left_top = the_sample_window[:, 0: 2].copy()
            left_bottom = the_sample_window[:, 2: 4].copy()
            right_bottom = the_sample_window[:, 4: 6].copy()
            right_top = the_sample_window[:, 6: 8].copy()
            center = the_sample_window.reshape((-1, 4, 2)).mean(axis=1)

            the_sample_window = the_sample_window.reshape((the_sample_window.shape[0], 4, 2))

            gt_db.append({
                'image': image_path,
                'left_top': left_top,
                'left_bottom': left_bottom,
                'right_bottom': right_bottom,
                'right_top': right_top,
                'center': center,
                'windows': np.array(the_sample_window),
                'im_width': im_width,
                'im_height': im_height
            })

            DEBUG = False
            if DEBUG:
                debug_vis(image_path, (left_top, left_bottom, right_bottom, right_top), 'Dataset Parsing')

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_db, fid, pickle.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        return gt_db

    @staticmethod
    def evaluate(windows_list, imdb_list, save_path, fullEval, plot):
        # pre-processing
        ap_gt = {}
        ap_pred = []
        for s_idx in range(len(windows_list)):
            im = imdb_list[s_idx]['image']

            # aggreate gt into dict
            winGT = imdb_list[s_idx]['windows']
            numOfWindows = winGT.shape[0]
            winGT = winGT.reshape((numOfWindows, 4, 2))

            wins_of_this_gt = list()
            for i in range(numOfWindows):
                temp = {}
                temp['position'] = winGT[i]  # 4x2 array
                temp['is_matched'] = 0  # initialize to 0 (not matched)
                wins_of_this_gt.append(temp)
            ap_gt[s_idx] = wins_of_this_gt.copy()

            # aggreate pred into list
            winPred = windows_list[s_idx]
            for i in range(len(winPred)):
                temp = {}
                temp['position'] = np.array(winPred[i]['position'])[:, :2].copy()  # 4x2 array
                temp['img_id'] = s_idx  # index of image
                temp['score'] = winPred[i]['score']  # confidence
                ap_pred.append(temp)

            if plot:
                visFilename = os.path.basename(im)
                visFilename = os.path.join(save_path, 'vis', visFilename)
                # vis_eval_result(im, winPred, plotLine=True, saveFilename=os.path.join(save_path, 'vis', visFilename))
                vis_eval_result_with_gt(im, winPred, winGT, plotLine=True, saveFilename=visFilename)

        ap, precision, recall = getAp(ap_gt, ap_pred, fullEval=fullEval)
        name_value = [
            ('AP   @ IoU0.5          :', ap[1]),
            ('Prec @ IoU0.5          :', precision[1]),
            ('Reca @ IoU0.5          :', recall[1]),
            ('-----------------------:', 0.),
            ('AP   @ IoU0.75         :', ap[2]),
            ('Prec @ IoU0.75         :', precision[2]),
            ('Reca @ IoU0.75         :', recall[2]),
            ('-----------------------:', 0.),
            ('AP   @ IoU0.5:0.95     :', ap[0]),
            ('Prec @ IoU0.5:0.95     :', precision[0]),
            ('Reca @ IoU0.5:0.95     :', recall[0]),
        ]

        pklSave = False
        if pklSave:
            logging.info("Save window into %s" % save_path)
            with open(os.path.join(save_path, 'window.pkl'), 'wb') as fid:
                pickle.dump(windows_list, fid, pickle.HIGHEST_PROTOCOL)

        return name_value

    @staticmethod
    def plot(windows_list, imdb_list, save_path):
        # pre-processing
        ap_pred = []
        for s_idx in range(len(windows_list)):
            im = imdb_list[s_idx]['image']

            # aggreate pred into list
            winPred = windows_list[s_idx]
            for i in range(len(winPred)):
                temp = {}
                temp['position'] = np.array(winPred[i]['position'])[:, :2].copy()  # 4x2 array
                temp['img_id'] = s_idx  # index of image
                temp['score'] = winPred[i]['score']  # confident
                ap_pred.append(temp)

            visFilename = "vis_" + os.path.basename(im)
            visFilename = os.path.join(save_path, visFilename)
            vis_eval_result(im, winPred, plotLine=True, saveFilename=visFilename)