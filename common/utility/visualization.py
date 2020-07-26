import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.utility.utils import float2int

FONT_STYLE = cv2.FONT_HERSHEY_PLAIN

def plot_LearningCurve(train_loss, valid_loss, log_path, jobName):
    '''
    Use matplotlib to plot learning curve at the end of training
    train_loss & valid_loss must be 'list' type
    '''
    plt.figure(figsize=(12, 5))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    epochs = np.arange(len(train_loss))
    plt.plot(epochs, np.array(train_loss), 'r', label='train')
    plt.plot(epochs, np.array(valid_loss), 'b', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_path, jobName + '.png'))


def debug_vis(img, window_corner, label=None, raw_img=None, plotLine=True):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    flag5 = False
    if len(window_corner) == 4:
        left_top, left_bottom, right_bottom, right_top = window_corner
    elif len(window_corner) == 5:
        left_top, left_bottom, right_bottom, right_top, center = window_corner
        flag5 = True
    else:
        assert 0

    num_windows = len(left_top)
    for idx in range(num_windows):
        cv2.putText(cv_img_patch_show,'1',float2int(left_top[idx]), FONT_STYLE, 1, (255,0,0), 1)
        cv2.putText(cv_img_patch_show,'2',float2int(left_bottom[idx]), FONT_STYLE, 1, (0,255,0), 1)
        cv2.putText(cv_img_patch_show,'3',float2int(right_bottom[idx]), FONT_STYLE, 1, (0,0,255), 1)
        cv2.putText(cv_img_patch_show,'4',float2int(right_top[idx]), FONT_STYLE, 1, (0,255,255), 1)

        cv2.circle(cv_img_patch_show, float2int(left_top[idx]), 3, (255, 0, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(left_bottom[idx]), 3, (0, 255, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(right_bottom[idx]), 3, (0, 0, 255), -1)
        cv2.circle(cv_img_patch_show, float2int(right_top[idx]), 3, (0, 255, 255), -1)

        if flag5:
            cv2.putText(cv_img_patch_show, '5', float2int(center[idx]), FONT_STYLE, 1, (255, 255, 0), 1)
            cv2.circle(cv_img_patch_show, float2int(center[idx]), 3, (255, 255, 0), -1)

        if plotLine:
            thickness = 2
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(left_top[idx]), float2int(left_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(left_bottom[idx]), float2int(right_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_bottom[idx]), float2int(right_top[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_top[idx]), float2int(left_top[idx]), color, thickness)

    # ----------- vis label --------------
    if isinstance(label, np.ndarray):
        label_ = label.copy() * 255.0
        empty = np.ones((10, cv_img_patch_show.shape[1], 3), dtype=cv_img_patch_show.dtype)*255
        label_to_draw = np.hstack((label_[0], label_[1], label_[2], label_[3])).astype(cv_img_patch_show.dtype)
        label_to_draw = cv2.cvtColor(label_to_draw, cv2.COLOR_GRAY2BGR)
        cv_img_patch_show = np.vstack((cv_img_patch_show, empty, label_to_draw))

    cv2.imshow('patch', cv_img_patch_show)
    cv2.waitKey(0)


def vis_eval_result(img, window, plotLine=False, saveFilename=None):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    for idx in range(len(window)):
        lt, lb, rb, rt = window[idx]['position'][:4]

        if plotLine:
            thickness = 3
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, thickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), 3, (255, 0, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), 3, (128, 200, 50), -1)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), 3, (0, 0, 255), -1)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), 3, (0, 255, 255), -1)

    if saveFilename != None:
        dirname = os.path.dirname(saveFilename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(os.path.join(saveFilename), cv_img_patch_show)
    else:
        cv2.imshow('Vis Evaluation Result', cv_img_patch_show)
        cv2.waitKey(0)


def vis_eval_result_with_gt(img, predWindow, gtWindow, plotLine=False, saveFilename=None):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    predColor = (0,255,0)
    gtColor = (0,0,255)
    kptRadius = 3
    kptThickness = -1

    # GT
    for idx in range(len(gtWindow)):
        lt, lb, rb, rt = gtWindow[idx]

        if plotLine:
            lineThickness = 2
            color = (50, 50, 250)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, lineThickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), 3, gtColor, kptThickness)

    # PRED
    for idx in range(len(predWindow)):
        lt, lb, rb, rt = predWindow[idx]['position'][:4]
        score = predWindow[idx]['score']

        if plotLine:
            lineThickness = 2
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, lineThickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), kptRadius, predColor, kptThickness)

        cv2.putText(cv_img_patch_show, '%.2f' % score,
                    float2int(lt[:2]), FONT_STYLE, 1, (0, 255, 255), 1)

    if saveFilename != None:
        dirname = os.path.dirname(saveFilename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(os.path.join(saveFilename), cv_img_patch_show)
    else:
        cv2.imshow('Vis Evaluation Result', cv_img_patch_show)
        cv2.waitKey(0)