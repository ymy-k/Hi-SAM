import os.path
import sys

import numpy as np
import torch
from skimage import io
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
import copy


def get_IandU(pred, gt, m=None):
    if pred.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
        raise ValueError
    if gt.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
        raise ValueError
    pred = pred.copy().astype(int)
    gt = (gt.copy() > 127).astype(int)
    max_index = 2

    if m is None:
        m = np.ones(pred.shape, dtype=np.uint8)
    mgt = m * (gt >= 0) & (gt < max_index)
    gt[mgt == 0] = max_index
    # here set it with a new "ignore" index

    mx = mgt * ((pred >= 0) & (pred < max_index))
    # all gt ignores will be ignored, but if x predict
    # ignore on non-ignore pixel, that will count as an error.
    pred[mx == 0] = max_index

    bdd_index = max_index + 1  # 3
    # include the ignore

    cmp = np.bincount((pred + gt * bdd_index).flatten())
    cm = np.zeros((bdd_index * bdd_index)).astype(int)
    cm[0:len(cmp)] = cmp
    cm = cm.reshape(bdd_index, bdd_index)
    pdn = cm.sum(axis=0)
    gtn = cm.sum(axis=1)
    tp = np.diag(cm)
    intersection = tp[:max_index].tolist()  # remove ignore
    union = pdn + gtn - tp
    union = union[:max_index].tolist()
    return {"intersection": intersection, "union": union}


def label_count(x, m=None):
    if x.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
        raise ValueError
    x = x.copy().astype(int)
    max_index = 2

    if m is None:
        m = np.ones(x.shape, dtype=int)
    else:
        m = (m > 0).astype(int)

    mx = m * (x >= 0) & (x < max_index)
    # here set it with a new "ignore" index
    x[mx == 0] = max_index

    counti = np.bincount(x.flatten())
    counti = counti[0:max_index]
    counti = list(counti)
    if len(counti) < max_index:
        counti += [0] * (max_index - len(counti))
    return counti


result_folder = "img_eval"
gt_folder = "datasets/HierText/test_gt"
gts = glob(gt_folder+'/*')
semantic_classname = {0: 'background', 1: 'text'}
predictions = []
for gt_name in tqdm(gts):
    img_id = os.path.basename(gt_name).split('.')[0]
    gt_data = io.imread(gt_name)
    gt_data = (gt_data > 127).astype(np.uint8) * 255
    if len(gt_data.shape) > 2:
        gt_data = gt_data[:, :, 0]

    res_data = io.imread(os.path.join(result_folder, img_id+'.png'))
    res_data = res_data > 127

    img_result_dict = get_IandU(res_data, gt_data)
    gn = label_count((gt_data>127).astype(int))
    pn = label_count(res_data)
    img_result_dict.update(pred_num=pn, gt_num=gn)
    predictions.append(img_result_dict)

results = OrderedDict()
final_i = [x["intersection"] for x in predictions]
final_i = np.array(final_i)
final_u = [x["union"] for x in predictions]
final_u = np.array(final_u)
final_pn = [x["pred_num"] for x in predictions]
final_pn = np.array(final_pn)
final_gn = [x["gt_num"] for x in predictions]
final_gn = np.array(final_gn)
ii = final_i.sum(axis=0)
uu = final_u.sum(axis=0)
iou = ii.astype(float) / (uu.astype(float))
miou = np.nanmean(iou)
pn_save = copy.deepcopy(final_pn)
gn_save = copy.deepcopy(final_gn)
pn_save[final_pn == 0] = 1
gn_save[final_gn == 0] = 1
prec_imwise = final_i.astype(float) / (pn_save.astype(float))
recl_imwise = final_i.astype(float) / (gn_save.astype(float))
prec_imwise[final_pn == 0] = 0
recl_imwise[final_gn == 0] = 0
prec_imwise = prec_imwise.mean(axis=0)
recl_imwise = recl_imwise.mean(axis=0)
fscore_imwise = 2 * prec_imwise * recl_imwise / (prec_imwise + recl_imwise)
results['---mIOU'] = float(miou)
for idx in range(1, len(iou)):  # ignore background
    results[str(idx).zfill(3)+'-' + semantic_classname[idx]+'-IOU'] = float(iou[idx])
    results[str(idx).zfill(3) + '-' + semantic_classname[idx] + '-Fscore'] = float(fscore_imwise[idx])
print(results)