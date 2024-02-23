import copy
import io
import itertools
import json
import numpy as np
import os
import torch
from collections import OrderedDict
from utils import misc


class Evaluator():
    def __init__(self, dataset_name, args, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        if "TotalText" in dataset_name:
            self.dataset_name = "totaltext"
            self.class_num = 2
            self.semantic_classname = {
                0: 'background',
                1: 'text',
            }
        elif "HierText" in dataset_name:
            self.dataset_name = "hiertext"
            self.class_num = 2
            self.semantic_classname = {
                0: 'background',
                1: 'text',
            }
        elif "TextSeg" in dataset_name:
            self.dataset_name = "textseg"
            self.class_num = 2
            self.semantic_classname = {
                0: 'background',
                1: 'text',
            }
        else:
            raise NotImplementedError

    def reset(self):
        self._predictions = []
        self._hr_predictions = []

    def process(self, predictions, hr_predictions, gts, ignore_mask=None):
        assert predictions.shape[0] == 1, "Only support one batch per GPU now."
        assert predictions.shape == gts.shape
        assert hr_predictions.shape == gts.shape
        if ignore_mask is not None:
            assert ignore_mask.shape == gts.shape
            predictions[ignore_mask==255] = False
            hr_predictions[ignore_mask==255] = False
        for pred, hr_pred, gt in zip(predictions, hr_predictions, gts):
            pred = pred.squeeze(0).to(self._cpu_device).detach().numpy()  # h, w
            hr_pred = hr_pred.squeeze(0).to(self._cpu_device).detach().numpy()  # h, w
            gt = gt.squeeze(0).to(self._cpu_device).detach().numpy()

            img_result_dict = self.get_IandU(pred, gt)
            pn = self.label_count(pred)
            gn = self.label_count((gt>127).astype(int))
            img_result_dict.update(pred_num=pn, gt_num=gn)
            self._predictions.append(img_result_dict)

            img_hr_result_dict = self.get_IandU(hr_pred, gt)
            hr_pn = self.label_count(hr_pred)
            img_hr_result_dict.update(pred_num=hr_pn, gt_num=gn)
            self._hr_predictions.append(img_hr_result_dict)

    def get_IandU(self, pred, gt, m=None):
        if pred.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError
        if gt.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError
        pred = pred.copy().astype(int)
        gt = (gt.copy() > 127).astype(int)
        max_index = self.class_num  # 2 for Total-Text

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
        intersection = tp[:max_index].tolist() # remove ignore
        union = pdn + gtn - tp
        union = union[:max_index].tolist()
        return {"intersection": intersection, "union": union}

    def label_count(self, x, m=None):
        if x.dtype not in [bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError
        x = x.copy().astype(int)
        max_index = self.class_num

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

    def evaluate(self):
        if self._distributed:
            misc.synchronize()
            predictions = misc.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            hr_predictions = misc.gather(self._hr_predictions, dst=0)
            hr_predictions = list(itertools.chain(*hr_predictions))

            if not misc.is_main_process():
                return {}
        else:
            predictions = self._predictions
            hr_predictions = self._hr_predictions

        if len(predictions) == 0:
            print("Did not receice valid predictions")
            return {}

        self._results = OrderedDict()

        final_i = [x["intersection"] for x in predictions]
        final_i = np.array(final_i)  # (300,2)
        final_i_hr = [x["intersection"] for x in hr_predictions]
        final_i_hr = np.array(final_i_hr)

        final_u = [x["union"] for x in predictions]
        final_u = np.array(final_u)
        final_u_hr = [x["union"] for x in hr_predictions]
        final_u_hr = np.array(final_u_hr)

        final_pn = [x["pred_num"] for x in predictions]
        final_pn = np.array(final_pn)
        final_pn_hr = [x["pred_num"] for x in hr_predictions]
        final_pn_hr = np.array(final_pn_hr)

        final_gn = [x["gt_num"] for x in predictions]
        final_gn = np.array(final_gn)

        ii = final_i.sum(axis=0)
        uu = final_u.sum(axis=0)
        iou = ii.astype(float) / (uu.astype(float))
        miou = np.nanmean(iou)
        ii_hr = final_i_hr.sum(axis=0)
        uu_hr = final_u_hr.sum(axis=0)
        iou_hr = ii_hr.astype(float) / (uu_hr.astype(float))
        miou_hr = np.nanmean(iou_hr)

        pn_save = copy.deepcopy(final_pn)
        gn_save = copy.deepcopy(final_gn)
        '''image-wise fscore'''
        pn_save[final_pn == 0] = 1
        gn_save[final_gn == 0] = 1
        prec_imwise = final_i.astype(float) / (pn_save.astype(float))
        recl_imwise = final_i.astype(float) / (gn_save.astype(float))
        prec_imwise[final_pn == 0] = 0
        recl_imwise[final_gn == 0] = 0
        prec_imwise = prec_imwise.mean(axis=0)
        recl_imwise = recl_imwise.mean(axis=0)
        fscore_imwise = 2 * prec_imwise * recl_imwise / (prec_imwise + recl_imwise)

        pn_save_hr = copy.deepcopy(final_pn_hr)
        pn_save_hr[final_pn_hr == 0] = 1
        prec_imwise_hr = final_i_hr.astype(float) / (pn_save_hr.astype(float))
        recl_imwise_hr = final_i_hr.astype(float) / (gn_save.astype(float))
        prec_imwise_hr[final_pn_hr == 0] = 0
        recl_imwise_hr[final_gn == 0] = 0
        prec_imwise_hr = prec_imwise_hr.mean(axis=0)
        recl_imwise_hr = recl_imwise_hr.mean(axis=0)
        fscore_imwise_hr = 2 * prec_imwise_hr * recl_imwise_hr / (prec_imwise_hr + recl_imwise_hr)

        self._results['---mIOU'] = float(miou)
        self._results['---mIOU_hr'] = float(miou_hr)
        for idx in range(1, len(iou)):  # ignore background
            self._results[str(idx).zfill(3)+'-'+self.semantic_classname[idx]+'-IOU'] = float(iou[idx])
            self._results[str(idx).zfill(3) + '-' + self.semantic_classname[idx] + '-Fscore'] = float(fscore_imwise[idx])
            self._results[str(idx).zfill(3) + '-' + self.semantic_classname[idx] + '-IOU_hr'] = float(iou_hr[idx])
            self._results[str(idx).zfill(3) + '-' + self.semantic_classname[idx] + '-Fscore_hr'] = float(fscore_imwise_hr[idx])

        return copy.deepcopy(self._results)
