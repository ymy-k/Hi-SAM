"""HierText's PQ-like evaluator class."""

import enum
from typing import Dict, List, Union

import numpy as np

from . import np_box_mask_list
from . import np_box_mask_list_ops
from . import polygon_list
from . import polygon_ops


class TextBoxRep(enum.Enum):
  MASK = 0
  POLY = 1

EPSILON = 1e-5
TextEntityListType = Union[
    polygon_list.PolygonList,
    np_box_mask_list.BoxMaskList]
NumpyDict = Dict[str, np.ndarray]


class HierTextEvaluator(object):
  """Evaluator class.

  Attributes:
    iou_threshold: IoU threshold to use for matching groundtruth boxes/masks
      to detection boxes/masks.
    text_box_type: The type of the detection representation.
    evaluate_text: Whether to evaluate predicted text.
    filter_invalid_thr: Predictions will be removed if the pixel-level
      precision with any groundtruths that are marked as `dont't care` is larger
      than this valule.
  """

  def __init__(self,
               iou_threshold: float = 0.5,
               text_box_type: TextBoxRep = TextBoxRep.MASK,
               evaluate_text: bool = False,
               filter_invalid_thr: float = 0.5):
    """Constructor."""
    self._iou_threshold = iou_threshold
    self._text_box_type = text_box_type
    self._evaluate_text = evaluate_text
    self._filter_invalid_thr = filter_invalid_thr

  def _init_new_accumulators(self):
    """Initialize the counter dict."""
    # For detection:
    accumulators = {
        'tp_det_cnt': 0.,
        'num_prediction': 0.,
        'num_groundtruth': 0.,
        'tp_det_tightness_sum': 0.,
    }
    # For E2E:
    if self._evaluate_text:
      accumulators_for_e2e = {
          'tp_e2e_cnt': 0.,
          'tp_e2e_tightness_sum': 0.,
      }
      accumulators.update(accumulators_for_e2e)
    return accumulators

  def _get_text(self,
                prediction_box_list: TextEntityListType,
                gt_box_list: TextEntityListType):
    """This function decodes the text content from the text box containers.

    Args:
      prediction_box_list: The predicted text entities.
      gt_box_list: The GT text entities.

    Returns:
      pred_text: A list of str for the predicted text.
      gt_text: A list of str for the ground-truth text.
    """
    pred_text = []
    gt_text = []

    if self._evaluate_text:
      pred_text = [str(text) for text in prediction_box_list.get_field(
          'pred_texts')]
      gt_text = [str(text) for text in gt_box_list.get_field(
          'gt_texts')]
    return pred_text, gt_text

  def _count_tp(self,
                num_gt: int,
                iou: np.ndarray,
                max_iou_ids_for_gts: np.ndarray,
                max_iou_ids_for_preds: np.ndarray,
                pred_text: List[str],
                gt_text: List[str],
                accumulators: Dict[str, Union[float, int]]):
    """Matching groundtruths and predictions."""
    for i in range(num_gt):
      # get the id of prediction that has highest IoU with i-th GT
      max_prediction_id = max_iou_ids_for_gts[i]

      box_score = iou[i, max_prediction_id]

      if (box_score >= self._iou_threshold and
          # mutually best match
          i == max_iou_ids_for_preds[max_prediction_id]):
        accumulators['tp_det_cnt'] += 1
        accumulators['tp_det_tightness_sum'] += box_score

        # For E2E:
        if self._evaluate_text:
          if pred_text[max_prediction_id] == gt_text[i]:
            accumulators['tp_e2e_cnt'] += 1
            accumulators['tp_e2e_tightness_sum'] += box_score

  def _filter_dont_care_region(self,
                               eval_dict: NumpyDict,
                               prediction_box_list: TextEntityListType,
                               gt_box_list: TextEntityListType):
    """Remove pred if intersection(pred, dont' care gt) / area(pred) >= filter_invalid_thr."""
    valid_gt_indices = np.where(eval_dict['gt_weights'] >= 0.5)[0]
    invalid_gt_indices = np.where(eval_dict['gt_weights'] < 0.5)[0]

    if invalid_gt_indices.size and prediction_box_list.num_boxes():
      invalid_gt_box_list = self._gather(gt_box_list, invalid_gt_indices)
      precision, _ = self._precision_recall(
          prediction_box_list, invalid_gt_box_list)
      valid_prediction_indices = np.where(
          np.max(precision, axis=1) < self._filter_invalid_thr)[0]
      prediction_box_list = self._gather(prediction_box_list,
                                         valid_prediction_indices)

    gt_box_list = self._gather(gt_box_list, valid_gt_indices)

    return prediction_box_list, gt_box_list

  def evaluate_one_image(self, eval_dict: NumpyDict):
    """Evaluation code."""
    accumulators = self._init_new_accumulators()

    # step 1: convert to box list objects
    gt_box_list, prediction_box_list = self._get_box_lists(eval_dict)

    # step 2: filter out don't care regions
    prediction_box_list, gt_box_list = self._filter_dont_care_region(
        eval_dict, prediction_box_list, gt_box_list)

    # step 3: evaluate
    num_gt = gt_box_list.num_boxes()
    num_prediction = prediction_box_list.num_boxes()

    if num_gt and num_prediction:
      iou = self._iou(gt_box_list, prediction_box_list)
      max_iou_ids_for_gts = np.argmax(iou, axis=1)  # for GT
      max_iou_ids_for_preds = np.argmax(iou, axis=0)  # for pred
      pred_text, gt_text = self._get_text(prediction_box_list, gt_box_list)

      self._count_tp(num_gt, iou,
                     max_iou_ids_for_gts, max_iou_ids_for_preds,
                     pred_text, gt_text,
                     accumulators)

    accumulators['num_prediction'] += num_prediction
    accumulators['num_groundtruth'] += num_gt

    return accumulators

  def _get_box_lists(self, eval_dict: NumpyDict):
    """Get ground truth and prediction box list."""
    if self._text_box_type == TextBoxRep.MASK:
      gt_box_list = np_box_mask_list.BoxMaskList(
          eval_dict['gt_boxes'], eval_dict['gt_masks'].astype(np.uint8))
      prediction_box_list = np_box_mask_list.BoxMaskList(
          eval_dict['detection_boxes'],
          eval_dict['detection_masks'].astype(np.uint8))
    elif self._text_box_type == TextBoxRep.POLY:
      gt_box_list = polygon_list.PolygonList(eval_dict['gt_boxes'])
      prediction_box_list = polygon_list.PolygonList(
          eval_dict['detection_boxes'])
    else:
      raise NotImplementedError()

    for key in ['gt_weights', 'gt_texts']:
      if key in eval_dict:
        gt_box_list.add_field(key, eval_dict[key])
    if 'pred_texts' in eval_dict:
      prediction_box_list.add_field('pred_texts', eval_dict['pred_texts'])

    return gt_box_list, prediction_box_list

  def _gather(self, box_list, indices):
    """Gather regions by indices."""
    if self._text_box_type == TextBoxRep.MASK:
      return np_box_mask_list_ops.gather(box_list, indices)
    elif self._text_box_type == TextBoxRep.POLY:
      return polygon_ops.gather(box_list, indices)
    else:
      raise NotImplementedError()

  def _iou(self, boxlist1, boxlist2):
    """Computes intersection-over-union."""
    if self._text_box_type == TextBoxRep.MASK:
      return np_box_mask_list_ops.iou(boxlist1, boxlist2)
    elif self._text_box_type == TextBoxRep.POLY:
      return polygon_ops.iou(boxlist1, boxlist2)
    else:
      raise NotImplementedError()

  def _precision_recall(self, boxlist1, boxlist2):
    """Computes pixel precision and recall matrices."""

    if self._text_box_type == TextBoxRep.MASK:
      # intersection: ndarray[N x M]
      intersection = np_box_mask_list_ops.intersection(boxlist1, boxlist2)
      # box_1_area: ndarray[N x 1], box_2_area: ndarray[1 x M]
      box_1_area = np_box_mask_list_ops.area(boxlist1).reshape(-1, 1)
      box_2_area = np_box_mask_list_ops.area(boxlist2).reshape(1, -1)
    elif self._text_box_type == TextBoxRep.POLY:
      # intersection: ndarray[N x M]
      # box_1_area: ndarray[N x 1], box_2_area: ndarray[1 x M]
      intersection, box_1_area, box_2_area = polygon_ops.intersection_and_area(
          boxlist1, boxlist2)
    else:
      raise NotImplementedError()
    precision = intersection / (box_1_area + EPSILON)
    recall = intersection / (box_2_area + EPSILON)
    return precision, recall

  def evaluate(self, accumulators):
    """Compute evaluation metrics."""
    metrics = {}
    total_gt = accumulators['num_groundtruth']
    total_prediction = accumulators['num_prediction']
    total_tp_det = accumulators['tp_det_cnt']
    r_det = total_tp_det / total_gt if total_gt else 1.0
    p_det = total_tp_det / total_prediction if total_prediction else 1.0
    metrics['Det-Recall'] = r_det
    metrics['Det-Precision'] = p_det
    metrics['Det-Fscore'] = (2 * r_det * p_det / (r_det + p_det) if
                             (r_det + p_det) else 0.0)
    metrics['Det-Tightness'] = (
        (accumulators['tp_det_tightness_sum'] /
         total_tp_det) if total_tp_det else 1.0)
    metrics['Det-PQ'] = metrics['Det-Tightness'] * metrics['Det-Fscore']

    if self._evaluate_text:
      total_tp_e2e = accumulators['tp_e2e_cnt']
      r_e2e = total_tp_e2e / total_gt if total_gt else 1.0
      p_e2e = total_tp_e2e / total_prediction if total_prediction else 1.0
      metrics['E2E-Recall'] = r_e2e
      metrics['E2E-Precision'] = p_e2e
      metrics['E2E-Fscore'] = (2 * r_e2e * p_e2e / (r_e2e + p_e2e) if
                               (r_e2e + p_e2e) else 0.0)
      metrics['E2E-Tightness'] = (
          accumulators['tp_e2e_tightness_sum'] /
          total_tp_e2e if total_tp_e2e else 1.0)
      metrics['E2E-PQ'] = metrics['E2E-Tightness'] * metrics['E2E-Fscore']

    return metrics
