"""PQ metrics for HierText."""

import json
import time
from typing import Sequence

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import cv2
import numpy as np

from evaluator import evaluator


_GT = flags.DEFINE_string('gt', None, 'Groundtruth JSON file.')
_RESULT = flags.DEFINE_string('result', None, 'Prediction JSON file.')
_OUTPUT = flags.DEFINE_string(
    'output', None, 'The output text file containing evaluation results.')

_EVAL_LINES = flags.DEFINE_bool(
    'eval_lines', False, 'Whether to perform line-level evaluation.')
_EVAL_PARAGRAPHS = flags.DEFINE_bool(
    'eval_paragraphs', False, 'Whether to perform paragraph-level evaluation.')
_E2E = flags.DEFINE_bool(
    'e2e', False, 'Whether to perform end-to-end evaluation.')
_MASK_STRIDE = flags.DEFINE_integer(
    'mask_stride', 1,
    ('Downsample the masks for faster but less accuracte results. '
     'Note that when reporting results for a paper, users should use '
     'mask_stirde=1 i.e. no downsampling!'))

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 8, 'The number of workers. Set to 0 to use all the cores.')


def load_annotations(gt_path: str, result_path: str):
  """Loading results and ground truths, and then pairing them."""
  outputs = []
  index = {}

  print('Loading ground-truth annotations.')
  gt_annos = json.load(open(gt_path, encoding='utf-8'))['annotations']
  for anno in gt_annos:
    outputs.append(anno)
    index[anno['image_id']] = anno

  print('Loading predictions.')
  results = json.load(open(result_path, encoding='utf-8'))['annotations']
  for anno in results:
    gt_dict = index[anno['image_id']]
    gt_dict['output_paragraphs'] = anno['paragraphs']

  print('Finished loading.')
  return outputs


def draw_mask(vertices: np.ndarray, w: int, h: int, s: int = 1):
  mask = np.zeros((h, w), dtype=np.float32)
  return cv2.fillPoly(mask, [vertices], [1.])[::s, ::s] > 0


def parse_annotation_dict(anno, eval_lines, eval_paragraphs, mask_stride):
  """Parse the data for the evaluator."""
  t_start = time.time()
  image_id = anno['image_id']

  w = anno['image_width']
  h = anno['image_height']

  gt_word_polygons = []
  gt_word_weights = []
  gt_word_texts = []
  if eval_lines:
    gt_line_masks = []
    gt_line_boxes = []
    gt_line_weights = []
    gt_line_texts = []
  if eval_paragraphs:
    gt_paragraph_masks = []
    gt_paragraph_boxes = []
    gt_paragraph_weights = []

  for paragraph in anno['paragraphs']:
    gt_paragraph_mask = []
    gt_paragraph_box = []
    for line in paragraph['lines']:
      gt_line_mask = []
      gt_line_box = []
      for word in line['words']:
        vertices = np.array(word['vertices'])
        gt_word_polygons.append(vertices)
        gt_word_weights.append(1.0 if word['legible'] else 0.0)
        gt_word_texts.append(word['text'])
        if eval_lines or eval_paragraphs:
          gt_word_mask = draw_mask(vertices, w, h, mask_stride)
          if eval_lines:
            gt_line_mask.append(gt_word_mask)
            gt_line_box.append(vertices)
          if eval_paragraphs:
            gt_paragraph_mask.append(gt_word_mask)
            gt_paragraph_box.append(vertices)
      if eval_lines:
        if not gt_line_mask:
          gt_line_mask = [
              draw_mask(np.array(line['vertices']), w, h, mask_stride)]
          gt_line_box.append(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
        gt_line_masks.append(
            np.any(np.stack(gt_line_mask, axis=0), axis=0).astype(np.float32))
        gt_line_box = np.concatenate(gt_line_box, axis=0)
        gt_line_boxes.append(
            [np.min(gt_line_box[:, 1]),
             np.min(gt_line_box[:, 0]),
             np.max(gt_line_box[:, 1]),
             np.max(gt_line_box[:, 0])])
        gt_line_weights.append(1.0 if line['legible'] else 0.0)
        gt_line_texts.append(line['text'])
    if eval_paragraphs:
      if not gt_paragraph_mask or not paragraph['legible']:
        gt_paragraph_mask = [draw_mask(
            np.array(paragraph['vertices']), w, h, mask_stride)]
        gt_paragraph_box.append(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
      gt_paragraph_masks.append(
          np.any(np.stack(gt_paragraph_mask, axis=0), axis=0).astype(np.float32))
      gt_paragraph_box = np.concatenate(gt_paragraph_box, axis=0)
      gt_paragraph_boxes.append([
          np.min(gt_paragraph_box[:, 1]),
          np.min(gt_paragraph_box[:, 0]),
          np.max(gt_paragraph_box[:, 1]),
          np.max(gt_paragraph_box[:, 0])
      ])
      gt_paragraph_weights.append(1.0 if paragraph['legible'] else 0.0)

  word_polygons = []
  word_texts = []
  if eval_lines:
    line_masks = []
    line_boxes = []
    line_texts = []
  if eval_paragraphs:
    paragraph_masks = []
    paragraph_boxes = []

  for paragraph in anno['output_paragraphs']:
    paragraph_mask = []
    paragraph_box = []
    for line in paragraph['lines']:
      line_mask = []
      line_box = []
      for word in line['words']:
        vertices = np.array(word['vertices'])
        word_polygons.append(vertices)
        word_texts.append(word['text'])
        if eval_lines or eval_paragraphs:
          word_mask = draw_mask(vertices, w, h, mask_stride)
          if eval_lines:
            line_mask.append(word_mask)
            line_box.append(vertices)
          if eval_paragraphs:
            paragraph_mask.append(word_mask)
            paragraph_box.append(vertices)
      if eval_lines:
        if not line_mask:
          raise ValueError('Line does not contain words: %s' % line)
        line_masks.append(
            np.any(np.stack(line_mask, axis=0), axis=0).astype(np.float32))
        line_box = np.concatenate(line_box, axis=0)
        line_boxes.append(
            [np.min(line_box[:, 1]),
             np.min(line_box[:, 0]),
             np.max(line_box[:, 1]),
             np.max(line_box[:, 0])])
        line_texts.append(line['text'])
    if eval_paragraphs:
      if not paragraph_mask:
        raise ValueError('Paragraph does not contain lines: %s' %
                         paragraph)
      paragraph_masks.append(
          np.any(np.stack(paragraph_mask, axis=0), axis=0).astype(np.float32))
      paragraph_box = np.concatenate(paragraph_box, axis=0)
      paragraph_boxes.append([
          np.min(paragraph_box[:, 1]),
          np.min(paragraph_box[:, 0]),
          np.max(paragraph_box[:, 1]),
          np.max(paragraph_box[:, 0])
      ])

  num_gt_words = len(gt_word_polygons)
  num_pred_words = len(word_polygons)

  word_dict = {
      'gt_weights': (np.array(gt_word_weights) if num_gt_words else np.zeros(
          (0,), np.float32)),
      'gt_boxes':
          gt_word_polygons,
      'gt_texts': (np.array(gt_word_texts) if num_gt_words else np.zeros(
          (0,), str)),
      'detection_boxes':
          word_polygons,
      'pred_texts': (np.array(word_texts) if num_pred_words else np.zeros(
          (0,), str)),
  }

  line_dict = {}
  if eval_lines:
    num_gt_lines = len(gt_line_masks)
    num_pred_lines = len(line_masks)
    line_dict = {
        'gt_weights': (np.array(gt_line_weights) if num_gt_lines else np.zeros(
            (0,), np.float32)),
        'gt_masks': (np.stack(gt_line_masks, 0) if num_gt_lines else np.zeros(
            (0, (h + 1) // 2, (w + 1) // 2), np.float32)),
        'gt_boxes': (np.array(gt_line_boxes, np.float32)
                     if num_gt_lines else np.zeros((0, 4), np.float32)),
        'gt_texts': (np.array(gt_line_texts) if num_gt_lines else np.zeros(
            (0,), str)),
        'detection_boxes':
            (np.array(line_boxes, np.float32)
             if num_pred_lines else np.zeros((0, 4), np.float32)),
        'detection_masks':
            (np.stack(line_masks, 0) if num_pred_lines else np.zeros(
                (0, (h + 1) // 2, (w + 1) // 2), np.float32)),
        'pred_texts': (np.array(line_texts) if num_pred_lines else np.zeros(
            (0,), str)),
    }

  paragraph_dict = {}
  if eval_paragraphs:
    num_gt_paragraphs = len(gt_paragraph_masks)
    num_pred_paragraphs = len(paragraph_masks)
    paragraph_dict = {
        'gt_weights':
            (np.array(gt_paragraph_weights) if num_gt_paragraphs else np.zeros(
                (0,), np.float32)),
        'gt_masks':
            (np.stack(gt_paragraph_masks, 0) if num_gt_paragraphs else np.zeros(
                (0, (h + 1) // 2, (w + 1) // 2), np.float32)),
        'gt_boxes':
            (np.array(gt_paragraph_boxes, np.float32)
             if num_gt_paragraphs else np.zeros((0, 4), np.float32)),
        'detection_boxes':
            (np.array(paragraph_boxes, np.float32)
             if num_pred_paragraphs else np.zeros((0, 4), np.float32)),
        'detection_masks':
            (np.stack(paragraph_masks, 0) if num_pred_paragraphs else np.zeros(
                (0, (h + 1) // 2, (w + 1) // 2), np.float32)),
    }
  t_end = time.time()
  print(f'Parsing {image_id} takes {t_end - t_start} secs')
  return image_id, word_dict, line_dict, paragraph_dict


def evaluate_one_image(image_id, word_dict, line_dict, paragraph_dict,
                       word_evaluator, line_evaluator, paragraph_evaluator):
  t_start = time.time()
  word_stats = word_evaluator.evaluate_one_image(word_dict)
  line_stats = {}
  if line_evaluator is not None:
    line_stats = line_evaluator.evaluate_one_image(line_dict)
  paragraph_stats = {}
  if paragraph_evaluator is not None:
    paragraph_stats = paragraph_evaluator.evaluate_one_image(paragraph_dict)
  t_end = time.time()
  print(f'Evaluating {image_id} takes {t_end - t_start} secs')
  return [word_stats, line_stats, paragraph_stats]


def compute_eval_metrics(word_sum, line_sum, paragraph_sum,
                         word_evaluator, line_evaluator, paragraph_evaluator):
  word_metrics = word_evaluator.evaluate(word_sum)
  if line_evaluator is not None:
    line_metrics = line_evaluator.evaluate(line_sum)
  else:
    line_metrics = {}
  if paragraph_evaluator is not None:
    paragraph_metrics = paragraph_evaluator.evaluate(paragraph_sum)
  else:
    paragraph_metrics = {}
  return (['word', word_metrics], ['line', line_metrics],
          ['paragraph', paragraph_metrics])


def dict_add(input_tuples):
  """Aggregating dictionary accumulators by summing up respective fields."""
  new_dicts = [{}, {}, {}]
  for dicts in input_tuples:
    for i, ent_dict in enumerate(dicts):
      for k, v in ent_dict.items():
        new_dicts[i][k] = new_dicts[i].get(k, 0) + v
  return new_dicts


def metric_format(metric_groups):
  """Formatting the metrics in dict for printing."""
  outputs = []
  for ent, metrics in metric_groups:
    if metrics:
      output = '========= ' + ent + ' =========\n'
      kv_pairs = list(metrics.items())
      kv_pairs.sort()
      output += '\n'.join(f'{k}: {v}' for k, v in kv_pairs)
      outputs.append(output)
  return '\n\n'.join(outputs)


def main(argv: Sequence[str]) -> None:
  eval_start_time = time.time()

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  word_evaluator = evaluator.HierTextEvaluator(
      text_box_type=evaluator.TextBoxRep.POLY,
      evaluate_text=_E2E.value)
  if _EVAL_LINES.value:
    line_evaluator = evaluator.HierTextEvaluator(
        evaluate_text=_E2E.value)
  else:
    line_evaluator = None
  if _EVAL_PARAGRAPHS.value:
    paragraph_evaluator = evaluator.HierTextEvaluator()
  else:
    paragraph_evaluator = None

  running_mode = 'in_memory' if _NUM_WORKERS.value == 1 else 'multi_threading'
  options = PipelineOptions([
      '--runner=DirectRunner',
      f'--direct_num_workers={_NUM_WORKERS.value}',
      f'--direct_running_mode={running_mode}',
  ])
  with beam.Pipeline(options=options) as pipeline:
    _ = (
        pipeline
        | 'Read' >> beam.Create(load_annotations(_GT.value, _RESULT.value))
        | 'Parse' >> beam.Map(lambda x: parse_annotation_dict(
            x, _EVAL_LINES.value, _EVAL_PARAGRAPHS.value, _MASK_STRIDE.value))
        | 'Eval' >> beam.Map(lambda x: evaluate_one_image(
            *x, word_evaluator, line_evaluator, paragraph_evaluator))
        | 'Sum' >> beam.CombineGlobally(dict_add)
        | 'Compute-metrics' >> beam.Map(lambda x: compute_eval_metrics(
            *x, word_evaluator, line_evaluator, paragraph_evaluator))
        | 'Format-metrics' >> beam.Map(lambda x: metric_format(x))
        | 'Write' >> beam.io.WriteToText(_OUTPUT.value))

  eval_end_time = time.time()
  print(f'Evaluation took {eval_end_time - eval_start_time} secs.')

if __name__ == '__main__':
  flags.mark_flag_as_required('gt')
  flags.mark_flag_as_required('result')
  flags.mark_flag_as_required('output')
  app.run(main)
  
  # python3 eval.py --gt=validation.jsonl --result=sample.jsonl --output=score.txt --mask_stride=1 --eval_lines --eval_paragraphs
