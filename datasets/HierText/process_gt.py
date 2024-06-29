import copy
import math
import io
from pycocotools.coco import COCO
from pycocotools import mask as maskutils
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import json
import pyclipper
from shapely.geometry import Polygon


def shrink_polygon(polygon, shrink_ratio):
    # from DB (https://github.com/MhLiao/DB)
    np_poly = np.array(polygon)
    polygon_shape = Polygon(np_poly)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in np_poly]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        return polygon
    else:
        shrinked = shrinked[0]
        return shrinked


''' 
An example script for processing the original gt.
Step(1) filter empty paragraphs to avoid training error 
'''
ANN_PATH = yours_jsonl_path
with open(ANN_PATH, 'r') as f:
    anns = json.load(f)
new_json = dict()
new_json['info'] = anns['info']
new_annotations = []
for old_ann in tqdm(anns['annotations']):
    # per image
    paragraphs = []
    old_paras = old_ann['paragraphs']
    assert len(old_paras) > 0
    for old_para in old_paras:
        if len(old_para['lines']) > 0:
            paragraphs.append(old_para)
    if len(paragraphs) > 0:
        new_ann = copy.deepcopy(old_ann)
        new_ann['paragraphs'] = paragraphs
        new_annotations.append(new_ann)
    else:
        continue
new_json.update(annotations=new_annotations)


''' 
Step(2) shrink the word polygon and reorganize the dict
'''
new_all_dict = {'info': new_json['info']}
new_annotations = []
old_annotations = new_json.pop('annotations')

for old_anno in tqdm(old_annotations):
    new_dict = old_anno
    paras = new_dict.pop('paragraphs')  # [{pa1}, {pa2}, ...]
    w = new_dict['image_width']
    h = new_dict['image_height']
    paragraph_masks, line_masks, word_masks = [], [], []
    line2paragraph_index, word2line_index = [], []
    para_legible, line_legible = [], []
    lineindex = 0
    for para_idx, para in enumerate(paras):
        assert len(para['vertices']) > 2
        paragraph_masks.append(para['vertices'])
        para_legible.append(para['legible'])
        for line in para['lines']:
            line_masks.append(line['vertices'])
            line2paragraph_index.append(para_idx)
            line_legible.append(line['legible'])
            word_mask_per_line = []
            for word in line['words']:
                shr_word_mask = shrink_polygon(word['vertices'], 0.4)
                assert len(shr_word_mask) > 2
                word_mask_per_line.append(shr_word_mask)
            word_masks.append(word_mask_per_line)
            lineindex += 1
    new_dict.update(
        paragraph_masks=paragraph_masks,
        line_masks=line_masks,
        word_masks=word_masks,
        line2paragraph_index=line2paragraph_index,
        para_legible=para_legible,
        line_legible=line_legible
    )
    assert len(line_masks) == len(word_masks)
    new_annotations.append(new_dict)
new_all_dict.update(annotations=new_annotations)
print(len(new_annotations))
with open('train_shrink_vert.json', 'w', encoding='utf-8') as fw:
    json.dump(new_all_dict, fw)
