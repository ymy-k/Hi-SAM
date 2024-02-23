import copy
import math
import time

import skimage
import torch
import io
from pycocotools.coco import COCO
from pycocotools import mask as maskutils
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import sys, os
import matplotlib.pyplot as plt
import cv2
import random
import json
from PIL import Image, TarIO
import contextlib
import torch.nn.functional as F
import pyclipper
from shapely.geometry import Polygon
import shutil
import skimage


os.makedirs('train_images', exist_ok=True)
os.makedirs('train_gt', exist_ok=True)
os.makedirs('val_images', exist_ok=True)
os.makedirs('val_gt', exist_ok=True)
os.makedirs('test_images', exist_ok=True)
os.makedirs('test_gt', exist_ok=True)

with open('split.json') as f_json:
    split_info = json.load(f_json)
train_img_list = split_info['train']
val_img_list = split_info['val']
test_img_list = split_info['test']

for im_id in train_img_list:
    shutil.copy(os.path.join('image', im_id+'.jpg'), os.path.join('train_images', im_id+'.jpg'))
    shutil.copy(os.path.join('semantic_label', im_id + '_maskfg.png'), os.path.join('train_gt', im_id + '.png'))
for im_id in val_img_list:
    shutil.copy(os.path.join('image', im_id+'.jpg'), os.path.join('val_images', im_id+'.jpg'))
    shutil.copy(os.path.join('semantic_label', im_id + '_maskfg.png'), os.path.join('val_gt', im_id + '.png'))
for im_id in test_img_list:
    shutil.copy(os.path.join('image', im_id+'.jpg'), os.path.join('test_images', im_id+'.jpg'))
    shutil.copy(os.path.join('semantic_label', im_id + '_maskfg.png'), os.path.join('test_gt', im_id + '.png'))
