# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import copy
import numbers
import sys

import numpy as np
import random
from copy import deepcopy

import torchvision.transforms
from skimage import io
import os
from glob import glob
from typing import Tuple, List, Optional
from collections.abc import Sequence
import json
from pycocotools import mask as mask_utils
import time
from tqdm import tqdm
import cv2
from PIL import Image


import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize, InterpolationMode
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, rotate, gaussian_blur
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>",flag," dataset ",i+1,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"],': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"]
                for x in tmp_im_list
            ]
            print('-gt-',datasets[i]["name"],datasets[i]["gt_dir"],': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"],
                                "json_path": datasets[i].get('json_dir', None)
                                })

    return name_im_gt_list

def custom_collate_fn(batch_samples):
    imidx, image, label, shape = [], [], [], []
    para_masks, line_masks, word_masks, line2para_idx = [], [], [], []
    for sample in batch_samples:
        imidx.append(sample['imidx'])
        image.append(sample['image'].unsqueeze(0))
        label.append(sample['label'].unsqueeze(0))
        shape.append(sample['shape'].unsqueeze(0))
        line_masks.append(sample['line_masks'])
        if sample.get('paragraph_masks', None) is not None:
            para_masks.append(sample['paragraph_masks'])
            word_masks.append(sample['word_masks'])
            line2para_idx.append(sample['line2paragraph_index'])
    return {
        'imidx': torch.as_tensor(imidx), 'image': torch.cat(image), 'label': torch.cat(label), 'shape': torch.cat(shape),
        'paragraph_masks': para_masks, 'line_masks': line_masks, 'word_masks': word_masks, 'line2paragraph_index': line2para_idx
    }

def create_dataloaders(
        name_im_gt_list, my_transforms=[], batch_size=1, training=False, hier_det=False, collate_fn=None
):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size >= 4:
        num_workers_ = 4
    if batch_size >= 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform = transforms.Compose(my_transforms),
                hier_det=hier_det
            )
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        dataloader = DataLoader(
            gos_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=num_workers_,
            pin_memory=True,
            prefetch_factor=6,
            collate_fn=collate_fn if hier_det else None
        )
        gos_dataloaders = dataloader
        gos_datasets = gos_dataset
    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                eval_ori_resolution=True
            )
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class ResizeLongestSide_ToTensor(object):
    def __init__(self, target_length=1024):
        self.target_length = target_length

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]

    def __call__(self, sample):
        # image: np.array, [h, w, c]
        image, shape = sample['image'], sample['shape']
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        target_size = self.get_preprocess_shape(image.shape[1], image.shape[2], self.target_length)
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), target_size, mode='bilinear'), dim=0)
        return {'image':image, 'shape':torch.tensor(target_size)}


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, hier_det=False):
        # if hier_det is True, the json file for word, line, paragraph
        # detection should be loaded.
        self.transform = transform
        self.dataset = {}
        im_name_list = []  # image name
        im_path_list = []  # im path
        gt_path_list = []  # gt path

        assert len(name_im_gt_list) == 1
        name_im_gt_list = name_im_gt_list[0]
        im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list["im_ext"])[0] for x in name_im_gt_list["im_path"]])
        im_path_list.extend(name_im_gt_list["im_path"])
        gt_path_list.extend(name_im_gt_list["gt_path"])

        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset_name = name_im_gt_list["dataset_name"]
        self.hier_det = hier_det
        if hier_det:
            json_path = name_im_gt_list.get('json_path', None)
            assert json_path is not None, "Please check settings."
            load_start = time.time()
            with open(json_path, 'r') as fw:
                annotations = json.load(fw)['annotations']
            self.dataset["annotations"] = {anns['image_id']: anns for anns in annotations}
            print(f"processed {len(self.dataset['annotations'])} samples from {json_path} in {time.time()-load_start:.2f} seconds.")
            del annotations

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im_name = self.dataset["im_name"][idx]
        im = io.imread(im_path)
        gt_ori = io.imread(gt_path)
        if 'TextSeg' in self.dataset_name:
            gt = (gt_ori == 100).astype(np.uint8) * 255
        elif 'COCO_TS' in self.dataset_name:
            gt = (gt_ori > 0).astype(np.uint8) * 255
        else:
            gt = (gt_ori > 127).astype(np.uint8) * 255  # for TotalText, HierText: 0 or 255
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt.astype(np.float32),
            "shape": torch.tensor(im.shape[:2]),
        }

        if self.hier_det:
            raise NotImplementedError
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = torch.unsqueeze(torch.from_numpy(gt), 0)
            if 'TextSeg' in self.dataset_name:
                ignore_mask = (gt_ori == 255).astype(np.uint8) * 255
                sample["ignore_mask"] = torch.unsqueeze(torch.from_numpy(ignore_mask), 0)
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample


eval_transforms = [
    ResizeLongestSide_ToTensor(),
]
