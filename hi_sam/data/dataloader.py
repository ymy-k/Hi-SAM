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


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        masks = sample.get('line_masks', None)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if len(label.shape) == 2:
            label = torch.from_numpy(label).unsqueeze(0)
        else:
            raise NotImplementedError
        sample['image'] = image
        sample['label'] = label
        if masks is not None:
            sample['line_masks'] = torch.from_numpy(masks).permute(2, 0, 1)

            if sample.get('paragraph_masks', None) is not None:
                sample['paragraph_masks'] = torch.from_numpy(sample['paragraph_masks']).permute(2, 0, 1)
                sample['word_masks'] = torch.from_numpy(sample['word_masks']).permute(2, 0, 1)
        return sample


class LargeScaleJitter(object):
    """
            implementation of large scale jitter from copy_paste
            https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
        """
    def __init__(self, output_size=1024, aug_scale_min=0.5, aug_scale_max=2.0):
        self.desired_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, sample):
        image, label, image_size = sample['image'], sample['label'], sample['shape']
        masks = sample.get('line_masks', None)
        image_size = image_size.numpy()

        random_scale = np.random.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()
        scale = np.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().astype(np.int64)  # h, w

        scaled_image = cv2.resize(image, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
        scaled_label = cv2.resize(label, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))
        margin_h = max(scaled_size[0] - crop_size[0], 0)
        margin_w = max(scaled_size[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        scaled_image = scaled_image[crop_y1:crop_y2, crop_x1:crop_x2, :]
        scaled_label = scaled_label[crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.shape[0], 0)
        padding_w = max(self.desired_size - scaled_image.shape[1], 0)
        image = cv2.copyMakeBorder(scaled_image,0,padding_h,0,padding_w,cv2.BORDER_CONSTANT,value=(128,128,128))
        label = cv2.copyMakeBorder(scaled_label,0,padding_h,0,padding_w,cv2.BORDER_CONSTANT,value=0)
        sample.update(image=image, label=label, shape=torch.tensor(image.shape[:2]))

        if masks is not None:
            masks = cv2.resize(masks, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            masks = masks[crop_y1:crop_y2, crop_x1:crop_x2, :]
            masks = cv2.copyMakeBorder(masks, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            sample['line_masks'] = masks

            masks = sample.get('paragraph_masks', None)
            if masks is not None:
                masks = cv2.resize(masks, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                masks = masks[crop_y1:crop_y2, crop_x1:crop_x2, :]
                masks = cv2.copyMakeBorder(masks, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                sample['paragraph_masks'] = masks

                masks = sample['word_masks']
                masks = cv2.resize(masks, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                masks = masks[crop_y1:crop_y2, crop_x1:crop_x2, :]
                masks = cv2.copyMakeBorder(masks, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                sample['word_masks'] = masks

        return sample


class ColorJitter(object):
    def __init__(self, brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_params(self,
                    brightness: Optional[List[float]],
                    contrast: Optional[List[float]],
                    saturation: Optional[List[float]],
                    hue: Optional[List[float]]
                    ):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, sample):
        image = sample['image']  # np.array, hwc
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self._get_params(self.brightness, self.contrast, self.saturation, self.hue)
        for idx in fn_idx:
            if idx == 0 and brightness_factor is not None:
                image = adjust_brightness(image, brightness_factor)
            elif idx == 1 and contrast_factor is not None:
                image = adjust_contrast(image, contrast_factor)
            elif idx == 2 and saturation_factor is not None:
                image = adjust_saturation(image, saturation_factor)
            elif idx == 3 and hue_factor is not None:
                image = adjust_hue(image, hue_factor)
        sample['image'] = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float32)
        return sample


class RandomBlur(object):
    def __init__(self, p=0.3, gaussian_kernel_size=[5, 7, 9, 11, 13, 15]):
        self.p = p
        self.gaussian_kernel_size = gaussian_kernel_size  # _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.gaussian_kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            g_kernel = random.choice(self.gaussian_kernel_size)
            image = cv2.GaussianBlur(image, (g_kernel, g_kernel), sigmaX=0, sigmaY=0)
            sample['image'] = image
            return sample
        else:
            return sample


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomRotate(object):
    def __init__(self, angle=180):
        self.angle = _setup_angle(angle, name="angle", req_sizes=(2, ))

    def apply(self, img, rm_img, bound_w, bound_h, interp=None, border_value=None):
        interp = interp if interp is not None else cv2.INTER_LINEAR
        return cv2.warpAffine(img, rm_img, (bound_w, bound_h), flags=interp, borderValue=border_value)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        masks = sample.get('line_masks', None)

        h, w = image.shape[:2]
        angle = np.random.uniform(self.angle[0], self.angle[1])
        center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        bound_w, bound_h = np.rint(
            [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
        ).astype(int)
        rm_image = create_rotationmatrix(center, angle, bound_w, bound_h, offset=-0.5)

        image = self.apply(image, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, (128,128,128))
        label = self.apply(label, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
        assert image.shape[:2] == label.shape[:2]
        sample['image'] = image
        sample['label'] = label
        sample['shape'] = torch.tensor(image.shape[:2])

        if masks is not None:
            masks = self.apply(masks, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]  # to avoid channel disappearing when there is only one mask
            sample['line_masks'] = masks

            masks = sample.get('paragraph_masks', None)
            if masks is not None:
                # masks = sample['paragraph_masks']
                masks = self.apply(masks, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                sample['paragraph_masks'] = masks

                masks = sample['word_masks']
                masks = self.apply(masks, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                sample['word_masks'] = masks

        return sample


def create_rotationmatrix(center, angle, bound_w, bound_h, offset=0):
    center_offset = (center[0] + offset, center[1] + offset)
    rm = cv2.getRotationMatrix2D(tuple(center_offset), angle, 1)
    rot_im_center = cv2.transform(center[None, None, :] + offset, rm)[0, 0, :]
    new_center = np.array([bound_w / 2, bound_h / 2]) + offset - rot_im_center
    rm[:, 2] += new_center
    return rm


def get_one_mask(vertices, w, h):
  mask = np.zeros((h, w), dtype=np.float32)
  mask = cv2.fillPoly(mask, [np.array(vertices)], [1])
  return mask


def get_word_mask(vertices, w, h):
  mask = np.zeros((h, w), dtype=np.float32)
  for ver in vertices:
    mask = cv2.fillPoly(mask, [np.array(ver)], [1])
  return mask


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
            anns = copy.deepcopy(self.dataset["annotations"][im_name])
            w, h = anns['image_width'], anns['image_height']
            line_num = len(anns['line_masks'])
            if 'HierText' in self.dataset_name:
                if line_num > 10:
                    select_line_idx = random.sample(range(line_num), 10)
                    select_line_idx.sort()
                    masks = [get_one_mask(anns['line_masks'][l_idx], w, h) for l_idx in select_line_idx]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['line_masks'] = masks
                    masks = [get_word_mask(anns['word_masks'][l_idx], w, h) for l_idx in select_line_idx]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['word_masks'] = masks
                    line2para_index = anns['line2paragraph_index']  # ordered
                    line2para_index = [line2para_index[l_idx] for l_idx in select_line_idx]
                    line2para_index_set = set(line2para_index)
                    masks = [get_one_mask(anns['paragraph_masks'][l2p_idx], w, h) for l2p_idx in line2para_index_set]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['paragraph_masks'] = masks
                    new_l2p_index = []
                    for ii, jj in enumerate(line2para_index_set):
                        new_l2p_index.extend([ii] * line2para_index.count(jj))
                    sample['line2paragraph_index'] = torch.tensor(new_l2p_index)
                else:
                    sample['line2paragraph_index'] = torch.tensor(anns['line2paragraph_index'])
                    masks = [get_one_mask(ver, w, h) for ver in anns['paragraph_masks']]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['paragraph_masks'] = masks
                    # value: 0,1,
                    masks = [get_one_mask(ver, w, h) for ver in anns['line_masks']]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['line_masks'] = masks

                    masks = [get_word_mask(ver, w, h) for ver in anns['word_masks']]
                    masks = np.array(masks).transpose((1, 2, 0))
                    sample['word_masks'] = masks
            else:
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


train_transforms = [
    ColorJitter(),  # image: np.uint8->np.float32
    RandomRotate(),
    LargeScaleJitter(),
    ToTensor()
]

eval_transforms = [
    ResizeLongestSide_ToTensor(),
]
