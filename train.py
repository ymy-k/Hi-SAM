import os
import argparse
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import time
import datetime

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.loss import loss_masks, loss_hi_masks, loss_iou_mse, loss_hi_iou_mse
from hi_sam.data.dataloader import get_im_gt_name_dict, create_dataloaders, eval_transforms, custom_collate_fn
from hi_sam.evaluation import Evaluator
import utils.misc as misc
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--output", type=str, default="work_dirs/", 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument("--train_datasets", type=str, nargs='+', default=['totaltext_train'])
    parser.add_argument("--val_datasets", type=str, nargs='+', default=['totaltext_test'])
    parser.add_argument("--hier_det", action='store_true',
                        help="If False, only text stroke segmentation.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_charac_mask_decoder_name', default=["mask_decoder"], type=str, nargs='+')
    parser.add_argument('--lr_charac_mask_decoder', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=40, type=int)
    parser.add_argument('--max_epoch_num', default=50, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--valid_period', default=1, type=int)
    parser.add_argument('--model_save_fre', default=90, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')

    return parser.parse_args()


def main(train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        raise NotImplementedError

    print("--- create valid dataloader ---")
    valid_datasets_names = [val_ds["name"] for val_ds in valid_datasets]
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        my_transforms=eval_transforms,
        batch_size=args.batch_size_valid,
        training=False
    )
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    model = model_registry[args.model_type](args=args)
    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module
 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        raise NotImplementedError
    else:
        print("restore model from:", args.checkpoint)
        evaluate(args, model, valid_dataloaders, valid_datasets_names)


def inference_on_dataset(model, data_loader, data_name, evaluator, args):
    print("Start inference on {}, {} batches".format(data_name, len(data_loader)))
    num_devices = misc.get_world_size()
    total = len(data_loader)
    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    start_data_time = time.perf_counter()
    for idx_val, data_val in enumerate(data_loader):
        inputs_val, labels_ori = data_val['image'], data_val['ori_label']
        ignore_mask = data_val.get('ignore_mask', None)
        if torch.cuda.is_available():
            labels_ori = labels_ori.cuda()
        batched_input = []
        for b_i in range(len(inputs_val)):
            dict_input = dict()
            dict_input['image'] = inputs_val[b_i].to(model.device).contiguous()
            dict_input['original_size'] = labels_ori[b_i].shape[-2:]
            batched_input.append(dict_input)

        total_data_time += time.perf_counter() - start_data_time
        if idx_val == num_warmup:
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0

        start_compute_time = time.perf_counter()
        with torch.no_grad():
            up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output = model(
                batched_input, multimask_output=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time

        start_eval_time = time.perf_counter()
        evaluator.process(up_masks, hr_masks, labels_ori, ignore_mask)
        total_eval_time += time.perf_counter() - start_eval_time

        iters_after_start = idx_val + 1 - num_warmup * int(idx_val >= num_warmup)
        data_seconds_per_iter = total_data_time / iters_after_start
        compute_seconds_per_iter = total_compute_time / iters_after_start
        eval_seconds_per_iter = total_eval_time / iters_after_start
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
        if (idx_val+1) % 20 == 0:
            eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx_val - 1)))
            print(
                f"Inference done [{idx_val + 1}]/[{total}]. ",
                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. ",
                f"Inference: {compute_seconds_per_iter:.4f} s/iter. ",
                f"Eval: {eval_seconds_per_iter:.4f} s/iter. ",
                f"Total: {total_seconds_per_iter:.4f} s/iter. ",
                f"ETA={eta}"
            )
        start_data_time = time.perf_counter()

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    print(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results


def evaluate(args, model, valid_dataloaders, valid_datasets_names):
    model.eval()
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        valid_dataset_name = valid_datasets_names[k]
        evaluator = Evaluator(valid_dataset_name, args, True)
        print('============================')
        results_k = inference_on_dataset(model, valid_dataloader, valid_dataset_name, evaluator, args)
        print("Evaluation results for {}:".format(valid_dataset_name))
        for task, res in results_k.items():
            if '_hr' not in task:
                print(f"copypaste: {task}={res}, {task}_hr={results_k[task+'_hr']}")
        print('============================')
        test_stats.update({valid_dataset_name: results_k})

    return test_stats


if __name__ == "__main__":

    # train
    totaltext_train = {
        "name": "TotalText-train",
        "im_dir": "./datasets/TotalText/Images/Train",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Train",
        "im_ext": ".jpg",
        "gt_ext": ".jpg",
    }
    hiertext_train = {
        "name": "HierText-train",
        "im_dir": "./datasets/HierText/train",
        "gt_dir": "./datasets/HierText/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png",
        "json_dir": "./datasets/HierText/train_shrink_vert.json"
    }
    textseg_train = {
        "name": "TextSeg-train",
        "im_dir": "./datasets/TextSeg/train_images",
        "gt_dir": "./datasets/TextSeg/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/COCO_TS_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_hier = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/hier-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_tt = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/tt-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_textseg = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/textseg-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    train_dataset_map = {
        'totaltext_train': totaltext_train,
        'hiertext_train': hiertext_train,
        'textseg_train': textseg_train,
        'cocots_train': cocots_train,
        'cocots_train_hier': cocots_train_hier,
        'cocots_train_tt': cocots_train_tt,
        'cocots_train_textseg': cocots_train_textseg,
    }

    # validation and test
    totaltext_test = {
        "name": "TotalText-test",
        "im_dir": "./datasets/TotalText/Images/Test",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Test",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    hiertext_val = {
        "name": "HierText-val",
        "im_dir": "./datasets/HierText/validation",
        "gt_dir": "./datasets/HierText/validation_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    hiertext_test = {
        "name": "HierText-test",
        "im_dir": "./datasets/HierText/test",
        "gt_dir": "./datasets/HierText/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_val = {
        "name": "TextSeg-val",
        "im_dir": "./datasets/TextSeg/val_images",
        "gt_dir": "./datasets/TextSeg/val_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_test = {
        "name": "TextSeg-test",
        "im_dir": "./datasets/TextSeg/test_images",
        "gt_dir": "./datasets/TextSeg/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    val_dataset_map = {
        'totaltext_test': totaltext_test,
        'hiertext_val': hiertext_val,
        'hiertext_test': hiertext_test,
        'textseg_val': textseg_val,
        'textseg_test': textseg_test,
    }

    train_datasets = []
    val_datasets = []
    args = get_args_parser()

    for ds_name in args.train_datasets:
        train_datasets.append(train_dataset_map[ds_name])
    for ds_name in args.val_datasets:
        val_datasets.append(val_dataset_map[ds_name])

    main(train_datasets, val_datasets, args)
