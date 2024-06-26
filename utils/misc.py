# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import random 
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import functools

import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

import colorsys
import torch.nn.functional as F

import cv2
from PIL import Image
import matplotlib.pyplot as plt

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        # args.rank = int(os.environ["RANK"])
        # args.world_size = int(os.environ['WORLD_SIZE'])
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])

        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        # print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def sample_foreground_points(batch_labels, batch_para_masks, batch_line_masks,
                             batch_word_masks, batch_line2para_idx, k=20, np_per_line=2):
    fg_points, select_line_masks, select_para_masks, select_word_masks = [], [], [], []
    only_line = False
    if len(batch_line_masks) > 0 and len(batch_word_masks) == 0:
        only_line = True
    dev = batch_labels.device
    h, w = batch_labels.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(batch_labels)
    x = x.to(batch_labels)

    for b_i in range(len(batch_labels)):
        line_masks = batch_line_masks[b_i].to(dev)
        # filter line masks without intersection with stroke mask
        intersection = (
                (batch_labels[b_i] * line_masks) > 127
        )  # (nl, h, w)
        intersec_num = torch.sum(intersection.flatten(1), dim=1)
        keep_index = torch.nonzero(intersec_num).reshape(-1)
        keep_num = keep_index.numel()

        intersection = intersection[keep_index]
        intersec_num = intersec_num[keep_index]
        line_masks = line_masks[keep_index]
        if not only_line:
            line2para_idx = batch_line2para_idx[b_i][keep_index]
            para_masks = batch_para_masks[b_i][line2para_idx].to(dev)
            word_masks_per_line = batch_word_masks[b_i][keep_index].to(dev)
        if keep_num == 0:
            sampled_xy = torch.tensor([[w / 2, h / 2]], dtype=torch.float32, device=dev)
            fg_points.append(sampled_xy)
            line_masks = torch.zeros((1, h, w)).to(line_masks)
            select_line_masks.append(line_masks.unsqueeze(1))
            if not only_line:
                para_masks = torch.zeros((1, h, w)).to(para_masks)
                select_para_masks.append(para_masks.unsqueeze(1))
                word_masks_per_line = torch.zeros((1, h, w)).to(word_masks_per_line)
                select_word_masks.append(word_masks_per_line.unsqueeze(1))
            continue
        x_idx = torch.masked_select(x, intersection)
        y_idx = torch.masked_select(y, intersection)
        del intersection
        intersec_cumsum = torch.cumsum(intersec_num, dim=0)
        start_idx = intersec_cumsum - intersec_num
        np_per_line = min(np_per_line, intersec_num.min().item())
        line_masks = torch.repeat_interleave(line_masks, np_per_line, dim=0)
        if not only_line:
            para_masks = torch.repeat_interleave(para_masks, np_per_line, dim=0)
            word_masks_per_line = torch.repeat_interleave(word_masks_per_line, np_per_line, dim=0)
        pos_idx = np.ravel([
            random.sample(range(start_idx[p_i].item(), intersec_cumsum[p_i].item()), np_per_line)
            for p_i in range(keep_num)
        ])

        sampled_x = x_idx[pos_idx]
        sampled_y = y_idx[pos_idx]
        sampled_xy = torch.cat((sampled_x[:, None], sampled_y[:, None]), dim=1)  # (k, 2)
        # if sampled_xy.shape[0] == 0:
        #     sampled_xy = torch.tensor([[w/2, h/2]], dtype=torch.float32, device=dev)
        #     para_masks = torch.zeros((1, h, w)).to(para_masks)
        #     line_masks = torch.zeros((1, h, w)).to(line_masks)
        #     word_masks_per_line = torch.zeros((1, h, w)).to(word_masks_per_line)

        perm_index = torch.randperm(sampled_xy.shape[0])
        sampled_xy = sampled_xy[perm_index]
        line_masks = line_masks[perm_index]
        if not only_line:
            para_masks = para_masks[perm_index]
            word_masks_per_line = word_masks_per_line[perm_index]

        fg_points.append(sampled_xy)
        select_line_masks.append(line_masks.unsqueeze(1))
        del line_masks
        if not only_line:
            select_para_masks.append(para_masks.unsqueeze(1))  # (k, 1, h, w)
            select_word_masks.append(word_masks_per_line.unsqueeze(1))
            del para_masks
            del word_masks_per_line
    select_line_masks = torch.cat(select_line_masks, dim=0)
    if not only_line:
        select_para_masks = torch.cat(select_para_masks, dim=0)
        select_word_masks = torch.cat(select_word_masks, dim=0)

    if not only_line:
        return fg_points, select_para_masks, select_line_masks, select_word_masks
    else:
        return fg_points, None, select_line_masks, None