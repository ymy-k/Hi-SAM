import json
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
import glob
from tqdm import tqdm
from PIL import Image
import random
from utils import utilities
from shapely.geometry import Polygon
import pyclipper
import datetime
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--existing_fgmask_input", type=str, default='./datasets/HierText/val_fgmask/',
                        help="A file or directory of foreground masks.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--use_fgmask", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_out_file", type=str, default='./hiertext_eval/res_per_img',
                        help="A file or directory to save results per image.")
    parser.add_argument('--total_points', default=1500, type=int, help='The number of foreground points')
    parser.add_argument('--batch_points', default=100, type=int, help='The number of points per batch')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)
    return parser.parse_args()


def show_points(coords, ax, marker_size=40):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_hi_masks(masks, filename, image):
    plt.figure(figsize=(15, 15), dpi=200)
    plt.imshow(image)

    for i, hi_mask in enumerate(masks):
        hi_mask = hi_mask[0]
        show_mask(hi_mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_binary_mask(mask: np.array, filename):
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0].astype(np.uint8)*255
    elif len(mask.shape) == 2:
        mask = mask.astype(np.uint8)*255
    else:
        raise NotImplementedError
    mask = Image.fromarray(mask)
    mask.save(filename)


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


if __name__ == '__main__':
    args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")
    amg = AutoMaskGenerator(hisam)
    none_num = 0

    if args.eval:
        os.makedirs(args.eval_out_file, exist_ok=True)
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm(args.input, disable=not args.output):
        img_id = os.path.basename(path).split('.')[0]
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = img_id + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(args.input) == 1
            out_filename = args.output

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
        img_h, img_w = image.shape[:2]

        if args.use_fgmask:
            fgmask_path = os.path.join(args.existing_fgmask_input, img_id+'.png')
            fgmask = skimage.io.imread(fgmask_path)
            amg.set_fgmask(fgmask)

        amg.set_image(image)

        masks, scores, affinity = amg.predict(
            from_low_res=False,
            fg_points_num=args.total_points,
            batch_points_num=args.batch_points,
            score_thresh=0.5,
            nms_thresh=0.5,
        )  # only return word masks here for evaluation

        if args.eval:
            if masks is None:
                lines = [{'words': [{'text': '', 'vertices': [[0,0],[1,0],[1,1],[0,1]]}], 'text': ''}]
                paragraphs = [{'lines': lines}]
                result = {
                    'image_id': img_id,
                    "paragraphs": paragraphs
                }
                none_num += 1
            else:
                masks = (masks[:, 0, :, :]).astype(np.uint8)  # word masks, (n, h, w)
                lines = []
                line_indices = []
                for index, mask in enumerate(masks):
                    line = {'words': [], 'text': ''}
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for cont in contours:
                        epsilon = 0.002 * cv2.arcLength(cont, True)
                        approx = cv2.approxPolyDP(cont, epsilon, True)
                        points = approx.reshape((-1, 2))
                        if points.shape[0] < 4:
                            continue
                        pts = unclip(points)
                        if len(pts) != 1:
                            continue
                        pts = pts[0].astype(np.int32)
                        if Polygon(pts).area < 32:
                            continue
                        pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
                        pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
                        cnt_list = pts.tolist()
                        line['words'].append({'text': '', 'vertices': cnt_list})
                    if line['words']:
                        lines.append(line)
                        line_indices.append(index)

                line_grouping = utilities.DisjointSet(len(line_indices))
                affinity = affinity[line_indices][:, line_indices]
                for i1, i2 in zip(*np.where(affinity > args.layout_thresh)):
                    line_grouping.union(i1, i2)
                line_groups = line_grouping.to_group()
                paragraphs = []
                for line_group in line_groups:
                    paragraph = {'lines': []}
                    for id_ in line_group:
                        paragraph['lines'].append(lines[id_])
                    if paragraph:
                        paragraphs.append(paragraph)
                result = {
                    'image_id': img_id,
                    "paragraphs": paragraphs
                }
            with open(os.path.join(args.eval_out_file, img_id+'.jsonl'), 'w', encoding='utf-8') as fw:
                json.dump(result, fw)
            fw.close()

    if args.eval:
        print(f'{none_num} images without predictions.')
