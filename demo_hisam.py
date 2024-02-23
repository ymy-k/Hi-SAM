import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor
import glob
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", action='store_true',
                        help="If False, only text stroke segmentation.")

    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--patch_mode', action='store_true')

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')

    return parser.parse_args()


def patchify(image: np.array, patch_size: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_num, w_num = h//patch_size, w//patch_size
    h_remain, w_remain = h%patch_size, w%patch_size
    row, col = h_num + int(h_remain>0), w_num + int(w_remain>0)
    h_slices = [[r * patch_size, (r + 1) * patch_size] for r in range(h_num)]
    if h_remain:
        h_slices = h_slices + [[h - h_remain, h]]
    h_slices = np.tile(h_slices, (1, col)).reshape(-1, 2).tolist()
    w_slices = [[i * patch_size, (i + 1) * patch_size] for i in range(w_num)]
    if w_remain:
        w_slices = w_slices + [[w-w_remain, w]]
    w_slices = w_slices * row
    assert len(w_slices) == len(h_slices)
    for idx in range(0, len(w_slices)):
        # from left to right, then from top to bottom
        patch_list.append(image[h_slices[idx][0]:h_slices[idx][1], w_slices[idx][0]:w_slices[idx][1], :])
    return patch_list, row, col


def unpatchify(patches, row, col):
    # return np.array
    whole = [np.concatenate(patches[r*col : (r+1)*col], axis=1) for r in range(row)]
    whole = np.concatenate(whole, axis=0)
    return whole


def patchify_sliding(image: np.array, patch_size: int=512, stride: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j+patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i+patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])

    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2  # (h, w)
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]

    return whole_logits


def show_points(coords, ax, marker_size=200):
    ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_res(masks, scores, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_hi_masks(masks, word_masks, input_points, filename, image, scores):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, (line_para_masks, word_mask, hi_score, point) in enumerate(zip(masks, word_masks, scores, input_points)):
        line_mask = line_para_masks[0]
        para_mask = line_para_masks[1]
        show_mask(para_mask, plt.gca(), color=np.array([255 / 255, 144 / 255, 30 / 255, 0.5]))
        show_mask(line_mask, plt.gca())
        word_mask = word_mask[0].astype(np.uint8)
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        select_word = None
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
            if cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False) >= 0:
                select_word = pts
                break
        if select_word is not None:
            word_mask = cv2.fillPoly(np.zeros(word_mask.shape), [select_word], 1)
            show_mask(word_mask, plt.gca(), color=np.array([30 / 255, 255 / 255, 144 / 255, 0.5]))
        show_points(point, plt.gca())
        print(f'point {i}: line {hi_score[1]}, para {hi_score[2]}')

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
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
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    predictor = SamPredictor(hisam)

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm(args.input, disable=not args.output):
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = os.path.basename(path).split('.')[0] + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(args.input) == 1
            out_filename = args.output

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.patch_mode:
            ori_size = image.shape[:2]
            patch_list, h_slice_list, w_slice_list = patchify_sliding(image, 512, 384)  # sliding window config
            mask_512 = []
            for patch in tqdm(patch_list):
               predictor.set_image(patch)
               m, hr_m, score, hr_score = predictor.predict(multimask_output=False, return_logits=True)
               assert hr_m.shape[0] == 1  # high-res mask
               mask_512.append(hr_m[0])
            mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
            assert mask_512.shape[-2:] == ori_size
            mask = mask_512
            mask = mask > predictor.model.mask_threshold
            save_binary_mask(mask, out_filename)
        else:
            predictor.set_image(image)
            if args.hier_det:
                input_point = np.array([[125, 275]])  # for demo/img293.jpg
                input_label = np.ones(input_point.shape[0])
                mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
                show_hi_masks(hi_mask, word_mask, input_point, out_filename, image, hi_iou)
            else:
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                save_binary_mask(hr_mask, out_filename)