from glob import glob
from tqdm import tqdm
import sys, os
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)
    parser.add_argument("--saved_name", type=str, required=True, default='res_1500pts.jsonl',
                        help="Saved name.")
    return parser.parse_args()


args = get_args_parser()
jsonl_list = glob('./res_per_img/*.jsonl')
final_results = {"annotations": []}
for jsonl_name in tqdm(jsonl_list):
    with open(jsonl_name, 'r') as fr:
        res = json.load(fr)
    fr.close()
    final_results['annotations'].append(res)

with open(args.saved_name, 'w') as fw:
    json.dump(final_results, fw, ensure_ascii=False)
