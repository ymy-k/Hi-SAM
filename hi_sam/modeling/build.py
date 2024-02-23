# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import pdb

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
import copy
import os

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, HiDecoder
from .prompt_encoder import PromptEncoder
from .hi_sam import HiSam
from .transformer import TwoWayTransformer
from .modal_aligner import ModalAligner


def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        args=args,
    )


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        args=args,
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        args=args,
    )


model_registry = {
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    args,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # 64

    checkpoint = args.checkpoint

    model = HiSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        modal_aligner=ModalAligner(
            prompt_embed_dim,
            attn_layers=args.attn_layers,
            prompt_len=args.prompt_len
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if args.hier_det:
        model.hier_det = True
        model.hi_decoder = HiDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        dict_keys = state_dict.keys()
        if 'optimizer' in dict_keys or 'lr_scheduler' in dict_keys or 'epoch' in dict_keys:
            state_dict = state_dict['model']
        dict_keys = state_dict.keys()

        # check whether to use the paras. of SAM's mask decoder to initialize H-Decoder
        if args.hier_det:
            contain_hi_decoder = False
            for key in dict_keys:
                if 'hi_decoder' in key:
                    contain_hi_decoder = True
                    break
            if not contain_hi_decoder:
                ckpt_dir = os.path.dirname(checkpoint)
                with open(os.path.join('pretrained_checkpoint', args.model_type+'_maskdecoder.pth'), "rb") as f:
                    mask_decoder_dict = torch.load(f)
                for key, value in mask_decoder_dict.items():
                    new_key = key.replace('mask_decoder', 'hi_decoder')
                    state_dict[new_key] = value

        # load SAM's ViT backbone paras.
        if args.model_type == 'vit_b':
            sam_path = os.path.join('pretrained_checkpoint', 'sam_vit_b_01ec64.pth')
        elif args.model_type == 'vit_l':
            sam_path = os.path.join('pretrained_checkpoint', 'sam_vit_l_0b3195.pth')
        elif args.model_type == 'vit_h':
            sam_path = os.path.join('pretrained_checkpoint', 'sam_vit_h_4b8939.pth')
        with open(sam_path, "rb") as f:
            sam_dict = torch.load(f)
        for key, value in sam_dict.items():
            if key not in dict_keys:
                state_dict[key] = value
        del sam_dict

        info = model.load_state_dict(state_dict, strict=False)
        print(info)

    return model
