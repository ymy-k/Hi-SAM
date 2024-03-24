# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import pdb

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from functools import partial
import copy
import os

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, HiDecoder
from .prompt_encoder import PromptEncoder
from .hi_sam import HiSam
from .transformer import TwoWayTransformer
from .modal_aligner import ModalAligner

from .efficient_hi_sam import EfficientHiSam
from .efficient_sam.efficient_sam_encoder import ImageEncoderViT as eImageEncoderViT
from .efficient_sam.efficient_sam_decoder import MaskDecoder as eMaskDecoder
from .efficient_sam.efficient_sam_decoder import HiDecoder as eHiDecoder
from .efficient_sam.efficient_sam_decoder import PromptEncoder as ePromptEncoder
from .efficient_sam.two_way_transformer import TwoWayTransformer as eTwoWayTransformer


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


def build_efficient_sam_vit_s(args):
    return _build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        args=args
    )
    

def build_efficient_sam_vit_t(args):
    return _build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        args=args
    )


model_registry = {
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_s": build_efficient_sam_vit_s,
    "vit_t": build_efficient_sam_vit_t,
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


def _build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, args):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False
    image_embedding_size = img_size // (encoder_patch_size if encoder_patch_size > 0 else 1)
    prompt_embed_dim = 256
    checkpoint = args.checkpoint
    
    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU
    
    model = EfficientHiSam(
        image_encoder=eImageEncoderViT(
            img_size=img_size,
            patch_size=encoder_patch_size,
            in_chans=3,
            patch_embed_dim=encoder_patch_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            neck_dims=encoder_neck_dims
        ),
        modal_aligner=ModalAligner(
            prompt_embed_dim,
            attn_layers=args.attn_layers,
            prompt_len=args.prompt_len
        ),
        prompt_encoder=ePromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=eMaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=eTwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=prompt_embed_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    
    if args.hier_det:
        model.hier_det = True
        model.hi_decoder = eHiDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=eTwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=prompt_embed_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        )
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        if args.hier_det:
            contain_hi_decoder = False
            for key in state_dict.keys():
                if 'hi_decoder' in key:
                    contain_hi_decoder = True
                    break
            if not contain_hi_decoder:
                with open(os.path.join('pretrained_checkpoint', args.model_type+'_maskdecoder.pth'), "rb") as f_maskdecoder:
                    mask_decoder_dict = torch.load(f_maskdecoder)
                for key, value in mask_decoder_dict.items():
                    new_key = key.replace('mask_decoder', 'hi_decoder')
                    state_dict[new_key] = value
        info = model.load_state_dict(state_dict, strict=False)
        print(info)
        
    return model
