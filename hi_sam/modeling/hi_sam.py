# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, HiDecoder
from .prompt_encoder import PromptEncoder
from .modal_aligner import ModalAligner


class HiSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        modal_aligner: ModalAligner,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        for n, p in self.image_encoder.named_parameters():
            if "Adapter" not in n:
                p.requires_grad = False
        print("Freeze image encoder.")

        self.prompt_encoder = prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        self.modal_aligner = modal_aligner
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.hier_det = False
        self.hi_decoder = None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        image_embeddings = self.image_encoder(input_images)
        sparse_emb = self.modal_aligner(image_embeddings)

        up_masks_logits = []
        up_masks = []
        iou_preds = []
        hr_masks_logits = []
        hr_masks = []
        iou_preds_hr = []
        if self.hier_det:
            hi_masks_logits = []
            hi_iou_preds = []
            word_masks_logits = []

        for image_record, curr_embedding, sparse_embeddings in zip(batched_input, image_embeddings, sparse_emb):
            low_res_masks, high_res_masks, iou_pred, iou_pred_hr = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings.unsqueeze(0),
                multimask_output=multimask_output
            )
            iou_preds.append(iou_pred)
            iou_preds_hr.append(iou_pred_hr)
            upscaled_masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            high_res_masks = self.postprocess_masks(
                high_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            up_masks_logits.append(upscaled_masks)
            up_masks.append(upscaled_masks > self.mask_threshold)
            hr_masks_logits.append(high_res_masks)
            hr_masks.append(high_res_masks > self.mask_threshold)

            if self.hier_det:
                points = (image_record["point_coords"], image_record["point_labels"])
                point_embeddings, _ = self.prompt_encoder(
                    points=points, boxes=None, masks=None
                )
                hi_masks, hi_iou_pred, word_masks = self.hi_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=point_embeddings,
                    multimask_output=True
                )
                hi_masks_logits.append(hi_masks)
                hi_iou_preds.append(hi_iou_pred)
                word_masks_logits.append(word_masks)

        up_masks_logits = torch.cat(up_masks_logits, dim=0)
        up_masks = torch.cat(up_masks, dim=0)
        iou_preds = torch.cat(iou_preds, dim=0)
        hr_masks_logits = torch.cat(hr_masks_logits, dim=0)
        hr_masks = torch.cat(hr_masks, dim=0)
        iou_preds_hr = torch.cat(iou_preds_hr, dim=0)

        if self.hier_det:
            hi_masks_logits = torch.cat(hi_masks_logits, dim=0)
            hi_iou_preds = torch.cat(hi_iou_preds, dim=0)
            word_masks_logits = torch.cat(word_masks_logits, dim=0)
            return (up_masks_logits, up_masks, iou_preds, hr_masks_logits, hr_masks, iou_preds_hr,
                    hi_masks_logits, hi_iou_preds, word_masks_logits)
        else:
            return up_masks_logits, up_masks, iou_preds, hr_masks_logits, hr_masks, iou_preds_hr

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x