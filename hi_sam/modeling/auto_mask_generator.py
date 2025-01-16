# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import sys

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from hi_sam.data.transforms import ResizeLongestSide


class AutoMaskGenerator:
    def __init__(
        self,
        sam_model,
        efficient_hisam: bool=False,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.efficient_hisam = efficient_hisam
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.reset_fgmask()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        # import pdb;pdb.set_trace()
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        # import pdb;pdb.set_trace()
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def set_fgmask(
            self,
            fgmask: np.ndarray,
    ) -> None:
        assert (len(fgmask.shape) == 2 or len(fgmask.shape) == 3)
        if len(fgmask.shape) > 2:
            fgmask = fgmask[:, :, 0]
        self.reset_fgmask()
        fgmask = self.transform.apply_image(fgmask)
        fgmask = fgmask > (fgmask.max() / 2)
        self.fgmask = torch.as_tensor(fgmask, device=self.device)
        self.is_fgmask_set = True

    @torch.no_grad()
    def forward_foreground_points(
            self,
            from_low_res: bool = False,
            fg_points_num: int = 600
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        if self.is_fgmask_set:
            fg_mask = self.fgmask
        else:
            sparse_emb = self.model.modal_aligner(self.features)
            low_res_mask, high_res_mask, iou_pred, iou_pred_hr = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                multimask_output=False,
            )
            assert low_res_mask.shape[0] == 1, "only support one image per batch now"

            if from_low_res:
                fg_mask = (low_res_mask > self.model.mask_threshold).squeeze(1)[0]  # (256, 256)
            else:
                fg_mask = (high_res_mask > self.model.mask_threshold).squeeze(1)[0]  # (1024, 1024)
            del low_res_mask
            # del high_res_mask

        # sample fg_points randomly
        y_idx, x_idx = torch.where(fg_mask > 0)

        p_n = x_idx.size(0)
        perm = torch.randperm(p_n)
        idx = perm[:min(p_n, fg_points_num)]

        y_idx, x_idx = y_idx[idx][:, None], x_idx[idx][:, None]
        fg_points = torch.cat((x_idx, y_idx), dim=1)[:, None, :]  # (k, 1, 2)
        if from_low_res and (not self.is_fgmask_set):
            fg_points = fg_points * 4  # from 256 to 1024
        return fg_points, high_res_mask

    @torch.no_grad()
    def forward_hi_decoder(
            self,
            point_coords,
            point_labels
    ):
        if self.efficient_hisam:
            point_embeddings = self.model.prompt_encoder(point_coords, point_labels)
        else:
            point_embeddings, _ = self.model.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        hi_masks_logits, hi_iou_preds, word_masks_logits = self.model.hi_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=point_embeddings,
            multimask_output=True
        )
        return hi_masks_logits, hi_iou_preds, word_masks_logits

    def predict(
        self,
        from_low_res: bool = False,
        fg_points_num: int = 1500,
        batch_points_num: int = 100,
        score_thresh: dict = {"word": 0.3, "line": 0.5, "para": 0.5, "refined_word": 0.3},
        nms_thresh: dict = {"word": 0.3, "line": 0.5, "para": 0.5, "refined_word": 0.3},
        return_logits: bool = False,
        oracle_point_prompts: np.ndarray = None,
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        assert batch_points_num <= fg_points_num

        if oracle_point_prompts is not None:
            # resize point prompts
            oracle_point_prompts = self.transform.apply_coords(oracle_point_prompts, self.original_size)
            fg_points = torch.tensor(oracle_point_prompts, dtype=torch.int64, device=self.features.device)
            fg_points = fg_points[:, None, :]
            hr_mask = None
        else:
            fg_points, hr_mask = self.forward_foreground_points(from_low_res, fg_points_num)
            hr_mask = self.model.postprocess_masks(hr_mask, self.input_size, self.original_size)
        fg_points_num = fg_points.shape[0]
        if fg_points_num == 0:
            return None, None, None

        masks, scores, word_masks = [], [], []
        for start_idx in range(0, fg_points_num, batch_points_num):
            end_idx = min(start_idx + batch_points_num, fg_points_num)
            hi_masks_logits, hi_iou_preds, word_masks_logits = self.forward_hi_decoder(
                fg_points[start_idx:end_idx, :, :],
                torch.ones((end_idx-start_idx, 1), device=fg_points.device)
            )
            # hi_masks_logits = hi_masks_logits[:, 1:, :, :]
            masks.append(hi_masks_logits)
            scores.append(hi_iou_preds)
            word_masks.append(word_masks_logits)
        del hi_masks_logits
        del word_masks_logits
        masks = torch.cat(masks, dim=0)  # (fg_points_num, x, 256, 256)
        scores = torch.cat(scores, dim=0)  # (fg_points_num, x)
        word_masks = torch.cat(word_masks, dim=0)
        
        masks = {"word": masks[:, 0], "line": masks[:, 1], "para": masks[:, 2], "refined_word": word_masks[:, 0]}
        scores = {"word": scores[:, 0], "line": scores[:, 1], "para": scores[:, 2], "refined_word": scores[:, 0]}
        for key in masks.keys():
            # filter low quality lines
            keep = scores[key] > score_thresh[key]
            masks[key] = masks[key][keep]
            scores[key] = scores[key][keep]
            
            # MNS
            updated_scores = matrix_nms(
                seg_masks=(masks[key] > self.model.mask_threshold),
                scores=scores[key]
            )
            keep = updated_scores > nms_thresh[key]
            masks[key] = masks[key][keep]
            scores[key] = scores[key][keep]
            
            # postprocess
            masks[key] = self.model.postprocess_masks(masks[key].unsqueeze(1), self.input_size, self.original_size)[:, 0]
            masks[key] = (masks[key] > self.model.mask_threshold).cpu().numpy()
            scores[key] = scores[key].cpu().numpy()
        stroke = hr_mask.cpu().numpy()
        return masks, scores, stroke


    def predict_text_detection(
        self,
        from_low_res: bool = False,
        fg_points_num: int = 600,
        batch_points_num: int = 100,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        return_logits: bool = False,
        oracle_point_prompts: np.ndarray = None,
        zero_shot: bool = True,
        dataset: str = 'totaltext'
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        assert batch_points_num <= fg_points_num

        if oracle_point_prompts is not None:
            # resize point prompts
            oracle_point_prompts = self.transform.apply_coords(oracle_point_prompts, self.original_size)
            fg_points = torch.tensor(oracle_point_prompts, dtype=torch.int64, device=self.features.device)
            fg_points = fg_points[:, None, :]
        else:
            fg_points = self.forward_foreground_points(from_low_res, fg_points_num)
        fg_points_num = fg_points.shape[0]
        if fg_points_num == 0:
            return None, None

        masks, scores, word_masks = [], [], []
        for start_idx in range(0, fg_points_num, batch_points_num):
            end_idx = min(start_idx + batch_points_num, fg_points_num)
            hi_masks_logits, hi_iou_preds, word_masks_logits = self.forward_hi_decoder(
                fg_points[start_idx:end_idx, :, :],
                torch.ones((end_idx-start_idx, 1), device=fg_points.device)
            )
            hi_masks_logits = hi_masks_logits[:, 1:, :, :]
            masks.append(hi_masks_logits)
            scores.append(hi_iou_preds)
            word_masks.append(word_masks_logits)
        del hi_masks_logits
        del word_masks_logits
        masks = torch.cat(masks, dim=0)  # (fg_points_num, x, 256, 256)
        scores = torch.cat(scores, dim=0)  # (fg_points_num, x)
        word_masks = torch.cat(word_masks, dim=0)

        # filter low quality lines
        if dataset != 'ctw1500' and not zero_shot:
            scores[:, 1] = 1.0  # since no iou score prediction
        keep = scores[:, 1] > score_thresh
        if keep.sum() == 0:
            return None, None
        masks = masks[keep]
        scores = scores[keep]
        word_masks = word_masks[keep]

        if dataset == 'totaltext' and zero_shot:
            word_masks = word_masks > self.model.mask_threshold
            word_masks = (word_masks.sum(dim=0)[None, ...] > 0).type(torch.float32)
            word_masks = self.model.postprocess_masks(word_masks, self.input_size, self.original_size) > 0
            masks_np = word_masks.cpu().numpy()
            return masks_np, None
        else:
            if dataset != 'ctw1500' and not zero_shot:
                updated_scores = matrix_nms(
                    seg_masks=(word_masks[:, 0, :, :] > self.model.mask_threshold),
                    scores=scores[:, 1]
                )
            else:
                updated_scores = matrix_nms(
                    seg_masks=(masks[:, -2, :, :] > self.model.mask_threshold),
                    scores=scores[:, 1]
                )
            keep = updated_scores > nms_thresh
            if keep.sum() == 0:
                return None, None
            if dataset == 'ctw1500':
                masks = masks[keep][:, 0:1, :, :]
            else:
                masks = word_masks[keep][:, 0:1, :, :]
            scores = scores[keep][:, 1]
            masks = self.model.postprocess_masks(masks, self.input_size, self.original_size)
            masks = masks > self.model.mask_threshold
            masks_np = masks.cpu().numpy()
            scores_np = scores.cpu().numpy()
            return masks_np, scores_np

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def reset_fgmask(self) -> None:
        self.is_fgmask_set = False
        self.fgmask = None


def matrix_nms(seg_masks, scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS from SOLOv2

    Args:
        seg_masks (Tensor): shape (n, h, w)
        scores (Tensor): shape (n)
        kernel (str): 'linear' or 'gaussian'
        sigma (float): std in gaussian method
        sum_masks (Tensor): the sum of seg_masks
    """
    n_samples = len(seg_masks)
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    del seg_masks
    # union
    sum_masks = sum_masks.expand(n_samples, n_samples)
    # iou
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # IOU compensation
    compensate_iou, _ = iou_matrix.max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    # IOU decay
    decay_iou = iou_matrix  # no label matrix because there is only one foreground class

    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coef, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coef, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError
    updated_score = scores * decay_coef
    return updated_score


def get_para_iou(para_masks):
    """
        Args:
            para_masks (Tensor): shape (n, h, w)
        """
    n_samples = len(para_masks)
    sum_masks = para_masks.sum((1, 2)).float()
    para_masks = para_masks.reshape(n_samples, -1).float()
    inter_matrix = torch.mm(para_masks, para_masks.transpose(1, 0))
    # del para_masks
    sum_masks = sum_masks.expand(n_samples, n_samples)
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix))

    return iou_matrix
