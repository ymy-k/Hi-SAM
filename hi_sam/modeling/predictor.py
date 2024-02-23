# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from .hi_sam import HiSam

from typing import Optional, Tuple
from hi_sam.data.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model: HiSam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

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

    def predict(
        self,
        multimask_output: bool = False,
        return_logits: bool = False,
        hier_det: bool = False,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if hier_det:
            assert point_coords is not None and point_labels is not None, \
                "point_coords and point_labels must be supplied"
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
            masks, hr_masks, iou_predictions, iou_predictions_hr, hi_masks, hi_iou, word_masks = self.predict_torch(
                multimask_output,
                return_logits=return_logits,
                hier_det=True,
                point_coords=coords_torch,
                point_labels=labels_torch
            )
            masks_np = masks[0].detach().cpu().numpy()
            hr_masks_np = hr_masks[0].detach().cpu().numpy()
            iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
            iou_predictions_hr_np = iou_predictions_hr[0].detach().cpu().numpy()
            hi_masks_np = hi_masks.detach().cpu().numpy()
            hi_iou_np = hi_iou.detach().cpu().numpy()
            word_masks_np = word_masks.detach().cpu().numpy()
            return (masks_np, hr_masks_np, iou_predictions_np, iou_predictions_hr_np,
                    hi_masks_np, hi_iou_np, word_masks_np)
        else:
            masks, hr_masks, iou_predictions, iou_predictions_hr = self.predict_torch(
                multimask_output,
                return_logits=return_logits,
            )
            masks_np = masks[0].detach().cpu().numpy()
            hr_masks_np = hr_masks[0].detach().cpu().numpy()
            iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
            iou_predictions_hr_np = iou_predictions_hr[0].detach().cpu().numpy()
            return masks_np, hr_masks_np, iou_predictions_np, iou_predictions_hr_np

    @torch.no_grad()
    def predict_torch(
            self,
            multimask_output: bool = False,
            return_logits: bool = False,
            hier_det: bool = False,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        sparse_emb = self.model.modal_aligner(self.features)

        low_res_masks, high_res_masks, iou_pred, iou_pred_hr = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        hr_masks = self.model.postprocess_masks(high_res_masks, self.input_size, self.original_size)

        if not hier_det:
            if not return_logits:
                masks = masks > self.model.mask_threshold
                hr_masks = hr_masks > self.model.mask_threshold
            return masks, hr_masks, iou_pred, iou_pred_hr
            
        else:
            points = (point_coords, point_labels)
            point_embeddings, _ = self.model.prompt_encoder(
                points=points, boxes=None, masks=None
            )
            hi_masks, iou_pred_hi, word_masks = self.model.hi_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=point_embeddings,
                multimask_output=True
            )
            hi_masks = self.model.postprocess_masks(hi_masks[:, 1:, :, :], self.input_size, self.original_size)
            word_masks = self.model.postprocess_masks(word_masks, self.input_size, self.original_size)
            if not return_logits:
                hi_masks = hi_masks > self.model.mask_threshold
                word_masks = word_masks > self.model.mask_threshold
            return masks, hr_masks, iou_pred, iou_pred_hr, hi_masks, iou_pred_hi, word_masks

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
