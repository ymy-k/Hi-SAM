# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBlock


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.invalid_points = nn.Embedding(1, embed_dim)
        self.point_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_top_left_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_bottom_right_embeddings = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        invalid_label_ids = torch.eq(labels, -1)[:,:,None]
        point_label_ids = torch.eq(labels, 1)[:,:,None]
        topleft_label_ids = torch.eq(labels, 2)[:,:,None]
        bottomright_label_ids = torch.eq(labels, 3)[:,:,None]
        point_embedding = point_embedding + self.invalid_points.weight[:,None,:] * invalid_label_ids
        point_embedding = point_embedding + self.point_embeddings.weight[:,None,:] * point_label_ids
        point_embedding = point_embedding + self.bbox_top_left_embeddings.weight[:,None,:] * topleft_label_ids
        point_embedding = point_embedding + self.bbox_bottom_right_embeddings.weight[:,None,:] * bottomright_label_ids
        return point_embedding

    def forward(
        self,
        coords,
        labels,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points: A tensor of shape [B, 2]
          labels: An integer tensor of shape [B] where each element is 1,2 or 3.

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
        """
        return self._embed_points(coords, labels)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones([h, w], device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        upscaling_layer_dims: List[int],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1
        else:
            self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim

        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )
        
        # add for high_res mask
        self.output_upscaling_hr = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            activation()
        )
        self.output_hypernetworks_mlps_hr = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 16,
            num_layers=2,
            act=activation
        )
        self.iou_prediction_head_hr = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=1,
            num_layers=iou_head_depth,
            act=activation,
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, hr_masks, iou_pred, iou_pred_hr = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        if multimask_output and self.num_multimask_outputs > 1:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        
        return masks[:, mask_slice, :, :], hr_masks[:, mask_slice, :, :], iou_pred[:, mask_slice], iou_pred_hr

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )  # [1, L, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)   # [1, L+N, 256]
        # Expand per-image data in batch direction to be per-mask
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape
        hs, src = self.transformer(image_embeddings, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)
        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)
            
        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        upscaled_embedding = self.output_upscaling_hr(upscaled_embedding)
        hyper_in_hr = self.output_hypernetworks_mlps_hr(mask_tokens_out[:, 0, :])
        b, c, h, w = upscaled_embedding.shape
        hr_masks = (hyper_in_hr @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # (1,1,1024,1024)
        
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # (1, 4)
        iou_pred_hr = self.iou_prediction_head_hr(iou_token_out)  # (1, 1)
        
        return masks, hr_masks, iou_pred, iou_pred_hr


class HiDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        upscaling_layer_dims: List[int],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1
        else:
            self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim

        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )
        
        self.word_mask_dc = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 16, kernel_size=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
        )
        self.word_mask_refine = nn.Sequential(
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            activation(),
        )
        self.output_word_mlp = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 16,
            num_layers=2,
            act=activation
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, iou_pred, word_masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        if multimask_output and self.num_multimask_outputs > 1:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        
        return masks[:, mask_slice, :, :], iou_pred[:, mask_slice], word_masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )  # [1, L, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)
        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)
            
        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # (1, 4)
        
        upscaled_embedding = self.word_mask_dc(upscaled_embedding)
        upscaled_embedding = F.interpolate(upscaled_embedding, (384, 384), mode="bilinear", align_corners=False)
        upscaled_embedding = self.word_mask_refine(upscaled_embedding)
        hyper_in_word = self.output_word_mlp(mask_tokens_out[:, 1:2, :])
        b, c, h, w = upscaled_embedding.shape
        word_masks = (hyper_in_word @ upscaled_embedding.view(b, c, h*w)).view(b, -1, h, w)
        
        return masks, iou_pred, word_masks
