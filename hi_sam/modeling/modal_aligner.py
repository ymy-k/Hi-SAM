# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Tuple
from einops import rearrange
from .common import LayerNorm2d
import math


class CrossModalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] *(num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = act()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, output_dim, nhead, dropout: float= 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(output_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(output_dim)

        self.cross_attn = nn.MultiheadAttention(output_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(output_dim)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim)
        )
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, sparse_embeddings, image_embeddings):
        sparse_embeddings = sparse_embeddings + self.dropout1(self.self_attn(
            query=sparse_embeddings,
            key=sparse_embeddings,
            value=sparse_embeddings)[0])
        sparse_embeddings = self.norm1(sparse_embeddings)
        sparse_embeddings = sparse_embeddings + self.dropout2(self.cross_attn(
            query=sparse_embeddings,
            key=rearrange(image_embeddings, "b c h w -> b (h w) c"),
            value=rearrange(image_embeddings, "b c h w -> b (h w) c"))[0])
        sparse_embeddings = sparse_embeddings + self.dropout3(self.ffn(self.norm2(sparse_embeddings)))

        return sparse_embeddings


class ModalAligner(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            act: Type[nn.Module] = nn.GELU,
            nhead: int = 8,
            dropout: float = 0.1,
            attn_layers: int = 1,
            prompt_len: int = 12,
    ) -> None:
        super().__init__()

        self.prompt_len = prompt_len
        self.conv = nn.Sequential(
            nn.Conv2d(transformer_dim, self.prompt_len, kernel_size=3, padding=1, bias=False),
            act(),
            nn.Conv2d(self.prompt_len, self.prompt_len, kernel_size=3, padding=1, bias=False),
            act(),
            nn.Conv2d(self.prompt_len, self.prompt_len, kernel_size=3, padding=1, bias=False),
            act(),
            nn.Conv2d(self.prompt_len, self.prompt_len, kernel_size=3, padding=1, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(transformer_dim, nhead, dropout) for _ in range(attn_layers)
        ])

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = image_embeddings.shape
        spatial_attention = self.conv(image_embeddings)  # bs, len, 64, 64
        spatial_attention = spatial_attention.reshape(bs, self.prompt_len, -1)
        spatial_attention = F.sigmoid(spatial_attention)[..., None]  # bs, len, h*w, 1
        feat = image_embeddings.reshape(bs, c, -1).permute(0, 2, 1)[:, None, ...]  # bs, 1, h*w, c
        sparse_embeddings = (feat * spatial_attention).mean(dim=2)  # bs, len, c

        for layer in self.transformer_layers:
            sparse_embeddings = layer(sparse_embeddings, image_embeddings)
            
        return sparse_embeddings
