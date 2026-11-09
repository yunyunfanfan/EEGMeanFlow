"""EEGDiffuser backbone for conditional velocity/interval-velocity prediction.

MeanFlow identity: Geng et al., https://arxiv.org/abs/2505.13447.
The four upstream DiT blocks are reused unchanged. This is an EEG adaptation,
not a reproduction of the image architecture or its reported image results.
"""
import torch
from torch import nn
from .eegdiffuser import EEGDiffuser, TimestepEmbedder


class EEGFlow(EEGDiffuser):
    def __init__(self, *, meanflow=False, **kwargs):
        kwargs['learn_sigma'] = False
        super().__init__(**kwargs)
        self.meanflow = meanflow
        self.time_scale = 1000.0
        if meanflow:
            self.interval_embedder = TimestepEmbedder(self.pos_embed.shape[-1])
            nn.init.normal_(self.interval_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.interval_embedder.mlp[0].bias)
            nn.init.zeros_(self.interval_embedder.mlp[2].weight)
            nn.init.zeros_(self.interval_embedder.mlp[2].bias)

    def forward(self, x, t, y, r=None):
        if self.meanflow and r is None:
            raise ValueError('MeanFlow requires both interval endpoints r and t')
        if not self.meanflow and r is not None:
            raise ValueError('Ordinary Flow Matching has no interval input')
        batch = x.shape[0]
        tokens = self.x_embedder(x) + self.pos_embed
        condition = self.t_embedder(t * self.time_scale)
        # Drop labels ONCE outside this function. Prediction and JVP must see
        # exactly the same label; random token_drop inside JVP is invalid.
        condition = condition + self.y_embedder(y, train=False)
        if self.meanflow:
            condition = condition + self.interval_embedder((t - r) * self.time_scale)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.final_layer(tokens, condition).reshape(
            batch, self.time_points, self.out_channels).transpose(1, 2)
