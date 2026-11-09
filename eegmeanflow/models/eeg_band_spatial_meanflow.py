"""EEGMeanFlow: conditional average flow with a differential frequency/spatial branch."""
import math
from collections.abc import Mapping

import torch
from torch import nn

from .eegflow import EEGFlow


class DifferentialChannelBandBranch(nn.Module):
    """Five frequency bands, one shared mathematical channel-attention layer.

Real and imaginary FFT coefficients retain phase information. An unnormalized
token residual keeps absolute signal amplitude available around LayerNorm.
Only the added spectral correction excludes DC and frequencies outside the
five bands; these components remain available in the original backbone.
"""

    def __init__(self, channels, samples, hidden, patch, *, fs=200.0, width=64):
        super().__init__()
        if width < 1 or samples % patch:
            raise ValueError('Positive branch width and exact patch divisibility required')
        self.channels = channels
        self.samples = samples
        self.width = width
        self.fs = fs
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / fs)
        edges = (0.3, 4.0, 8.0, 13.0, 30.0, 45.0)
        masks = torch.stack([
            (frequencies >= lower) &
            ((frequencies < upper) if index < 4 else (frequencies <= upper))
            for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
        ])
        if not masks.any(dim=1).all() or masks.sum(dim=0).max() > 1:
            raise ValueError('Every frequency band must be represented without overlap')
        self.register_buffer('masks', masks)
        modes = len(frequencies)
        self.encode = nn.Linear(2 * modes, width)
        self.channel_embed = nn.Parameter(torch.randn(channels, width) * 0.02)
        self.band_embed = nn.Parameter(torch.randn(5, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        # A channel-shared decoder bias would cancel under the differential
        # projection below, so do not create an unidentifiable parameter.
        self.decode = nn.Linear(width, 2 * modes, bias=False)
        self.condition = nn.Linear(hidden, 5 * 2 * width)
        self.to_tokens = nn.Conv1d(channels, hidden, patch, stride=patch)
        nn.init.zeros_(self.to_tokens.weight)
        nn.init.zeros_(self.to_tokens.bias)

    def forward(self, x, condition):
        differential = x - x.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(differential, norm='ortho')
        masked = spectrum[:, None] * self.masks[None, :, None]
        features = torch.cat((masked.real, masked.imag), dim=-1)
        tokens = self.encode(features)
        tokens = tokens + self.channel_embed[None, None] + self.band_embed[None, :, None]
        shift, scale = self.condition(condition).reshape(
            len(x), 5, 2, self.width).unbind(2)
        attended_input = self.norm(tokens) * (1 + scale[:, :, None]) + shift[:, :, None]
        q, k, v = self.qkv(attended_input).chunk(3, dim=-1)
        # Explicit attention supports forward-mode AD. Parameters are shared
        # across bands, with separate C x C attention matrices for each band.
        attention = (q @ k.transpose(-1, -2) / math.sqrt(self.width)).softmax(-1)
        tokens = tokens + attention @ v
        real, imag = self.decode(tokens).chunk(2, dim=-1)
        correction = (torch.complex(real, imag) * self.masks[None, :, None]).sum(dim=1)
        wave = torch.fft.irfft(correction, n=self.samples, norm='ortho')
        # Preserve the branch's spatial-difference interpretation before the
        # token projection. The complete model is free to update common mode.
        wave = wave - wave.mean(dim=1, keepdim=True)
        return self.to_tokens(wave).transpose(1, 2)


class EEGBandSpatialMeanFlow(EEGFlow):
    """Original four DiT blocks plus one zero-initialized differential branch."""

    def __init__(self, *, branch_width=64, fs=200.0, **kwargs):
        super().__init__(meanflow=True, **kwargs)
        self.time_scale = 1.0
        if (len(self.blocks) != 4 or self.pos_embed.shape[-1] != 512 or
                self.num_heads != 16 or self.patch_size != 5 or
                self.blocks[0].mlp.fc1.out_features != 2048):
            raise ValueError('EEGMeanFlow requires the 4/512/16/2048/patch5 backbone')
        self.channel_band = DifferentialChannelBandBranch(
            self.in_channels, self.time_points, self.pos_embed.shape[-1],
            self.patch_size, fs=fs, width=branch_width)

    def forward(self, x, t, y, r=None):
        if r is None:
            raise ValueError('MeanFlow requires both interval endpoints r and t')
        tokens = self.x_embedder(x) + self.pos_embed
        condition = self.t_embedder(t * self.time_scale) + self.y_embedder(y, train=False)
        condition = condition + self.interval_embedder((t - r) * self.time_scale)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, condition)
            if index == 1:
                tokens = tokens + self.channel_band(x, condition)
        return self.final_layer(tokens, condition).reshape(
            len(x), self.time_points, self.out_channels).transpose(1, 2)
