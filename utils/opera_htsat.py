"""
Vendored OPERA-CT audio encoder (HTS-AT) with a pure-torch front-end.  No torchlibrosa / librosa / timm.

Source: https://github.com/evelyn0414/OPERA  (MIT, Copyright (c) 2024 OPERA)
        commit 3622310e667afb8aa40169050b4dd45de75946a2 (default branch `main` when vendored, 2026-10-18)
        files:  src/model/htsat/htsat.py, src/model/htsat/htsat_util.py, src/model/htsat/config.py (values used),
                src/model/models_cola.py (EncoderHTSAT / Cola.extract_feature), src/util.py (pre_process_audio_mel_t)
        which in turn are based on HTS-AT (Ke Chen et al., https://github.com/RetroCirce/HTS-Audio-Transformer, MIT)
        and Swin Transformer (Microsoft, https://github.com/microsoft/Swin-Transformer, MIT).
Weights (NOT in this file): OPERA-CT `encoder-operaCT.ckpt` from https://huggingface.co/evelyn0414/OPERA,
        revision d8de4322870b596f0a6ff6ea907b9a6996cd243a, sha256 83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae,
        licence CC-BY-NC-4.0 (research use; the NC term carries over to anything distilled from it).

MIT License

Copyright (c) 2024 OPERA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Changes against upstream (everything else is verbatim):
  * torchlibrosa Spectrogram / LogmelFilterBank / SpecAugmentation and the unused mixup / crop / repeat code removed;
    OPERA feeds librosa log-mels (not torchlibrosa), so the ckpt's `spectrogram_extractor.*` / `logmel_extractor.*`
    buffers are unused and are dropped on load.
  * BasicLayer does not collect the (unused) attention maps; the output (latent_output) is unchanged.
  * The forward only computes `latent_output` (768-d, average-pooled tokens before the AudioSet head);
    `tscam_conv` and `head` are kept so that the ckpt loads with strict=True once the extractor keys are dropped.
  * Front-end `opera_frontend` re-implements `pre_process_audio_mel_t` (librosa) in torch.
Only T <= 1024 mel frames are supported (a 5 s crop at 16 kHz / hop 512 is 157 frames; HTS-AT bicubic-resizes to 1024).
"""

import collections.abc
import math
import warnings
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchaudio
from torch.nn.init import _calculate_fan_in_and_fan_out

OPERA_REVISION = "3622310e667afb8aa40169050b4dd45de75946a2"  # code commit
OPERA_CKPT_REVISION = "d8de4322870b596f0a6ff6ea907b9a6996cd243a"  # HF weights revision
OPERA_CKPT_SHA256 = "83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae"

# values of src/model/htsat/config.py that the model uses
MEL_BINS = 64
HTSAT_SPEC_SIZE = 256
EMBED_DIM = 768


# ---------------------------------------------------------------- htsat_util.py (subset)
# from PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, patch_stride = 16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled



# ---------------------------------------------------------------- htsat.py: building blocks (verbatim except BasicLayer)
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        # print("H: ", H)
        # print("W: ", W)
        # pdb.set_trace()
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # VENDOR CHANGE: the attention maps (only used for the unused heat-map output) are not collected
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x, _ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, None

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# ---------------------------------------------------------------- htsat.py: HTSAT_Swin_Transformer (trimmed)
class HTSAT_Swin_Transformer(nn.Module):
    """HTS-AT (Swin-based) as configured by OPERA (defaults of upstream HTSATWrapper, enable_tscam=True,
    mel_bins=64, spec_size=256, patch 4/stride (4,4), embed_dim 96, depths [2,2,6,2], heads [4,8,16,32], window 8,
    drop_path_rate 0.1).  forward(x[B,1,T,64]) -> latent_output [B,768]."""

    def __init__(self, spec_size=HTSAT_SPEC_SIZE, patch_size=4, patch_stride=(4, 4),
                 in_chans=1, num_classes=527,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 32),
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln', mel_bins=MEL_BINS):
        super().__init__()
        depths = list(depths)
        num_heads = list(num_heads)
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm_layer = norm_layer if patch_norm else None
        self.freq_ratio = spec_size // mel_bins
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.patch_embed = PatchEmbed(
            img_size=spec_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=self.norm_layer, patch_stride=patch_stride)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint, norm_before_mlp=norm_before_mlp)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        SF = spec_size // (2 ** (len(depths) - 1)) // patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(self.num_features, num_classes, kernel_size=(SF, 3), padding=(0, 1))
        self.head = nn.Linear(num_classes, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        frames_num = x.shape[2]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, Fq, T = x.shape
        c_freq_bin = Fq // self.freq_ratio
        x = x.reshape(B, C, Fq // c_freq_bin, c_freq_bin, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        latent_output = self.avgpool(torch.flatten(x, 2))
        return torch.flatten(latent_output, 1)

    def reshape_wav2img(self, x):
        B, C, T, Fq = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and Fq <= target_F, "the wav size should less than or equal to the swin input size"
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if Fq < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    def forward(self, x):
        """x: [B, 1, T, 64] min-max normalised log-mel -> latent_output [B, 768]"""
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.reshape_wav2img(x)
        return self.forward_features(x)


# ---------------------------------------------------------------- OPERA encoder, loader, front-end (new code)
class OperaCTEncoder(nn.Module):
    """OPERA-CT audio encoder = upstream `Cola.encoder` (EncoderHTSAT -> HTSATWrapper -> htsat).
    forward(mel [B, T, 64]) -> [B, 768] (== Cola.extract_feature(x, 768))."""

    out_emb = EMBED_DIM

    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.htsat = HTSAT_Swin_Transformer(use_checkpoint=use_checkpoint)

    def forward(self, mel):
        return self.htsat(mel.unsqueeze(1))


_IGNORED_PREFIXES = ("htsat.spectrogram_extractor.", "htsat.logmel_extractor.", "g.", "layer_norm.", "linear.")


def audit_opera_state(state_dict, encoder):
    """Strip the Lightning prefix `encoder.encoder.` and audit the keys against `encoder`.
    Returns (filtered_state_dict, report); report = dict(loaded, missing, unexpected_ignored, unexpected_other,
    shape_mismatch, n_ckpt)."""
    prefix = "encoder.encoder."
    sd, ignored = {}, []
    for k, v in state_dict.items():
        k2 = k[len(prefix):] if k.startswith(prefix) else k  # g.* / layer_norm.* / linear.* are Cola's projection
        if k2.startswith(_IGNORED_PREFIXES):
            ignored.append(k)
        else:
            sd[k2] = v
    own = encoder.state_dict()
    missing = sorted(k for k in own if k not in sd)
    other = sorted(k for k in sd if k not in own)
    mismatch = sorted(k for k in sd if k in own and own[k].shape != sd[k].shape)
    report = dict(loaded=len(sd) - len(other), missing=missing, unexpected_ignored=len(ignored),
                  unexpected_other=other, shape_mismatch=mismatch, n_ckpt=len(state_dict))
    return sd, report


def load_opera_ct(path, map_location="cpu", use_checkpoint=False, verbose=True):
    """Build OperaCTEncoder and load `encoder-operaCT.ckpt` (Lightning ckpt; torch.load(weights_only=True)).
    The extractor / projection keys (see _IGNORED_PREFIXES) are dropped; everything else must match exactly
    (strict). Returns (encoder.eval(), report)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    enc = OperaCTEncoder(use_checkpoint=use_checkpoint)
    sd, rep = audit_opera_state(ckpt["state_dict"], enc)
    assert not rep["missing"], f"missing keys: {rep['missing'][:5]}"
    assert not rep["unexpected_other"], f"unexpected keys: {rep['unexpected_other'][:5]}"
    assert not rep["shape_mismatch"], f"shape mismatch: {rep['shape_mismatch'][:5]}"
    enc.load_state_dict(sd, strict=True)
    if verbose:
        print(f"opera_ct: loaded {rep['loaded']} tensors, ignored {rep['unexpected_ignored']} "
              f"(extractor/projection), missing 0, unexpected 0, ckpt keys {rep['n_ckpt']}", flush=True)
    return enc.eval(), rep


_FB_CACHE = {}


def _mel_filterbank(f_max, n_fft=1024, sr=16000, n_mels=MEL_BINS, f_min=50.0):
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) (Slaney scale + Slaney norm) -> [n_mels, n_fft//2+1]"""
    key = float(f_max)
    if key not in _FB_CACHE:
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=f_min, f_max=float(f_max), n_mels=n_mels,
            sample_rate=sr, norm="slaney", mel_scale="slaney")  # [n_freqs, n_mels]
        _FB_CACHE[key] = fb.t().contiguous().double()
    return _FB_CACHE[key]


def opera_frontend(wave16k, f_max=8000, n_fft=1024, hop=512):
    """Torch version of OPERA's `pre_process_audio_mel_t` (src/util.py):
    librosa melspectrogram(n_mels=64, fmin=50, fmax=f_max, n_fft=1024, hop=512, center=True, pad 'constant', power 2)
    -> power_to_db(ref=np.max, amin=1e-10, top_db=80) -> per-clip min-max to [0, 1] (unchanged if max == min)
    -> transpose.  wave16k [B, N] float -> [B, T, 64] float32 (T = 1 + N // 512).  Computed in float64 internally."""
    x = wave16k.to(torch.float64)
    win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64, device=x.device)
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True,
                      pad_mode="constant", return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2  # [B, 513, T]
    fb = _mel_filterbank(f_max, n_fft=n_fft).to(x.device)
    mel = torch.einsum("mf,bft->bmt", fb, power)  # [B, 64, T]
    amin = 1e-10
    log_spec = 10.0 * torch.log10(torch.clamp(mel, min=amin))
    ref = torch.clamp(mel.amax(dim=(-2, -1), keepdim=True), min=amin)
    log_spec = log_spec - 10.0 * torch.log10(ref)
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 80.0)
    lo = log_spec.amin(dim=(-2, -1), keepdim=True)
    hi = log_spec.amax(dim=(-2, -1), keepdim=True)
    norm = (log_spec - lo) / (hi - lo).clamp(min=1e-30)
    out = torch.where(hi > lo, norm, log_spec)
    return out.transpose(1, 2).to(torch.float32).contiguous()


def opera_embed(encoder, wave4k, f_max=8000, resample=True):
    """The student's 4 kHz waveform crops [B, T] (5 s = 20000 samples) -> OPERA-CT embedding [B, 768].
    Resample 4 -> 16 kHz (torchaudio.functional.resample defaults), OPERA front-end, encoder.
    The front-end runs in fp32/fp64 outside any autocast and without gradient; gradients flow only through
    the encoder."""
    if wave4k.dim() == 3:
        wave4k = wave4k.squeeze(1)
    with torch.no_grad():
        w = wave4k.float()
        if resample:
            w = torchaudio.functional.resample(w, 4000, 16000)
        with torch.autocast(device_type=w.device.type, enabled=False):
            mel = opera_frontend(w, f_max=f_max)
    return encoder(mel)
