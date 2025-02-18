import torch
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.utils import to_2tuple
from mmcv.cnn import build_norm_layer
import torch.utils.checkpoint as cp
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import functional as F


class WindowMCA(BaseModule):
    """Window based multi-head cross-attention (W-MCA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 cfgs):

        super().__init__(init_cfg=None)
        self.embed_dims = embed_dims
        self.window_size = cfgs.window_size  # Wh, Ww
        self.pre_window_size = cfgs.pre_window_size  # Wh_pre, Ww_pre
        self.num_heads = cfgs.num_heads
        head_embed_dims = self.embed_dims // cfgs.num_heads
        self.scale = cfgs.qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table_1 = nn.Parameter(
            torch.zeros((self.window_size[0] * self.window_size[1], self.pre_window_size[0] * self.pre_window_size[1], self.num_heads))
            )  # (Wh * Ww, pre_Wh * pre_Ww, nH)
        
        self.relative_position_bias_table_2 = nn.Parameter(
            torch.zeros((self.window_size[0] * self.window_size[1], self.pre_window_size[0] * self.pre_window_size[1], self.num_heads))
            )  # (Wh * Ww, pre_Wh * pre_Ww, nH)

        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3, bias=cfgs.qkv_bias)
        self.attn_drop = nn.Dropout(cfgs.attn_drop_rate)
        self.proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.proj_drop = nn.Dropout(cfgs.drop_rate)

        self.softmax_1 = nn.Softmax(dim=-1)
        self.softmax_2 = nn.Softmax(dim=-2)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table_1, std=0.02)
        trunc_normal_(self.relative_position_bias_table_2, std=0.02)

    def forward(self, x1, x2):
        """
        Args:
            x1 (num_windows*B, N, C)
            x2 (num_windows*B, M, C)
        """
        B, N, C = x1.shape
        _, M, _ = x2.shape
        qkv_1 = self.qkv(x1).reshape(B, N, 3, self.num_heads,
                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_2 = self.qkv(x2).reshape(B, M, 3, self.num_heads,
                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2] # B, nH, N, C // self.num_heads
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2] # B, nH, M, C // self.num_heads

        q_1 = q_1 * self.scale
        q_2 = q_2 * self.scale
        attn_1 = (q_1 @ k_2.transpose(-2, -1)) # (B, nH, N, M)
        attn_2 = (q_2 @ k_1.transpose(-2, -1)) # (B, nH, M, N)

        relative_position_bias_1 = self.relative_position_bias_table_1.permute(
            2, 0, 1)  # (nH, Wh*Ww, pre_Wh*pre_Ww)
        relative_position_bias_2 = self.relative_position_bias_table_2.permute(
            2, 0, 1) # (nH, pre_Wh*pre_Ww, Wh*Ww)
        attn_1 = attn_1 + relative_position_bias_1.unsqueeze(0)
        attn_2 = attn_2 + relative_position_bias_2.permute(0, 2, 1).unsqueeze(0)
        attn_to_multiply_1 = self.softmax_1(attn_1)
        attn_to_multiply_2 = self.softmax_1(attn_2)
        attn_to_output_1 = self.softmax_2(attn_1)
        attn_to_output_2 = self.softmax_2(attn_2)
        attn_to_output_1 = torch.mean(attn_to_output_1, dim=1) # (B, N, M)
        attn_to_output_2 = torch.mean(attn_to_output_2, dim=1) # (B, M, N)

        attn_1 = self.attn_drop(attn_to_multiply_1) # (B, nH, N, M)
        attn_2 = self.attn_drop(attn_to_multiply_2) # (B, nH, M, N)

        x1 = (attn_1 @ v_2).transpose(1, 2).reshape(B, N, C)
        x2 = (attn_2 @ v_1).transpose(1, 2).reshape(B, M, C)
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        x1 = self.proj_drop(x1)
        x2 = self.proj_drop(x2)
        return x1, attn_to_output_1, x2, attn_to_output_2

class CwindowBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 cfgs):

        super().__init__(init_cfg=None)
        self.norm1 = build_norm_layer(cfgs.norm_cfg, embed_dims)[1]
        self.attn = WindowMCA(
            embed_dims=embed_dims,
            cfgs=cfgs)
        self.norm2 = build_norm_layer(cfgs.norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * cfgs.mlp_ratio,
            num_fcs=2,
            ffn_drop=cfgs.drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=cfgs.drop_path_rate),
            act_cfg=cfgs.act_cfg,
            add_identity=True,
            init_cfg=None)
        self.window_size = cfgs.window_size
        self.pre_window_size = cfgs.pre_window_size

    def forward(self, x1, x2):
        """
        Args:
            x1 (B * num_re, num_k, C)
            x2 (B * num_re, num_k_pre, C)
        """
        B, N, C = x1.shape
        _, M, _ = x2.shape
        assert N == self.window_size[0] * self.window_size[1], 'input feature has wrong size'
        assert M == self.pre_window_size[0] * self.pre_window_size[1], 'input feature has wrong size'
        identity1 = x1
        identity2 = x2
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x1, attn1, x2, attn2 = self.attn(x1, x2)
        x1 = x1 + identity1
        x2 = x2 + identity2
        identity1 = x1
        identity2 = x2
        x1 = self.norm2(x1)
        x2 = self.norm2(x2)
        x1 = self.ffn(x1, identity=identity1)
        x2 = self.ffn(x2, identity=identity2)
        return x1, attn1, x2, attn2

class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 cfgs):

        super().__init__(init_cfg=None)
        self.embed_dims = embed_dims
        self.window_size = cfgs.window_size  # Wh, Ww
        self.num_heads = cfgs.num_heads
        head_embed_dims = embed_dims // cfgs.num_heads
        self.scale = cfgs.qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=cfgs.qkv_bias)
        self.attn_drop = nn.Dropout(cfgs.attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(cfgs.drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
    """

    def __init__(self,
                 embed_dims,
                 cfgs,
                 shift_size=0):
        super().__init__(init_cfg=None)

        self.window_size = cfgs.window_size_int
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            cfgs=cfgs,
            )

        self.drop = build_dropout(dict(type='DropPath', drop_prob=cfgs.drop_path_rate))

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        shift (bool, optional): whether to shift window or not. Default False.
    """

    def __init__(self,
                 embed_dims,
                 cfgs,
                 shift=False):

        super().__init__(init_cfg=None)

        self.with_cp = cfgs.with_cp

        self.norm1 = build_norm_layer(cfgs.norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            cfgs=cfgs,
            shift_size=cfgs.window_size_int // 2 if shift else 0,)

        self.norm2 = build_norm_layer(cfgs.norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * cfgs.mlp_ratio,
            num_fcs=2,
            ffn_drop=cfgs.drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=cfgs.drop_path_rate),
            act_cfg=cfgs.act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x