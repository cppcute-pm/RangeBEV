import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from typing import Union, Dict, Optional, Tuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_0 = LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask_self: torch.Tensor = None, attn_mask_cross: torch.Tensor = None):
        """
        x: Lx, b, dx
        y: Ly, b, dy
        attn_mask_self: B, Lx, Lx
        attn_mask_cross: B, Lx, Ly
        """
        ##### self
        if attn_mask_self is not None:
            B, Lx, _ = attn_mask_self.shape
            attn_mask_self_input = attn_mask_self.unsqueeze(1).expand(-1, self.n_head, -1, -1).reshape(B * self.n_head, Lx, Lx)
        else:
            attn_mask_self_input = None
        if attn_mask_cross is not None:
            B, Lx, Ly = attn_mask_cross.shape
            attn_mask_cross_input = attn_mask_cross.unsqueeze(1).expand(-1, self.n_head, -1, -1).reshape(B * self.n_head, Lx, Ly)
        else:
            attn_mask_cross_input = None
        x0 = self.ln_0(x)
        x0_ = self.self_attn(x0, x0, x0, attn_mask=attn_mask_self_input)[0]
        x = x + x0_

        ##### cross
        x_ = self.attn(query = self.ln_1(x),
                       key = y, 
                       value = y,
                       attn_mask = attn_mask_cross_input)[0]
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        """
        x: L, b, d
        attn_mask: b, L, L
        """
        if attn_mask is not None:
            b, L, _ = attn_mask.shape
            attn_mask_input = attn_mask.unsqueeze(1).expand(-1, self.n_head, -1, -1).reshape(b * self.n_head, L, L)
        else:
            attn_mask_input = None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_input)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        x_ = self.attention(self.ln_1(x), attn_mask)
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttention(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 att_type = 'cross',
                 out_norm = None):
        super().__init__()
        self.att_type = att_type
        if self.att_type == 'self':
            ResidualAttentionBlock = ResidualSelfAttentionBlock(d_model=d_model, n_head=n_head)
            self.layers = nn.ModuleList([
                deepcopy(ResidualAttentionBlock) for _ in range(num_layers)
            ])
        elif self.att_type == 'cross':
            ResidualAttentionBlock = ResidualCrossAttentionBlock(d_model=d_model, n_head=n_head)
            self.modal1_layers = nn.ModuleList([
                deepcopy(ResidualAttentionBlock) for _ in range(num_layers)
            ])
            self.modal2_layers = nn.ModuleList([
                deepcopy(ResidualAttentionBlock) for _ in range(num_layers)
            ])
        self.num_layers = num_layers
        self.norm = out_norm(normalized_shape=d_model) if out_norm is not None else None

    def forward(self, x, y=None, x_mask=None, y_mask=None):
        '''
            x: b, Lx, dx
            y: b, Ly, dy
            x_mask: b, Lx,
            y_mask: b, Ly,
        '''
        x = x.permute(1, 0, 2)
        if self.att_type == 'cross':
            y = y.permute(1, 0, 2)
        if self.att_type == 'self':
            if x_mask is not None:
                attn_mask_self = torch.logical_or(x_mask.unsqueeze(2), x_mask.unsqueeze(1))
            else:
                attn_mask_self = None
            output = x
        elif self.att_type == 'cross':
            if x_mask is None:
                x_mask = torch.zeros_like(x[:, :, 0], dtype=torch.bool).transpose(0, 1)
            if y_mask is None:
                y_mask = torch.zeros_like(y[:, :, 0], dtype=torch.bool).transpose(0, 1)
            attn_mask_self_x = torch.logical_or(x_mask.unsqueeze(2), x_mask.unsqueeze(1))
            attn_mask_self_y = torch.logical_or(y_mask.unsqueeze(2), y_mask.unsqueeze(1))
            attn_mask_cross_x2y = torch.logical_or(x_mask.unsqueeze(2), y_mask.unsqueeze(1))
            attn_mask_cross_y2x = torch.logical_or(y_mask.unsqueeze(2), x_mask.unsqueeze(1))
            x_output = x
            y_output = y
        if self.att_type == 'self':
            for layer in self.layers:
                output = layer(output, attn_mask=attn_mask_self)
        else:
            for i in range(len(self.modal1_layers)):
                x_output_next = self.modal1_layers[i](x_output, y_output, attn_mask_cross=attn_mask_cross_x2y, attn_mask_self=attn_mask_self_x)
                y_output_next = self.modal2_layers[i](y_output, x_output, attn_mask_cross=attn_mask_cross_y2x, attn_mask_self=attn_mask_self_y)
                x_output = x_output_next
                y_output = y_output_next
        if self.att_type == 'self':
            if self.norm is not None:
                # Lx, b, dx -> b, Lx, dx
                output = self.norm(output).permute(1, 0, 2)
                return output
            else:
                return output.permute(1, 0, 2)
        elif self.att_type == 'cross':
            if self.norm is not None:
                # Lx, b, dx -> b, Lx, dx
                x_output = self.norm(x_output).permute(1, 0, 2)
                y_output = self.norm(y_output).permute(1, 0, 2)
                return x_output, y_output
            else:
                return x_output.permute(1, 0, 2), y_output.permute(1, 0, 2)


class ResidualSelfAttentionBlock_v2(nn.Module):
    def __init__(self, d_model, cfgs):
        super(ResidualSelfAttentionBlock_v2, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, cfgs.n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = cfgs.n_head

    def attention(self, x: torch.Tensor):
        """
        x: L, b, d
        """
        return self.attn(x, x, x, need_weights=True, average_attn_weights=True)

    def forward(self, x: torch.Tensor):
        x_, attn_weights = self.attention(self.ln_1(x))
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights # (L, b, d), (b, L, L)

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    r"""Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    else:
        return nn.Dropout(p=p, **kwargs)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout=None):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices):
        r"""Learnable Positional Embedding.

        `emb_indices` are truncated to fit the finite embedding space.

        Args:
            emb_indices: torch.LongTensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings