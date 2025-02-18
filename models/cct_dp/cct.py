from torch.hub import load_state_dict_from_url
import torch.nn as nn
from .transformers import TransformerClassifier
from .tokenizer import Tokenizer
from .helpers import pe_check, fc_check
import torch.nn.functional as F
import torch
import math

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'cct_7_3x1_32':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 seq_pool=False,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=seq_pool,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            if positional_embedding == 'learnable':
                state_dict = pe_check(model, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
            state_dict = fc_check(model, state_dict)

                # check patch_embed, if not match， then delete 'patch_embed'
            patch_embed_pretrained = state_dict['tokenizer.conv_layers.0.0.weight']
            Nc1 = patch_embed_pretrained.shape[1]
            Nc2 = model.tokenizer.conv_layers[0][0].weight.shape[1]
            if (Nc1 != Nc2):
                del state_dict['tokenizer.conv_layers.0.0.weight']
                del state_dict['tokenizer.conv_layers.0.0.bias']

            model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


def cct_2(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def cct_2_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_2('cct_2_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_2('cct_2_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_4('cct_4_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_4('cct_4_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_c100(pretrained=False, progress=False,
                      img_size=32, positional_embedding='learnable', num_classes=100,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine_c100(pretrained=False, progress=False,
                           img_size=32, positional_embedding='sine', num_classes=100,
                           *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return cct_7('cct_7_7x2_224', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=102,
                       *args, **kwargs):
    return cct_7('cct_7_7x2_224_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384(pretrained=False, progress=False,
                   img_size=384, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_384', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384_fl(pretrained=False, progress=False,
                      img_size=384, positional_embedding='learnable', num_classes=102,
                      *args, **kwargs):
    return cct_14('cct_14_7x2_384_fl', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)



class CCTv1(CCT):
    
    def __init__(self, cfgs, out_dim):
        embedding_dim = 384
        kernel_size = 7
        stride = max(1, (kernel_size // 2) - 1)
        padding = max(1, (kernel_size // 2))
        super(CCTv1, self).__init__(                 
                                    img_size=cfgs.img_size,
                                    embedding_dim=embedding_dim,
                                    n_input_channels=3,
                                    n_conv_layers=2,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    pooling_kernel_size=3,
                                    pooling_stride=2,
                                    pooling_padding=1,
                                    dropout=0.,
                                    attention_dropout=0.1,
                                    stochastic_depth=0.1,
                                    num_layers=14,
                                    num_heads=6,
                                    mlp_ratio=3.0,
                                    num_classes=2,
                                    positional_embedding='learnable',
                                    seq_pool=True,)
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.out_dim = out_dim
        self.f_fc = nn.Sequential(nn.Linear(embedding_dim, self.out_dim), nn.LayerNorm(self.out_dim))
        self.c_fc = nn.Sequential(nn.Linear(embedding_dim, self.out_dim), nn.LayerNorm(self.out_dim))
        self.ratio = 1.0 if isinstance(cfgs.img_size, int) else float(cfgs.img_size[0] / cfgs.img_size[1]) # H/W
    
    def forward(self, x):
        x = self.tokenizer(x)
        if self.classifier.positional_emb is None and x.size(1) < self.classifier.sequence_length:
            x = F.pad(x, (0, 0, 0, self.classifier.n_channels - x.size(1)), mode='constant', value=0)

        if not self.classifier.seq_pool:
            cls_token = self.classifier.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.classifier.positional_emb is not None:
            x += self.classifier.positional_emb

        x = self.classifier.dropout(x)

        for i, blk in enumerate(self.classifier.blocks):
            x, attn_weight = blk(x) # (B, L, C)
            if self.f_layer == (i + 1):
                f_ebds = x
                B, L, C = f_ebds.shape
                W = int(math.sqrt(L / self.ratio))
                H = int(L / W)
                f_ebds = self.f_fc(f_ebds)
                f_ebds = f_ebds.permute(0, 2, 1).reshape(B, -1, H, W)
                f_ebds_output = F.interpolate(f_ebds, scale_factor=2 ** (3 - i//4), mode='bilinear', align_corners=True)
            if self.c_layer == (i + 1):
                c_ebds = x
                B, L, C = c_ebds.shape
                W = int(math.sqrt(L / self.ratio))
                H = int(L / W)
                c_ebds = self.c_fc(c_ebds)
                c_ebds = c_ebds.permute(0, 2, 1).reshape(B, -1, H, W)
                c_ebds_output = F.interpolate(c_ebds, scale_factor=2 ** (3 - i//4), mode='bilinear', align_corners=True)
                return c_ebds_output, f_ebds_output
            
        x = self.classifier.norm(x)

        if self.classifier.seq_pool:
            global_token = torch.matmul(F.softmax(self.classifier.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = torch.cat((global_token.unsqueeze(1), x), dim=1) # (B, L+1, C)
            return x
        else:
            return x, attn_weight # shouldn't be used


class CCTv2(CCT):
    
    def __init__(self, cfgs, out_dim):
        embedding_dim = 384
        kernel_size = 7
        stride = max(1, (kernel_size // 2) - 1)
        padding = max(1, (kernel_size // 2))
        super(CCTv2, self).__init__(                 
                                    img_size=cfgs.img_size,
                                    embedding_dim=embedding_dim,
                                    n_input_channels=3,
                                    n_conv_layers=2,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    pooling_kernel_size=3,
                                    pooling_stride=2,
                                    pooling_padding=1,
                                    dropout=0.,
                                    attention_dropout=0.1,
                                    stochastic_depth=0.1,
                                    num_layers=14,
                                    num_heads=6,
                                    mlp_ratio=3.0,
                                    num_classes=2,
                                    positional_embedding='learnable',
                                    seq_pool=True,)
        self.layer_1 = cfgs.layer_1
        self.layer_2 = cfgs.layer_2
        self.layer_3 = cfgs.layer_3
        self.out_dim = out_dim
        self.layer_fc_1 = nn.Sequential(nn.Linear(embedding_dim, self.out_dim), nn.LayerNorm(self.out_dim))
        self.layer_fc_2 = nn.Sequential(nn.Linear(embedding_dim, self.out_dim), nn.LayerNorm(self.out_dim))
        self.layer_fc_3 = nn.Sequential(nn.Linear(embedding_dim, self.out_dim), nn.LayerNorm(self.out_dim))
        self.ratio = 1.0 if isinstance(cfgs.img_size, int) else float(cfgs.img_size[0] / cfgs.img_size[1]) # H/W
    
    def forward(self, x):
        x = self.tokenizer(x)
        if self.classifier.positional_emb is None and x.size(1) < self.classifier.sequence_length:
            x = F.pad(x, (0, 0, 0, self.classifier.n_channels - x.size(1)), mode='constant', value=0)

        if not self.classifier.seq_pool:
            cls_token = self.classifier.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.classifier.positional_emb is not None:
            x += self.classifier.positional_emb

        x = self.classifier.dropout(x)

        for i, blk in enumerate(self.classifier.blocks):
            x, attn_weight = blk(x) # (B, L, C)
            if self.layer_1 == (i + 1):
                ebds_1 = x
                B, L, C = ebds_1.shape
                W = int(math.sqrt(L / self.ratio))
                H = int(L / W)
                ebds_1 = self.layer_fc_1(ebds_1)
                ebds_1 = ebds_1.permute(0, 2, 1).reshape(B, -1, H, W)
                ebds_1_output = F.interpolate(ebds_1, scale_factor=2 ** (3 - i//3), mode='bilinear', align_corners=True)
            if self.layer_2 == (i + 1):
                ebds_2 = x
                B, L, C = ebds_2.shape
                W = int(math.sqrt(L / self.ratio))
                H = int(L / W)
                ebds_2 = self.layer_fc_2(ebds_2)
                ebds_2 = ebds_2.permute(0, 2, 1).reshape(B, -1, H, W)
                ebds_2_output = F.interpolate(ebds_2, scale_factor=2 ** (3 - i//3), mode='bilinear', align_corners=True)
            if self.layer_3 == (i + 1):
                ebds_3 = x
                B, L, C = ebds_3.shape
                W = int(math.sqrt(L / self.ratio))
                H = int(L / W)
                ebds_3 = self.layer_fc_3(ebds_3)
                ebds_3 = ebds_3.permute(0, 2, 1).reshape(B, -1, H, W)
                ebds_3_output = F.interpolate(ebds_3, scale_factor=2 ** (3 - i//3), mode='bilinear', align_corners=True)
                return ebds_1_output, ebds_2_output, ebds_3_output
            
        x = self.classifier.norm(x)

        if self.classifier.seq_pool:
            global_token = torch.matmul(F.softmax(self.classifier.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = torch.cat((global_token.unsqueeze(1), x), dim=1) # (B, L+1, C)
            return x
        else:
            return x, attn_weight # shouldn't be used