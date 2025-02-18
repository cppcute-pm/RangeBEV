import torch.nn as nn
import torch


class Proj1(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj1, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.BatchNorm1d(embeddim), # bn(B, C, N) == bn(B * N, C)
                nn.ReLU(inplace=True), 
                nn.Linear(embeddim, embeddim),
                nn.BatchNorm1d(embeddim))

    def forward(self, x):
        """
        x of shape: (B, C)
        """
        return self.model(x)

class Proj1_mask(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj1_mask, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.BatchNorm1d(embeddim), # bn(B, C, N) == bn(B * N, C)
                nn.ReLU(inplace=True), 
                nn.Linear(embeddim, embeddim),
                nn.BatchNorm1d(embeddim))
        self.embeddim = embeddim

    def forward(self, x, mask):
        """
        x of shape: (B, C)
        mask of shape: (B,)
        """
        mask1 = mask.unsqueeze(1).expand_as(x)
        x_input = torch.masked_select(x, mask1).view(-1, x.size(1))
        temp_out = self.model(x_input)
        x_output = torch.zeros((x.shape[0], self.embeddim), device=temp_out.device, dtype=temp_out.dtype)
        x_output[mask, :] = temp_out
        return x_output

class Proj3_mask(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj3_mask, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.ReLU(inplace=True), 
                nn.Linear(embeddim, embeddim),
                nn.BatchNorm1d(embeddim))
        self.embeddim = embeddim

    def forward(self, x, mask):
        """
        x of shape: (B, C)
        mask of shape: (B,)
        """
        mask1 = mask.unsqueeze(1).expand_as(x)
        x_input = torch.masked_select(x, mask1).view(-1, x.size(1))
        temp_out = self.model(x_input)
        x_output = torch.zeros((x.shape[0], self.embeddim), device=temp_out.device, dtype=temp_out.dtype)
        x_output[mask, :] = temp_out
        return x_output

class Proj4_mask(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj4_mask, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.BatchNorm1d(embeddim),
                nn.Linear(embeddim, embeddim),
                nn.BatchNorm1d(embeddim))
        self.embeddim = embeddim

    def forward(self, x, mask):
        """
        x of shape: (B, C)
        mask of shape: (B,)
        """
        mask1 = mask.unsqueeze(1).expand_as(x)
        x_input = torch.masked_select(x, mask1).view(-1, x.size(1))
        temp_out = self.model(x_input)
        x_output = torch.zeros((x.shape[0], self.embeddim), device=temp_out.device, dtype=temp_out.dtype)
        x_output[mask, :] = temp_out
        return x_output

class Proj5_mask(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj5_mask, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.ReLU(inplace=True),
                nn.Linear(embeddim, embeddim),
                nn.ReLU(inplace=True))
        self.embeddim = embeddim

    def forward(self, x, mask):
        """
        x of shape: (B, C)
        mask of shape: (B,)
        """
        mask1 = mask.unsqueeze(1).expand_as(x)
        x_input = torch.masked_select(x, mask1).view(-1, x.size(1))
        temp_out = self.model(x_input)
        x_output = torch.zeros((x.shape[0], self.embeddim), device=temp_out.device, dtype=temp_out.dtype)
        x_output[mask, :] = temp_out
        return x_output

class Proj6_mask(nn.Module):
    def __init__(self, inputdim, embeddim):
        super(Proj6_mask, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(inputdim, embeddim),
                nn.ReLU(inplace=True),
                nn.Linear(embeddim, embeddim))
        self.embeddim = embeddim

    def forward(self, x, mask):
        """
        x of shape: (B, C)
        mask of shape: (B,)
        """
        mask1 = mask.unsqueeze(1).expand_as(x)
        x_input = torch.masked_select(x, mask1).view(-1, x.size(1))
        temp_out = self.model(x_input)
        x_output = torch.zeros((x.shape[0], self.embeddim), device=temp_out.device, dtype=temp_out.dtype)
        x_output[mask, :] = temp_out
        return x_output

# the code is from: https://github.com/Shubodh/lidar-image-pretrain-VPR/blob/master/models/CLIPModelV1_vit_768.py:33
class Proj2(nn.Module):

    def __init__(
        self,
        inputdim,
        embeddim, 
        dropout=0.1,
    ):
        super(Proj2, self).__init__()
        self.projection = nn.Linear(inputdim, embeddim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embeddim, embeddim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embeddim)
    
    def forward(self, x, mask):
        """
        x of shape: (B, C)
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Proj(nn.Module):

    def __init__(self, embeddim, fuse_type=None, proj_type=1):
        super(Proj, self).__init__()
        if fuse_type == 'concat':
            inputdim = embeddim * 2
        elif fuse_type == 'concat_three':
            inputdim = embeddim * 3
        else:
            inputdim = embeddim
        if proj_type == 1:
            self.proj = Proj1_mask(inputdim, embeddim)
        elif proj_type == 2:
            self.proj = Proj2(inputdim, embeddim)
        elif proj_type == 3:
            self.proj = Proj3_mask(inputdim, embeddim)
        elif proj_type == 4:
            self.proj = Proj4_mask(inputdim, embeddim)
        elif proj_type == 5:
            self.proj = Proj5_mask(inputdim, embeddim)
        elif proj_type == 6:
            self.proj = Proj6_mask(inputdim, embeddim)
        else:
            raise ValueError("Invalid proj_type")
    
    def forward(self, x, mask=None):
        """
        x of shape: (B, C) or (B, C, N) pr (B, C, H, W) 
        """
        if mask is None:
            mask = torch.ones_like(x[:, 0, ...], dtype=torch.bool)
        input_shape = x.shape
        if len(input_shape) == 3:
            B, C, N = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(-1, C)
            mask = mask.reshape(-1)
        elif len(input_shape) == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(-1, C)
            mask = mask.reshape(-1)
        x = self.proj(x, mask)
        if len(input_shape) == 3:
            x = x.reshape(B, N, C)
            x = x.permute(0, 2, 1)
        elif len(input_shape) == 4:
            x = x.reshape(B, H, W, C)
            x = x.permute(0, 3, 1, 2)
        return x
        