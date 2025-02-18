import torch.nn as nn
import torch


class GeM(nn.Module):
    def __init__(self, cfgs):
        super(GeM, self).__init__()
        p = cfgs.p
        eps = cfgs.eps
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        args:
            x: shape (B, C, N) for pc, shape (B, C, H, W) for img
        """
        x = torch.flatten(x, start_dim=2)
        return (nn.functional.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1./self.p)).squeeze(-1)

class GeM_mask(nn.Module):
    def __init__(self, cfgs):
        super(GeM_mask, self).__init__()
        p = cfgs.p
        eps = cfgs.eps
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        if 'final_fc' in cfgs.keys(): 
            if cfgs.final_fc == 1:
                self.final_fc = nn.Linear(256, 256, bias=True)
            elif cfgs.final_fc == 2:
                self.final_fc = nn.Sequential(
                    nn.Linear(256, 256, bias=True),
                    nn.BatchNorm1d(256),
                )
            else:
                raise NotImplementedError
        else:
            self.final_fc = None

    def forward(self, x, mask):
        """
        args:
            x: shape (B, C, N) for pc, shape (B, C, H, W) for img
            mask: shape (B, N) for pc, shape (B, H, W) for img
        """
        x = torch.flatten(x, start_dim=2)
        mask = torch.flatten(mask, start_dim=1)
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        # x.masked_fill_(~mask.unsqueeze(1), 0.0)
        x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        x = x.sum(dim=-1)
        mask_2 = torch.count_nonzero(mask, dim=-1)
        x = x / mask_2.unsqueeze(1)
        x = x.pow(1./self.p)

        if self.final_fc is not None:
            x = self.final_fc(x)
        
        return x