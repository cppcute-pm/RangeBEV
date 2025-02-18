import torch

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    # def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, nheads_base_dim=32):
    def __init__(self, cfgs):
        super().__init__()
        self.norm_input = torch.nn.LayerNorm(cfgs.proj_channels)
        
        in_dim = cfgs.proj_channels
        self.boqs = torch.nn.ModuleList([
            # BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
            BoQBlock(in_dim, cfgs.num_queries, nheads=in_dim//cfgs.nheads_base_dim) for _ in range(cfgs.num_layers)])
        
        self.fc = torch.nn.Linear(cfgs.num_layers*cfgs.num_queries, cfgs.row_dim)
        if cfgs.fc2_type == 1:
            self.fc2 = torch.nn.Linear(cfgs.proj_channels * cfgs.row_dim, cfgs.proj_channels)
        elif cfgs.fc2_type == 2:
            self.fc2 = torch.nn.Sequential(
                torch.nn.Linear(cfgs.proj_channels * cfgs.row_dim, cfgs.proj_channels),
                torch.nn.BatchNorm1d(cfgs.proj_channels),
            )
        else:
            raise ValueError('fc2 must be 1 or 2')
        
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = self.fc2(out)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns