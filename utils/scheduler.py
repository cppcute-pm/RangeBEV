import torch
from .misc import get_world_size


def make_scheduler(cfgs, dataloader_cfgs, optimizer):
    batch_size = dataloader_cfgs.batch_sampler_cfgs.batch_size
    if cfgs.scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=cfgs.warmup_epoch, 
                                                                        T_mult=cfgs.T_mult, 
                                                                        eta_min=cfgs.min_lr * batch_size * get_world_size())
    elif cfgs.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=cfgs.step_size, 
                                                    gamma=cfgs.gamma,)
    else:
        raise ValueError('Invalid scheduler type')
    return scheduler