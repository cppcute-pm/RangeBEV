import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STCH_MTL_v1(nn.Module):

    def __init__(self, cfgs, model_loss, device):
        super(STCH_MTL_v1, self).__init__()

        self.mu = cfgs.STCH_mu
        self.warmup_epoch = cfgs.STCH_warmup_epoch

        self.shared_params_type = cfgs.shared_params_type
        self.task_num = cfgs.task_num

        self.step = 0 # need to consider the resume
        self.nadir_vector = None
        self.average_loss = 0.0
        self.average_loss_count = 0
        self.device = device
    
    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        if state_dict['nadir_vector'] is not None:
            self.nadir_vector = state_dict['nadir_vector'].to(self.device)
        else:
            self.nadir_vector = None
        if not isinstance(state_dict['average_loss'], float):
            self.average_loss = state_dict['average_loss'].to(self.device)
        else:
            self.average_loss = 0.0
        self.average_loss_count = state_dict['average_loss_count']
    
    def get_state_dict(self,):
        if self.nadir_vector is not None:
            nadir_vector_temp = self.nadir_vector.cpu()
        else:
            nadir_vector_temp = None
        if not isinstance(self.average_loss, float):
            average_loss_temp = self.average_loss.cpu()
        else:
            average_loss_temp = 0.0
        state_dict = {
            'step': self.step,
            'nadir_vector': nadir_vector_temp,
            'average_loss': average_loss_temp,
            'average_loss_count': self.average_loss_count,
        }
        return state_dict
    
    def backward(self, losses, model_loss, epoch=None):
        self.step += 1

        if epoch < self.warmup_epoch:
            loss = torch.mul(torch.log(losses+1e-20), torch.ones_like(losses).to(self.device)).sum()
            loss.backward() 
        elif epoch == self.warmup_epoch:
            loss = torch.mul(torch.log(losses+1e-20), torch.ones_like(losses).to(self.device)).sum()
            self.average_loss += losses.detach() 
            self.average_loss_count += 1
            
            loss.backward()
        else:
            if self.nadir_vector == None:
                self.nadir_vector = self.average_loss / self.average_loss_count
            
            losses = torch.log(losses/self.nadir_vector+1e-20)
            max_term = torch.max(losses.data).detach()
            reg_losses = losses - max_term
           
            loss = self.mu * torch.log(torch.sum(torch.exp(reg_losses/self.mu))) * self.task_num
            loss.backward() 
