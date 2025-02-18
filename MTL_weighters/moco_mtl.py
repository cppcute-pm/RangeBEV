import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo_MTL_v1(nn.Module):

    def __init__(self, cfgs, model_loss, device):
        super(MoCo_MTL_v1, self).__init__()

        self.beta = cfgs.MoCo_beta
        self.beta_sigma = cfgs.MoCo_beta_sigma
        self.gamma = cfgs.MoCo_gamma
        self.gamma_sigma = cfgs.MoCo_gamma_sigma
        self.rho = cfgs.MoCo_rho

        self.shared_params_type = cfgs.shared_params_type
        self.task_num = cfgs.task_num

        self.step = 0
        self.grad_index = []
        self.grad_dim = 0
        self.rep_grad = False
        self._compute_grad_dim(model_loss)
        self.y = torch.zeros((self.task_num, self.grad_dim), device=device)
        self.lambd = torch.ones((self.task_num, ), device=device) / self.task_num
    
    def _compute_grad_dim(self, model_loss):
        self.grad_index = []
        for param in self.get_share_params(model_loss):
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def get_share_params(self, model_loss):
        p = []
        if self.shared_params_type == 1:
            p += model_loss.model.image_encoder.module.backbone.parameters()
            p += model_loss.model.pc_encoder.module.encoder.parameters()
        else:
            raise NotImplementedError
        return p
    
    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.grad_index = state_dict['grad_index']
        self.grad_dim = state_dict['grad_dim']
        self.y = state_dict['y'].to(self.y.device)
        self.lambd = state_dict['lambd'].to(self.lambd.device)
    
    def get_state_dict(self,):
        state_dict = {
            'step': self.step,
            'grad_index': self.grad_index,
            'grad_dim': self.grad_dim,
            'y': self.y.cpu(),
            'lambd': self.lambd.cpu()
        }
        return state_dict
    
    def _grad2vec(self, device, model_loss):
        grad = torch.zeros(self.grad_dim, device=device)
        count = 0
        for param in self.get_share_params(model_loss):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, model_loss):
        '''
        mode: backward, autograd
        '''
        grads = torch.zeros((self.task_num, self.grad_dim), device=losses.device)
        for tn in range(self.task_num):
            losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
            grads[tn] = self._grad2vec(device=losses.device, model_loss=model_loss)
            self.zero_grad_share_params(model_loss)
        return grads
    
    def zero_grad_share_params(self, model_loss):
        params = self.get_share_params(model_loss)
        for param in params:
            if param.grad is not None:
                param.grad.data.zero_()
    
    def _reset_grad(self, new_grads, model_loss):
        count = 0
        for param in self.get_share_params(model_loss):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    def backward(self, losses, model_loss, epoch=None):
        self.step += 1
        grads = self._compute_grad(losses, model_loss=model_loss)
        with torch.no_grad():
            for tn in range(self.task_num):
                grads[tn] = grads[tn]/(grads[tn].norm()+1e-8)*losses[tn]
        self.y = self.y - (self.beta/self.step**self.beta_sigma) * (self.y - grads)
        self.lambd = F.softmax(self.lambd - (self.gamma/self.step**self.gamma_sigma) * (self.y@self.y.t()+self.rho*torch.eye(self.task_num).to(self.y.device))@self.lambd, -1)
        new_grads = self.y.t()@self.lambd

        self._reset_grad(new_grads, model_loss)