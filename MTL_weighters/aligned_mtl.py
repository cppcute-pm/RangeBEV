import torch
import torch.nn as nn


class Aligned_MTL_v1(nn.Module):

    def __init__(self, cfgs, model_loss, device):
        super(Aligned_MTL_v1, self).__init__()

        self.shared_params_type = cfgs.shared_params_type
        self.task_num = cfgs.task_num

        self.rep_grad = False
        self.grad_index = []
        self.grad_dim = 0
        self._compute_grad_dim(model_loss)
    
    def _reset_grad(self, new_grads, model_loss):
        count = 0
        for param in self.get_share_params(model_loss):
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, model_loss):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        grads = self._compute_grad(losses, model_loss)
        return grads
    
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
        self.grad_index = state_dict['grad_index']
        self.grad_dim = state_dict['grad_dim']
    
    def get_state_dict(self,):
        state_dict = {
            'grad_index': self.grad_index,
            'grad_dim': self.grad_dim,
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
    
    def _backward_new_grads(self, batch_weight, grads=None, model_loss=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads, model_loss)

    def backward(self, losses, model_loss, epoch=None):
        grads = self._get_grads(losses, model_loss)
        M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
        lmbda, V = torch.symeig(M, eigenvectors=True)
        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
        alpha = B.sum(0)
        self._backward_new_grads(batch_weight=alpha, 
                                 grads=grads,
                                 model_loss=model_loss)