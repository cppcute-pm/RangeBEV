import torch
import numpy as np
import random
import torch.distributed as dist
import os
import torch.distributed
import torch.nn as nn
list1 = ['embeddings', 'embeddings1', 'embeddings2']
import time

def all_gather_fn(data, gather_type):
    if gather_type in list1:
        with torch.no_grad():
            all_tensors = [torch.zeros_like(data[gather_type]) for _ in range(get_world_size())]
            torch.distributed.all_gather(all_tensors, data[gather_type])
        all_tensors[get_rank()] = data[gather_type]
        all_tensors = torch.cat(all_tensors, dim=0)
        data[gather_type] = all_tensors
    elif gather_type == 'positives_mask':
        with torch.no_grad():
            all_tensors = [torch.zeros_like(data[gather_type]) for _ in range(get_world_size())]
            torch.distributed.all_gather(all_tensors, data[gather_type])
        all_tensors[get_rank()] = data[gather_type]
        all_tensors = torch.block_diag(*all_tensors)
        data[gather_type] = all_tensors
    elif gather_type == 'negatives_mask':
        with torch.no_grad():
            all_tensors = [torch.zeros_like(data[gather_type]) for _ in range(get_world_size())]
            torch.distributed.all_gather(all_tensors, data[gather_type])
        all_tensors[get_rank()] = data[gather_type]
        all_tensors = [torch.logical_not(all_tensors[i]) for i in range(len(all_tensors))]
        all_tensors = torch.block_diag(*all_tensors)
        data[gather_type] = torch.logical_not(all_tensors)
    else:
        raise ValueError(f'Unknown gather type: {gather_type}')
    
    return data

class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_fn, cfgs):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.all_gather_cfgs = cfgs.all_gather_cfgs
    
    def forward(self, data_input, logger, epoch, bn, iter_num, all_epochs, dataset, dataset_cfgs):
        data_output = self.model(data_input)
        if self.all_gather_cfgs.all_gather_flag:
            for gather_type in self.all_gather_cfgs.types:
                if gather_type in data_output.keys():
                    data_output = all_gather_fn(data_output, gather_type)
                elif gather_type in data_input.keys():
                    data_input = all_gather_fn(data_input, gather_type)
                else:
                    raise ValueError(f'Unknown gather type: {gather_type}')
        if 'use_memory_bank' in dataset_cfgs.keys() and dataset_cfgs.use_memory_bank: # assume it's only for KITTI
            curr_seq_ID = data_input['labels'][0][0]
            labels = torch.tensor([frame_ID for seq_ID, frame_ID in data_input['labels']], dtype=torch.int64)
            if 'dist_caculation_type' in dataset_cfgs.keys() and dataset_cfgs.dist_caculation_type == 'all_coords_L2_mean':
                query_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID][:, labels, :]
                query_UTM_coord = query_UTM_coord.cuda()
                database_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID]
                database_UTM_coord = database_UTM_coord.cuda()
                query_2_database_dist = torch.cdist(query_UTM_coord, database_UTM_coord, p=2.0).mean(dim=0, keepdim=False) # (B, DB)
                zero_mask = torch.ge(query_2_database_dist, dataset_cfgs.pose_dist_threshold)
                overlap_ratio = (dataset_cfgs.pose_dist_threshold - query_2_database_dist) * 1.0 / dataset_cfgs.pose_dist_threshold # (B, DB)
                overlap_ratio[zero_mask] = 0.0
                data_input['query_to_database_positives_mask'] = torch.gt(overlap_ratio, dataset_cfgs.pos_overlap_threshold)
                data_input['query_to_database_negatives_mask'] = torch.le(overlap_ratio, dataset_cfgs.neg_overlap_threshold)
                data_input['database_key_embeddings1'] = dataset.memory_bank[curr_seq_ID][0].cuda()
                data_input['database_key_embeddings2'] = dataset.memory_bank[curr_seq_ID][1].cuda()
            else:
                query_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID][labels]
                query_UTM_coord = query_UTM_coord.cuda()
                database_UTM_coord = dataset.UTM_coord_tensor[curr_seq_ID]
                database_UTM_coord = database_UTM_coord.cuda()
                query_2_database_dist = torch.cdist(query_UTM_coord, database_UTM_coord, p=2.0) # (B, DB)
                data_input['query_to_database_positives_mask'] = torch.le(query_2_database_dist, dataset_cfgs.positive_distance)
                data_input['query_to_database_negatives_mask'] = torch.gt(query_2_database_dist, dataset_cfgs.non_negative_distance)
                data_input['database_key_embeddings1'] = dataset.memory_bank[curr_seq_ID][0].cuda()
                data_input['database_key_embeddings2'] = dataset.memory_bank[curr_seq_ID][1].cuda()
        loss = self.loss_fn(data_output, data_input, logger, epoch, bn, iter_num, all_epochs)
        if 'use_memory_bank' in dataset_cfgs.keys() and dataset_cfgs.use_memory_bank:
            curr_seq_ID = data_input['labels'][0][0]
            labels = torch.tensor([frame_ID for seq_ID, frame_ID in data_input['labels']], dtype=torch.int64)
            dataset.memory_bank[curr_seq_ID][0, labels] = data_output['key_embeddings1'].detach().cpu()
            dataset.memory_bank[curr_seq_ID][1, labels] = data_output['key_embeddings2'].detach().cpu()
        return loss

class LossScaler(object):
    state_dict_key = "amp_scaler"

    def __init__(self, use_mp):
        self._scaler = torch.cuda.amp.GradScaler()
        self.use_mp = use_mp

    def __call__(self, loss, optimizer, update_grad=False):
        if self.use_mp:
            self._scaler.scale(loss).backward()
            if update_grad:
                self._scaler.unscale_(optimizer)
                self._scaler.step(optimizer)
                self._scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if update_grad:
                optimizer.step()
                optimizer.zero_grad()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

class LossScaler_v2(object):
    state_dict_key = "amp_scaler"

    def __init__(self, use_mp):
        self._scaler = torch.cuda.amp.GradScaler()
        self.use_mp = use_mp

    def __call__(self, loss, optimizer, epoch, model_loss, mtl_weighter, update_grad=False):
        if self.use_mp:
            new_loss = self._scaler.scale(loss)
            mtl_weighter.backward(new_loss, model_loss, epoch)
            if update_grad:
                self._scaler.unscale_(optimizer)
                self._scaler.step(optimizer)
                self._scaler.update()
                optimizer.zero_grad()
        else:
            mtl_weighter.backward(loss, model_loss, epoch)
            if update_grad:
                optimizer.step()
                optimizer.zero_grad()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def save_model(cfgs, model_without_ddp, optimizer, scheduler, epoch, loss_fn, seed, loss_scaler, train_batch_sampler):
    if is_main_process():
        checkpoint_path = os.path.join(cfgs.model_weight_path, 'epoch_{}.pth'.format(str(epoch)))
    else:
        checkpoint_path = None
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'seed': seed,
        'loss_scaler': loss_scaler.state_dict(),
    }
    if torch.distributed.is_initialized():
        rng_states = {get_rank(): [torch.get_rng_state(), 
                      np.random.get_state(), 
                      random.getstate()]}
        if is_main_process():
            rng_states_list = [None for _ in range(get_world_size())]
        else:
            rng_states_list = None
        torch.distributed.gather_object(rng_states, rng_states_list, dst=0)

        if is_main_process():
            for rng_states in rng_states_list:
                for rank_id, curr_rng_states in rng_states.items():
                    to_save[f'torch_rng_state_{str(rank_id)}'] = curr_rng_states[0]
                    to_save[f'numpy_rng_state_{str(rank_id)}'] = curr_rng_states[1]
                    to_save[f'rng_state_{str(rank_id)}'] = curr_rng_states[2]
    else:
        to_save['torch_rng_state_0'] = torch.get_rng_state()
        to_save['numpy_rng_state_0'] = np.random.get_state()
        to_save['rng_state_0'] = random.getstate()

    if isinstance(loss_fn, nn.Module):
        to_save['loss_fn'] = loss_fn.state_dict()
    if hasattr(train_batch_sampler, 'state_dict'):
        to_save['batch_sampler'] = train_batch_sampler.state_dict()
        
    save_on_master(to_save, checkpoint_path)

def save_model_v2(cfgs, model_without_ddp, optimizer, scheduler, epoch, loss_fn, seed, loss_scaler, MTL_weighter):
    if is_main_process():
        checkpoint_path = os.path.join(cfgs.model_weight_path, 'epoch_{}.pth'.format(str(epoch)))
    else:
        checkpoint_path = None
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'seed': seed,
        'loss_scaler': loss_scaler.state_dict(),
        'MTL_weighter': MTL_weighter.get_state_dict()
    }
    if torch.distributed.is_initialized():
        rng_states = {get_rank(): [torch.get_rng_state(), 
                      np.random.get_state(), 
                      random.getstate()]}
        if is_main_process():
            rng_states_list = [None for _ in range(get_world_size())]
        else:
            rng_states_list = None
        torch.distributed.gather_object(rng_states, rng_states_list, dst=0)

        if is_main_process():
            for rng_states in rng_states_list:
                for rank_id, curr_rng_states in rng_states.items():
                    to_save[f'torch_rng_state_{str(rank_id)}'] = curr_rng_states[0]
                    to_save[f'numpy_rng_state_{str(rank_id)}'] = curr_rng_states[1]
                    to_save[f'rng_state_{str(rank_id)}'] = curr_rng_states[2]
    else:
        to_save['torch_rng_state_0'] = torch.get_rng_state()
        to_save['numpy_rng_state_0'] = np.random.get_state()
        to_save['rng_state_0'] = random.getstate()

    if isinstance(loss_fn, nn.Module):
        to_save['loss_fn'] = loss_fn.state_dict()
    save_on_master(to_save, checkpoint_path)

class BestMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.best_metric = 0.0
        self.best_epoch = 0

    def update(self, metric, epoch):
        self.best_metric = metric
        self.best_epoch = epoch
    
    def get_best_metric(self):
        return self.best_metric
    
    def get_best_epoch(self):
        return self.best_epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeEstimator(object):
    """Computes and stores the average and current value"""
    def __init__(self, start_epoch, end_epoch):
        self.end_epoch = end_epoch
        self.passed_iter_times = 0.0
        self.start_epoch = start_epoch
        self.ave_iters_per_epoch = 0
        self.passed_iters = 0
    
    def epoch_start(self, curr_iter_num_in_epoch, curr_epoch):
        self.ave_iters_per_epoch = (self.passed_iters + curr_iter_num_in_epoch) / (curr_epoch - self.start_epoch + 1)
        self.curr_epoch = curr_epoch
        self.start_time = time.perf_counter()
    
    def end_iter(self, curr_iter_in_epoch):
        end_time = time.perf_counter()
        curr_iter_time = end_time - self.start_time
        self.start_time = end_time
        self.passed_iters += 1
        self.passed_iter_times += curr_iter_time
        ave_iter_time_per_epoch = self.passed_iter_times / self.passed_iters
        left_iters = (self.end_epoch - self.curr_epoch) * self.ave_iters_per_epoch - curr_iter_in_epoch - 1
        left_time = left_iters * ave_iter_time_per_epoch
        spent_time = ave_iter_time_per_epoch * self.passed_iters

        return left_time, spent_time
        