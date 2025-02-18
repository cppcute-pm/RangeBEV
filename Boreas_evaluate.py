from utils import get_logger 
from models.builder import make_model
from utils import get_rank, Config, make_freeze
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from mmseg.utils import register_all_modules
from evaluator import get_evaluate_without_wandb
import os

register_all_modules()
SuperGlobal = None
wandb_id = 'u0c9met1'

model_weight_path = os.path.join('/home/pengjianyi/weights/CMVPR', wandb_id)
out_put_pair_idxs = False
epoch_num = 65
model_path = os.path.join(model_weight_path, f'epoch_{epoch_num}.pth')
config_path = os.path.join(model_weight_path, 'config.py')
data_path = '/DATA1/pengjianyi'
cfgs = Config.fromfile(config_path)

cfgs.evaluator_type = 4
cfgs.dataloader_cfgs.dataset_cfgs.img_neighbor_num = 1
cfgs.dataloader_cfgs.dataset_cfgs.boreas_eval_type = 'v2'
cfgs.dataloader_cfgs.dataset_cfgs.true_neighbor_dist = 25.0
cfgs.dataloader_cfgs.dataset_cfgs.eval_query_filename = 'all_test_queries.pickle'
cfgs.dataloader_cfgs.dataset_cfgs.all_coords_filename = 'all_test_UTM_coords.npy'


if 'pcnet_cfgs' in cfgs.model_cfgs:
    if 'pointnextv3_op' not in cfgs.model_cfgs.pcnet_cfgs.backbone_config.keys():
        cfgs.model_cfgs.pcnet_cfgs.backbone_config.pointnextv3_op = 1
elif 'cmvpr_cfgs' in cfgs.model_cfgs:
    if 'pointnextv3_op' not in cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.keys():
        cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.pointnextv3_op = 1
else:
    pass

if SuperGlobal == '2':
    wandb_id_pc = 'u0c9met1'
    model_weight_path_pc = os.path.join('/home/pengjianyi/weights/CMVPR', wandb_id_pc)
    epoch_num_pc = 65
    model_path_pc = os.path.join(model_weight_path_pc, f'epoch_{epoch_num_pc}.pth')
    config_path_pc = os.path.join(model_weight_path_pc, 'config.py')
    cfgs_pc = Config.fromfile(config_path_pc)

device = torch.device('cuda:0')
torch.cuda.set_device(device)
state_dict = torch.load(model_path, map_location='cpu')

torch.manual_seed(state_dict['seed'] + get_rank())
torch.set_rng_state(state_dict[f'torch_rng_state_{get_rank()}'])
np.random.seed(state_dict['seed'] + get_rank())
np.random.set_state(state_dict[f'numpy_rng_state_{get_rank()}'])
random.seed(state_dict['seed'] + get_rank())
random.setstate(state_dict[f'rng_state_{get_rank()}'])

model = make_model(cfgs.model_cfgs, device)
model.load_state_dict(state_dict['model'])
make_freeze(cfgs.freeze_cfgs, cfgs.model_cfgs, model)

if SuperGlobal == '1':
    logger = get_logger('CoarseFromFine', os.path.join(model_weight_path, 'log_evaluate_superglobal_v1.txt'))
    cfgs.evaluator_type = 6
elif SuperGlobal == '2':
    logger = get_logger('CoarseFromFine', os.path.join(model_weight_path, 'log_evaluate_superglobal_v2.txt'))
    cfgs.evaluator_type = 7
    state_dict_pc = torch.load(model_path_pc, map_location='cpu')
    model_pc = make_model(cfgs_pc.model_cfgs, device)
    model_pc.load_state_dict(state_dict_pc['model'])
    make_freeze(cfgs_pc.freeze_cfgs, cfgs_pc.model_cfgs, model_pc)
    cfgs.dataloader_cfgs.dataset_cfgs.train_pc_aug_mode = cfgs_pc.dataloader_cfgs.dataset_cfgs.train_pc_aug_mode
    cfgs.dataloader_cfgs.dataset_cfgs.eval_pc_aug_mode = cfgs_pc.dataloader_cfgs.dataset_cfgs.eval_pc_aug_mode
    cfgs.dataloader_cfgs.dataset_cfgs.pc_dir_name = cfgs_pc.dataloader_cfgs.dataset_cfgs.pc_dir_name
    cfgs.dataloader_cfgs.dataset_cfgs.use_cloud = cfgs_pc.dataloader_cfgs.dataset_cfgs.use_cloud
else:
    logger = get_logger('CoarseFromFine', os.path.join(model_weight_path, 'log_evaluate.txt'))

if SuperGlobal == '2':
    get_evaluate_without_wandb(model, device, cfgs, data_path, logger, epoch_num, 'eval', out_put_pair_idxs, wandb_id, model_pc)
else:
    get_evaluate_without_wandb(model, device, cfgs, data_path, logger, epoch_num, 'eval', out_put_pair_idxs, wandb_id)

print('wandb_id:', wandb_id)
