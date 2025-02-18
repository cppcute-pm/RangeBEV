import os
os.environ["OMP_NUM_THREADS"]="4"
import argparse
import torch
import numpy as np
import time
import random
import json
import psutil
import objgraph
import gc

from contextlib import nullcontext
import torch.distributed
from models import make_model
from datasets import make_dataloader
from losses import make_loss
from utils import (Config, 
                   BestMeter, 
                   get_logger, 
                   save_model, 
                   make_freeze, 
                   load_pretrained_weights, 
                   make_optimizer, 
                   make_scheduler, 
                   initialize_netvlad_layer,
                   is_main_process,
                   get_rank,
                   get_world_size,
                   ModelWithLoss,
                   LossScaler,
                   generate_overlap_ratio,
                   generate_original_pc_correspondence,
                   generate_original_pc_correspondence_v2,
                   generate_original_pc_correspondence_v3,
                   generate_masks,
                   generate_UTM_overlap_ratio,
                   process_labels,
                   generate_pixel_point_correspondence,
                   generate_pixel_point_correspondence_v2,
                   TimeEstimator)
from datetime import datetime
from evaluator import get_evaluate

import torch.nn.functional as F
import torch.nn as nn
import shutil
import MinkowskiEngine as ME
import wandb
from mmseg.utils import register_all_modules
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ["NCCL_DEBUG"]="INFO"
# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["USE_GLODG"]="1"





# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"






DEBUG = False
MEMORY_CHECK = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train CMVPR')
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data storing file')
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='Path to weight saving file')
    parser.add_argument(
        '--resume_from',
        type=str,
        help='the model weights chose to load')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file')
    # parser.add_argument(
    #     '--need_eval',
    #     dest='need_eval',
    #     action='store_true')
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='random seed')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='whether to use distributed training')
    parser.add_argument(
        '--gpu_num',
        type=str,
        help='num of gpus to use')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0)

    parser.set_defaults(debug=False)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def train_one_epoch(model_loss, 
                    dataloader, 
                    optimizer,
                    loss_scaler, 
                    device, 
                    cfgs,
                    args, 
                    logger, 
                    epoch, 
                    scheduler,
                    te1):
    optimizer.zero_grad()
    model_loss.train()
    #-----------------------------------do the memory check-----------------------------
    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"before into the train, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"before into the train, VMS: {mem_info.vms / 1024 / 1024} MB")

    #-----------------------------------train one epoch----------------------------------
    for bn, data in enumerate(dataloader):
        if MEMORY_CHECK:
            mem_info = p.memory_info()
            print(f"after get batch from dataloader, RSS: {mem_info.rss / 1024 / 1024} MB")
            print(f"after get batch from dataloader, VMS: {mem_info.vms / 1024 / 1024} MB")

        if args.debug and bn == 3:
            break
        iter_num = len(dataloader)

        if bn == 0:
            te1.epoch_start(iter_num, epoch)
        #-----------------------------------data-----------------------------------
        data_input = {}
        if "imgnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr_cfgs) or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            if 'images' in data.keys():
                data_input['images'] = data['images'].to(device)
        if "rendernet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "render_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            if 'render_imgs' in data.keys():
                data_input['render_imgs'] = data['render_imgs'].to(device)
            elif 'range_imgs' in data.keys():
                data_input['render_imgs'] = data['range_imgs'].to(device)
            else:
                pass
        
        if "imagebevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "image_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['image_bevs'] = data['image_bevs'].to(device)
        
        if "pcbevnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr2_cfgs" in cfgs.model_cfgs.keys() and "pc_bev_encoder_type" in cfgs.model_cfgs.cmvpr2_cfgs):
            data_input['pc_bevs'] = data['pc_bevs'].to(device)

        if "pcnet_cfgs" in cfgs.model_cfgs.keys() or ("cmvpr_cfgs" in cfgs.model_cfgs.keys() and "pc_encoder_type" in cfgs.model_cfgs.cmvpr_cfgs):
            # another_input_list = [pc.type(torch.float32).to(device) for pc in data['clouds']]
            another_input_list = [pc.type(torch.float32) for pc in data['clouds']]
            if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                pc_backbone_type = cfgs.model_cfgs.pcnet_cfgs.backbone_type
            else:
                pc_backbone_type = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_type
            if pc_backbone_type=='FPT':
                if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                    radius_max_raw = cfgs.model_cfgs.pcnet_cfgs.backbone_config.radius_max_raw
                    voxel_size = cfgs.model_cfgs.pcnet_cfgs.backbone_config.voxel_size
                else:
                    radius_max_raw = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.radius_max_raw
                    voxel_size = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.voxel_size
                coordinates_input = [coords * radius_max_raw / voxel_size for coords in another_input_list]
                features_input = [torch.ones(coords.shape, device=coords.device, dtype=torch.float32) for coords in another_input_list] # let the z coordinate be the feature
                # features_input = [coords for coords in another_input_list]
                coords, feats = ME.utils.sparse_collate(
                        coordinates_input,
                        features_input,
                        dtype=torch.float32
                    )
                data_input['clouds'] = ME.TensorField(features=feats,
                                    coordinates=coords,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                    device=device)
            elif pc_backbone_type=='MinkUNet':
                if 'pcnet_cfgs' in cfgs.model_cfgs.keys():
                    voxel_size = cfgs.model_cfgs.pcnet_cfgs.backbone_config.voxel_size
                else:
                    voxel_size = cfgs.model_cfgs.cmvpr_cfgs.pc_encoder_cfgs.voxel_size
                coordinates_input = torch.stack(another_input_list, dim=0)
                x, y, z = coordinates_input[..., 0], coordinates_input[..., 1], coordinates_input[..., 2]
                rho = torch.sqrt(x ** 2 + y ** 2) / voxel_size
                phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
                z = z / voxel_size
                coordinates_input = torch.stack((rho, phi, z), dim=-1) # (B, N, 3)
                coordinates_input = [coordinates_input[i] for i in range(coordinates_input.shape[0])]
                features_input = [coords[:, 2:] for coords in another_input_list] # let the z coordinate be the feature
                coords, feats = ME.utils.sparse_collate(
                        coordinates_input,
                        features_input,
                        dtype=torch.float32
                    )
                data_input['clouds'] = ME.TensorField(features=feats,
                                    coordinates=coords,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                    device=device).sparse(quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            else:
                data_input['clouds'] = torch.stack(another_input_list, dim=0).to(device)

        if 'positives_mask' in data.keys():
            data_input['positives_mask'] = data['positives_mask'].to(device)
        
        if 'negatives_mask' in data.keys():
            data_input['negatives_mask'] = data['negatives_mask'].to(device)
        
        if 'gt_mat' in data.keys():
            data_input['gt_mat'] = data['gt_mat'].to(device)
        
        if 'cloud_remove_masks' in data.keys():
            data_input['cloud_remove_masks'] = data['cloud_remove_masks'].to(device) # (B, 4096)

        if 'rgb_depth_labels' in data.keys():
            data_input['rgb_depth_labels'] = data['rgb_depth_labels'].to(device)
        
        if MEMORY_CHECK:
            mem_info = p.memory_info()
            print(f"after to(device), RSS: {mem_info.rss / 1024 / 1024} MB")
            print(f"after to(device), VMS: {mem_info.vms / 1024 / 1024} MB")
        
        with torch.no_grad():
            if "use_overlap_ratio_type" not in cfgs.dataloader_cfgs.dataset_cfgs.keys() or cfgs.dataloader_cfgs.dataset_cfgs.use_overlap_ratio_type == 'modal_dis':
                if "save_for_visualization_pc_overlap" in cfgs.dataloader_cfgs.dataset_cfgs.keys() and cfgs.dataloader_cfgs.dataset_cfgs.save_for_visualization_pc_overlap:
                    data['bn'] = bn
                data_input['overlap_ratio'] = generate_overlap_ratio(device, data, cfgs.model_cfgs, cfgs.dataloader_cfgs.dataset_cfgs, dataset=dataloader.dataset)
            else:
                data_input['overlap_ratio'] = generate_UTM_overlap_ratio(device, data, dataloader, cfgs.dataloader_cfgs.dataset_cfgs)
            if "save_for_visualization" in cfgs.dataloader_cfgs.dataset_cfgs.keys() and cfgs.dataloader_cfgs.dataset_cfgs.save_for_visualization:
                data['bn'] = bn
                data_input['labels'] = torch.tensor(data['labels'])
                data_input['bn'] = bn
                data_input['visualization_batches'] = cfgs.dataloader_cfgs.dataset_cfgs.visualization_batches
            if cfgs.dataloader_cfgs.dataset_cfgs.use_original_pc_correspondence:
                data_input['original_pc_2_many_1'], data_input['original_pc_2_many_2'] = generate_original_pc_correspondence(device, data, cfgs.dataloader_cfgs.dataset_cfgs)
            if 'use_original_pc_correspondence_v2' in cfgs.dataloader_cfgs.dataset_cfgs.keys() and cfgs.dataloader_cfgs.dataset_cfgs.use_original_pc_correspondence_v2:
                data_input['original_pc_2_many_1'], data_input['original_pc_2_many_2'] = generate_original_pc_correspondence_v2(device, data, cfgs.dataloader_cfgs.dataset_cfgs)
            if 'use_original_pc_correspondence_v3' in cfgs.dataloader_cfgs.dataset_cfgs.keys() and cfgs.dataloader_cfgs.dataset_cfgs.use_original_pc_correspondence_v3:
                (data_input['cloud_poses'], 
                 data_input['original_cloud_remove_masks'], 
                 data_input['clouds_original'], 
                 data_input['cloud_poses_original']) = generate_original_pc_correspondence_v3(device, data, cfgs.dataloader_cfgs.dataset_cfgs)
                data_input['local_correspondence_in_k'] = cfgs.dataloader_cfgs.dataset_cfgs.batch_sampler_cfgs.num_k
            data_input['pixel_selection_method'], data_input['pixels_selected_indices'], data_input['points_selected_indices'], data_input['local_positive_mask'], data_input['local_negative_mask'] = generate_pixel_point_correspondence(device, data, cfgs.dataloader_cfgs.dataset_cfgs)
            generate_masks(device, data_input, data, dataloader, cfgs.dataloader_cfgs.dataset_cfgs)
            process_labels(device, data_input, data, dataloader.dataset, cfgs.dataloader_cfgs.dataset_cfgs)
            data_input['pixels_selected_indices'], data_input['render_pixels_selected_indices'], data_input['local_positive_mask'], data_input['local_negative_mask'] = generate_pixel_point_correspondence_v2(device, data, cfgs.dataloader_cfgs.dataset_cfgs)

        del data
            
        if MEMORY_CHECK:
            mem_info = p.memory_info()
            print(f"after generate overlap_ratio and original_pc_2_many, RSS: {mem_info.rss / 1024 / 1024} MB")
            print(f"after generate overlap_ratio and original_pc_2_many, VMS: {mem_info.vms / 1024 / 1024} MB")
        
        #-----------------------------------model and loss and optimizer-----------------------------------
        # with torch.autograd.set_detect_anomaly(True):
        if torch.distributed.is_initialized() and (bn + 1) % cfgs.accumulate_iter != 0:
            my_context = model_loss.no_sync
        else:
            my_context = nullcontext
        with torch.cuda.amp.autocast(enabled=cfgs.use_mp):
            with my_context():
                loss = model_loss(data_input, logger, epoch, bn, iter_num, cfgs.epoch, dataloader.dataset, cfgs.dataloader_cfgs.dataset_cfgs)
                loss = loss / cfgs.accumulate_iter
                loss_scaler(loss, optimizer, update_grad=(bn + 1) % cfgs.accumulate_iter == 0)
        
        if MEMORY_CHECK:
            mem_info = p.memory_info()
            print(f"after forward, backward, step, RSS: {mem_info.rss / 1024 / 1024} MB")
            print(f"after forward, backward, step, VMS: {mem_info.vms / 1024 / 1024} MB")

        # #-----------------------------------optimizer-----------------------------------
        # if (bn + 1) % cfgs.accumulate_iter == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        if cfgs.scheduler_cfgs.scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + float(bn) / iter_num)
        # if (bn + 1) % 50 == 0:
        #     torch.cuda.empty_cache()
        left_time, spent_time = te1.end_iter(bn)
        left_hours, left_rem = divmod(left_time, 3600)
        left_minutes, left_seconds = divmod(left_rem, 60)
        spent_hours, spent_rem = divmod(spent_time, 3600)
        spent_minutes, spent_seconds = divmod(spent_rem, 60)
        logger.info(f'Time Spent: {spent_hours:.0f}h {spent_minutes:.0f}m {spent_seconds:.0f}s     Time Left: {left_hours:.0f}h {left_minutes:.0f}m {left_seconds:.0f}s')
    # torch.cuda.empty_cache()

def main():
    #-----------------------------------set up-----------------------------------

    args = parse_args()
    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)
    # torch.multiprocessing.set_start_method('spawn')

    # device='cpu'

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', world_size=int(args.gpu_num))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # TODO: to test if it's faster
    state_dict = None
    if is_main_process():
        average_op_recall_meter = BestMeter()
    if args.resume_from is not None:
        model_weight_path = os.path.join(args.weight_path, args.resume_from)
        assert os.path.exists(model_weight_path), f'the path [{model_weight_path}] resumed from does not exist'
        cfgs = Config.fromfile(os.path.join(model_weight_path, 'config.py'))
        cfgs.model_weight_path = model_weight_path
        with open(os.path.join(model_weight_path, 'the_best.txt'), 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            best_metric = float(last_line.split()[0])
            best_epoch = int(last_line.split()[1])
            average_op_recall_meter.update(best_metric, best_epoch)
        for i in range(cfgs.epoch):
            model_path = os.path.join(cfgs.model_weight_path, f'epoch_{str(cfgs.epoch - i - 1)}.pth')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                cfgs.start_epoch = cfgs.epoch - i
                break
        torch.manual_seed(state_dict['seed'] + get_rank())
        torch.set_rng_state(state_dict[f'torch_rng_state_{get_rank()}'])
        np.random.seed(state_dict['seed'] + get_rank())
        np.random.set_state(state_dict[f'numpy_rng_state_{get_rank()}'])
        random.seed(state_dict['seed'] + get_rank())
        
        random.setstate(state_dict[f'rng_state_{get_rank()}'])
        args.seed = state_dict['seed']
        if is_main_process():
            run_id = args.resume_from
        logger = get_logger('CoarseFromFine', os.path.join(cfgs.model_weight_path, 'log.txt'))

    else:
        np.random.seed(args.seed + get_rank())
        torch.manual_seed(args.seed + get_rank())
        random.seed(args.seed + get_rank())
        cfgs = Config.fromfile(args.config)
        # s = datetime.now().strftime('%Y%m%d_%H%M%S')
        if is_main_process():
            run_id = wandb.util.generate_id()
            if not args.debug:
                cfgs.model_weight_path = os.path.join(args.weight_path, run_id)
                os.makedirs(cfgs.model_weight_path, exist_ok=True)
                cfgs.dump(os.path.join(cfgs.model_weight_path, 'config.py'))
                with open(os.path.join(cfgs.model_weight_path, 'the_best.txt'), 'a') as f:
                    f.write(f'{average_op_recall_meter.best_metric} {average_op_recall_meter.best_epoch}\n')
                logger = get_logger('CoarseFromFine', os.path.join(cfgs.model_weight_path, 'log.txt'))
            else:
                logger = get_logger('CoarseFromFine', None)
        else:
            logger = get_logger('CoarseFromFine', None)


    #-----------------------------------wandb-----------------------------------
        
    s = datetime.now().strftime('%Y%m%d_%H%M%S')
    if is_main_process():
        wandb.init(
            # set the wandb project where this run will be logged
            project="CFF",
            # entity="my_entity",
            config=cfgs,
            save_code=False,
            # group='wand_test',
            # job_type='train',
            # tags=["bug_test", "test2"],
            name=s,
            notes="fpt, out_layer is 2, no pretrained backbone",
            # dir=".wandb",
            resume='allow',
            reinit=False, # don't reinit the wandb run in the evaluation stage
            magic=False,
            # config_exclude_keys=["train_val", "probability"]
            # config_include_keys=["min_lr", "warmup_epochs"]
            anonymous="never",
            mode="offline",
            allow_val_change=False,
            force=False,
            sync_tensorboard=False,
            # monitor_gym=False,
            id=run_id
        )
        wandb.define_metric("loss", summary="min")
        wandb.define_metric("eval.ave_recall.0.oxford", summary="max")
        wandb.define_metric("val.ave_one_percent_recall.oxford", summary="max")
        wandb.define_metric("eval.ave_recall.0.oxford", summary="max")
        wandb.define_metric("val.ave_one_percent_recall.oxford", summary="max")

    #-----------------------------------dataloader-----------------------------------
    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"before make_dataloader, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"before make_dataloader, VMS: {mem_info.vms / 1024 / 1024} MB")

    dataloader = {}
    dataloader['train'] = make_dataloader(cfgs.dataloader_cfgs, args.data_path, cfgs.start_epoch, 'train')
    if cfgs.train_val:
        dataloader['val'] = make_dataloader(cfgs.dataloader_cfgs, args.data_path, cfgs.start_epoch, 'val')
    logger.info(f'make dataloader done')

    # del dataloader['train'].dataset.official_dataset
    # del dataloader['train'].dataset.queries
    # del dataloader['train'].dataset.lidar2image

    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"after make_dataloader, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"after make_dataloader, VMS: {mem_info.vms / 1024 / 1024} MB")

    #-----------------------------------model----------------------------------
    register_all_modules() # register all modules for mmseg
    model = make_model(cfgs.model_cfgs, device)
    logger.info(f'make model done')
    # objgraph.show_most_common_types()
    # gc.collect()

    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"after make_model, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"after make_model, VMS: {mem_info.vms / 1024 / 1024} MB")

    # objgraph.show_most_common_types()
    #-----------------------------------loss-----------------------------------

    loss_fn = make_loss(cfgs.loss_cfgs, device)
    logger.info(f'make loss done')
    loss_scaler = LossScaler(use_mp=cfgs.use_mp)

    model_loss = ModelWithLoss(model, loss_fn, cfgs) # incase of using ddp when the loss has the parameters to update
    model_loss.to(device)

    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"after make_loss, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"after make_loss, VMS: {mem_info.vms / 1024 / 1024} MB")

    #-----------------------------------freeze-------------------------------------------------

    make_freeze(cfgs.freeze_cfgs, cfgs.model_cfgs, model_loss.model)
    logger.info(f'make model parameters freeze done')

    #-----------------------------------load pretrained weights-----------------------------------
    if not args.resume_from:    
        pretrained_msg = load_pretrained_weights(cfgs.pretrained_cfgs, cfgs.model_cfgs, args.weight_path, model_loss.model, model_loss.loss_fn)
        logger.info(f'load pretrained weights done')
        logger.warning(pretrained_msg)
    
    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"after load pretrained weights, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"after load pretrained weights, VMS: {mem_info.vms / 1024 / 1024} MB")
    
    #-----------------------------------encoder or aggregator special operation-----------------------------------

    # below is for the netvlad initialization
    # ① dive into the cfgs.model_cfgs, search all the keys who's value is 'NetVLAD'
    # ② in the same dictionary level, search the keys end_with 'init'
    if 'imgnet_cfgs' in cfgs.model_cfgs.keys() and cfgs.model_cfgs.imgnet_cfgs.aggregate_type == 'NetVLAD':
        initialize_netvlad_layer(cfgs=cfgs, 
                                 netvlad_cfgs=cfgs.model_cfgs.imgnet_cfgs.netvlad_cfgs, 
                                 device=device, 
                                 dataset=dataloader['train'].dataset, 
                                 backbone_img=model_loss.model.backbone, 
                                 backbone_pc=None, 
                                 netvlader=model_loss.model.aggregator.module)
    elif 'pcnet_cfgs' in cfgs.model_cfgs.keys() and cfgs.model_cfgs.pcnet_cfgs.aggregate_type == 'NetVLAD':
        initialize_netvlad_layer(cfgs=cfgs, 
                                 netvlad_cfgs=cfgs.model_cfgs.pcnet_cfgs.netvlad_cfgs, 
                                 device=device, 
                                 dataset=dataloader['train'].dataset, 
                                 backbone_img=None, 
                                 backbone_pc=model_loss.model.backbone, 
                                 netvlader=model_loss.model.aggregator)
        
    #----------------------------------initialize DDP----------------------------------

    model_loss_without_ddp = model_loss
    if args.distributed:
        model_loss = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_loss)
        model_loss = torch.nn.parallel.DistributedDataParallel(model_loss, device_ids=[args.local_rank], find_unused_parameters=cfgs.find_unused_parameters)
        model_loss_without_ddp = model_loss.module
    
    #-----------------------------------optimizer and scheduler-----------------------------------

    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"before make_optimizer_and_scheduler, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"before make_optimizer and scheduler, VMS: {mem_info.vms / 1024 / 1024} MB")
    optimizer = make_optimizer(cfgs.optimizer_cfgs, cfgs.dataloader_cfgs, model_loss_without_ddp.model, model_loss_without_ddp.loss_fn)
    scheduler = make_scheduler(cfgs.scheduler_cfgs, cfgs.dataloader_cfgs, optimizer)
    logger.info(f'make optimizer and scheduler done')

    if MEMORY_CHECK:
        pid = os.getpid()
        p = psutil.Process(pid)
        mem_info = p.memory_info()
        print(f"after make_optimizer_and_scheduler, RSS: {mem_info.rss / 1024 / 1024} MB")
        print(f"after make_optimizer and scheduler, VMS: {mem_info.vms / 1024 / 1024} MB")

    #-----------------------------------resume from-----------------------------------

    if args.resume_from:
        model_loss_without_ddp.model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['lr_scheduler'])
        if isinstance(loss_fn, nn.Module):
            model_loss_without_ddp.loss_fn.load_state_dict(state_dict['loss_fn'])
        loss_scaler.load_state_dict(state_dict['loss_scaler'])
        if hasattr(dataloader['train'].batch_sampler, 'load_state_dict'):
            dataloader['train'].batch_sampler.load_state_dict(state_dict['batch_sampler'])
        logger.info(f'load resume_from done')

    #-----------------------------------train-----------------------------------
    te1 = TimeEstimator(start_epoch=cfgs.start_epoch, end_epoch=cfgs.epoch)
    for epoch in range(cfgs.start_epoch, cfgs.epoch):

        #-----------------------------------train one epoch-----------------------------------
        if torch.distributed.is_initialized():
            dataloader['train'].batch_sampler.sampler.set_epoch(epoch)
        train_one_epoch(model_loss, 
                    dataloader['train'], 
                    optimizer, 
                    loss_scaler,
                    device, 
                    cfgs,
                    args, 
                    logger, 
                    epoch, 
                    scheduler,
                    te1)
        
        #-----------------------------------scheduler-----------------------------------

        
        if cfgs.scheduler_cfgs.scheduler_type != 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch)
        
        if ((epoch + 1) % cfgs.save_interval == 0) and (not args.debug):
            save_model(cfgs, model_loss_without_ddp.model, optimizer, scheduler, epoch, model_loss_without_ddp.loss_fn, args.seed, loss_scaler, dataloader['train'].batch_sampler)
            logger.info(f'model saving done')
        
        if cfgs.train_val and ((epoch + 1) % cfgs.val_interval == 0):
            logger.info(f'validating the model')
            if is_main_process():
                get_evaluate(model_loss_without_ddp.model, device, cfgs, args.data_path, logger, epoch, 'val', average_op_recall_meter)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
        
        if cfgs.need_eval and ((epoch + 1) % cfgs.eval_interval == 0) :
            logger.info(f'evaluating the model')
            if is_main_process():
                current_metric = get_evaluate(model_loss_without_ddp.model, device, cfgs, args.data_path, logger, epoch, 'eval')
                if average_op_recall_meter.get_best_metric() <= current_metric:
                    previous_best = average_op_recall_meter.get_best_epoch()
                    average_op_recall_meter.update(current_metric, epoch)
                    with open(os.path.join(cfgs.model_weight_path, 'the_best.txt'), 'a') as f:
                        f.write(f'{average_op_recall_meter.best_metric} {average_op_recall_meter.best_epoch}\n')
                    if previous_best != average_op_recall_meter.get_best_epoch():
                        for i in range(max(0, (previous_best - cfgs.eval_interval)), previous_best + 1):
                            to_del_epoch_ckpt = os.path.join(cfgs.model_weight_path, f'epoch_{str(i)}.pth')
                            if os.path.exists(to_del_epoch_ckpt):
                                os.remove(to_del_epoch_ckpt)
                else:
                    for i in range(max(0, epoch - cfgs.eval_interval), epoch):
                        to_del_epoch_ckpt = os.path.join(cfgs.model_weight_path, f'epoch_{str(i)}.pth')
                        if os.path.exists(to_del_epoch_ckpt):
                            if i == average_op_recall_meter.get_best_epoch():
                                continue
                            os.remove(to_del_epoch_ckpt)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            
    wandb.finish()
    print('success!')


if __name__ == '__main__':

    main()
    