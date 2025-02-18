import torch.distributed
from .triplet_loss import TripletLoss, TripletLoss_v2, TripletLoss_v3, TripletLoss_v4
from .cmpm_loss import CmpmLoss
from .circle_loss import CircleLoss, CircleLoss_v2
from .infoNCE_loss import InfoNCELoss, InfoNCELoss_v2, MBInfoNCELoss
from .general_contrastive_loss import GeneralContrastiveLoss
from .cfi2p_loss import CFI2P_loss
from .silog_loss import SiLogLoss
from .loftr_focal_loss import LoftrFocalLoss
from .ap_loss import APLoss
from .calibration_loss import CalibrationLoss
from .smooth_rank_ap_loss import SupAP
from utils import AverageMeter, is_main_process, get_world_size
import wandb
import torch.nn as nn
import torch
from .huber_loss import HuberLoss
from .angle_loss import AngleLoss, AngleLossV2
from .kl_loss import KLLoss

class triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(triplet_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = TripletLoss(cfgs)
            self.loss2 = TripletLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = TripletLoss(cfgs)
            self.loss2 = TripletLoss(cfgs)
            self.loss3 = TripletLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = TripletLoss(cfgs)
            self.loss2 = TripletLoss(cfgs)
            self.loss3 = TripletLoss(cfgs)
            self.loss4 = TripletLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = TripletLoss(cfgs)
            self.loss2 = TripletLoss(cfgs)
            self.loss3 = TripletLoss(cfgs)
            self.loss4 = TripletLoss(cfgs)
            self.loss5 = TripletLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = TripletLoss(cfgs)

        self.pair_dist = cfgs.pair_dist_info and cfgs.hard_mining
        if self.pair_dist:
            self.max_pos_pair_dist_meter = AverageMeter()
            self.max_neg_pair_dist_meter = AverageMeter()
            self.mean_pos_pair_dist_meter = AverageMeter()
            self.mean_neg_pair_dist_meter = AverageMeter()
            self.min_pos_pair_dist_meter = AverageMeter()
            self.min_neg_pair_dist_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss, stat = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        elif self.loss_num == 2:
            loss1, stat1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2, stat2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            stat = {k: 0.5 * stat1[k] + 0.5 * stat2[k] for k in stat1}
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1, stat1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2, stat2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3, stat3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            stat = {k: 0.333 * stat1[k] + 0.333 * stat2[k] + 0.333 * stat3[k] for k in stat1}
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1, stat1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2, stat2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3, stat3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4, stat4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            stat = {k: 0.25 * stat1[k] + 0.25 * stat2[k] + 0.25 * stat3[k] + 0.25 * stat4[k] for k in stat1}
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1, stat1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2, stat2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3, stat3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4, stat4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss5, stat5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            self.loss_meter5.update(loss5.detach().cpu().numpy())
            stat = {k: 0.2 * stat1[k] + 0.2 * stat2[k] + 0.2 * stat3[k] + 0.2 * stat4[k] + 0.2 * stat5[k] for k in stat1}
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.pair_dist:
            self.max_pos_pair_dist_meter.update(stat['max_pos_pair_dist'])
            self.max_neg_pair_dist_meter.update(stat['max_neg_pair_dist'])
            self.mean_pos_pair_dist_meter.update(stat['mean_pos_pair_dist'])
            self.mean_neg_pair_dist_meter.update(stat['mean_neg_pair_dist'])
            self.min_pos_pair_dist_meter.update(stat['min_pos_pair_dist'])
            self.min_neg_pair_dist_meter.update(stat['min_neg_pair_dist'])
            logger.info(
                    f'max_pos_pair_dist: {self.max_pos_pair_dist_meter.val:.6f}  '
                    f'max_neg_pair_dist: {self.max_neg_pair_dist_meter.val:.6f}  '
                    f'mean_pos_pair_dist: {self.mean_pos_pair_dist_meter.val:.6f}  '
                    f'mean_neg_pair_dist: {self.mean_neg_pair_dist_meter.val:.6f}  '
                    f'min_pos_pair_dist: {self.min_pos_pair_dist_meter.val:.6f}  '
                    f'min_neg_pair_dist: {self.min_neg_pair_dist_meter.val:.6f}  '
                    )
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.pair_dist:
                stat_wandb = "none"
                stat_wandb = {
                    "max_pos_pair_dist": self.max_pos_pair_dist_meter.avg,
                    "max_neg_pair_dist": self.max_neg_pair_dist_meter.avg,
                    "mean_pos_pair_dist": self.mean_pos_pair_dist_meter.avg,
                    "mean_neg_pair_dist": self.mean_neg_pair_dist_meter.avg,
                    "min_pos_pair_dist": self.min_pos_pair_dist_meter.avg,
                    "min_neg_pair_dist": self.min_neg_pair_dist_meter.avg
                }
                if is_main_process():
                    wandb.log(data={"pair_dist_stat": stat_wandb}, step=epoch)
                self.max_pos_pair_dist_meter.reset()
                self.max_neg_pair_dist_meter.reset()
                self.mean_pos_pair_dist_meter.reset()
                self.mean_neg_pair_dist_meter.reset()
                self.min_pos_pair_dist_meter.reset()
                self.min_neg_pair_dist_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
        return loss

class triplet_v2_lossor(nn.Module):

    def __init__(self, cfgs):
        super(triplet_v2_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = TripletLoss_v2(cfgs)
            self.loss2 = TripletLoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = TripletLoss_v2(cfgs)
            self.loss2 = TripletLoss_v2(cfgs)
            self.loss3 = TripletLoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = TripletLoss_v2(cfgs)
            self.loss2 = TripletLoss_v2(cfgs)
            self.loss3 = TripletLoss_v2(cfgs)
            self.loss4 = TripletLoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = TripletLoss_v2(cfgs)
            self.loss2 = TripletLoss_v2(cfgs)
            self.loss3 = TripletLoss_v2(cfgs)
            self.loss4 = TripletLoss_v2(cfgs)
            self.loss5 = TripletLoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = TripletLoss_v2(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['overlap_ratio'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'].T)
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            self.loss_meter5.update(loss5.detach().cpu().numpy())
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process:
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
        return loss

class G1M_triplet_ap_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G1M_triplet_ap_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss = TripletLoss(cfgs.triplet_loss_cfgs)
        self.ap_loss = APLoss(cfgs.ap_loss_cfgs)
        self.triplet_loss_weight = cfgs.triplet_loss_weight
        self.ap_loss_weight = cfgs.ap_loss_weight
        self.triplet_loss_meter = AverageMeter()
        self.ap_loss_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss, _ = self.triplet_loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        normalized_embeddings = torch.nn.functional.normalize(data_output['embeddings'], p=2.0, dim=1)
        simmat = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        ap_loss = self.ap_loss(simmat, data_input['true_neighbors_mask'])
        loss = self.triplet_loss_weight * triplet_loss + self.ap_loss_weight * ap_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'triplet_loss: {triplet_loss.detach().cpu().numpy():.6f}  '
                f'ap_loss: {ap_loss.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"triplet_loss1": self.triplet_loss_meter.avg,
                                "ap_loss": self.ap_loss_meter.avg}, step=epoch)
            self.triplet_loss_meter.reset()
            self.ap_loss_meter.reset()
        return loss

class G1M_triplet_roadmap_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G1M_triplet_roadmap_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss = TripletLoss(cfgs.triplet_loss_cfgs)
        self.smooth_rank_ap_loss = SupAP(cfgs.smooth_rank_ap_loss_cfgs)
        self.calibration_loss = CalibrationLoss(cfgs.calibration_loss_cfgs)
        self.triplet_loss_weight = cfgs.triplet_loss_weight
        self.smooth_rank_ap_loss_weight = cfgs.smooth_rank_ap_loss_weight
        self.calibration_loss_weight = cfgs.calibration_loss_weight
        self.triplet_loss_meter = AverageMeter()
        self.smooth_rank_ap_loss_meter = AverageMeter()
        self.calibration_loss_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss, _ = self.triplet_loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        normalized_embeddings = torch.nn.functional.normalize(data_output['embeddings'], p=2.0, dim=1)
        simmat = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        smooth_rank_ap_loss = self.smooth_rank_ap_loss(simmat, data_input['true_neighbors_mask'].type(torch.int64))
        calibration_loss = self.calibration_loss(normalized_embeddings, data_input['true_neighbors_mask'], ~data_input['true_neighbors_mask'])
        loss = self.triplet_loss_weight * triplet_loss + self.smooth_rank_ap_loss_weight * smooth_rank_ap_loss + self.calibration_loss_weight * calibration_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'triplet_loss: {triplet_loss.detach().cpu().numpy():.6f}  '
                f'smooth_rank_ap_loss: {smooth_rank_ap_loss.detach().cpu().numpy():.6f}  '
                f'calibration_loss: {calibration_loss.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"triplet_loss": self.triplet_loss_meter.avg,
                                "smooth_rank_ap_loss": self.smooth_rank_ap_loss_meter.avg,
                                "calibration_loss": self.calibration_loss_meter.avg}, step=epoch)
            self.triplet_loss_meter.reset()
            self.smooth_rank_ap_loss_meter.reset()
            self.calibration_loss_meter.reset()
        return loss

class G1M_huber_angle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G1M_huber_angle_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.huber_loss_dist = HuberLoss(cfgs.huber_loss_dist_cfgs)
        if cfgs.huber_loss_angle_cfgs.angle_type is not None:
            self.huber_loss_angle = AngleLoss(cfgs.huber_loss_angle_cfgs)
        else:
            self.huber_loss_angle = AngleLossV2(cfgs.huber_loss_angle_cfgs)
        self.huber_loss_dist_weight = cfgs.huber_loss_dist_weight
        self.huber_loss_angle_weight = cfgs.huber_loss_angle_weight
        self.embedding_normalized = cfgs.embedding_normalized
        self.huber_loss_dist_meter = AverageMeter()
        self.huber_loss_angle_meter = AverageMeter()
        if cfgs.huber_loss_angle_cfgs.angle_type == 1:
            self.huber_loss_angle_fc = torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features)
        elif cfgs.huber_loss_angle_cfgs.angle_type == 2:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 3:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 4:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 5:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 6:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 7:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 8:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),)
        elif cfgs.huber_loss_angle_cfgs.angle_type == 9:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),)
        elif cfgs.huber_loss_angle_cfgs.angle_type == None:
            self.huber_loss_angle_fc = None
        else:
            raise ValueError("Invalid angle_type")
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.embedding_normalized:
            embeddings = torch.nn.functional.normalize(data_output['embeddings'], p=2.0, dim=1)
        else:
            embeddings = data_output['embeddings']
        feat_distmat = torch.cdist(embeddings, embeddings, p=2.0)
        coords = data_input['coords'] # [N, 2]
        true_distmat = torch.cdist(coords, coords, p=2.0) # [N, N]
        positives_mask = data_input['positives_mask']
        huber_loss_dist = self.huber_loss_dist(feat_distmat, positives_mask, true_distmat)

        feat_angle_distmat = embeddings.unsqueeze(1) - embeddings.unsqueeze(0) # [N, N, D]
        true_angle_distmat = coords.unsqueeze(1) - coords.unsqueeze(0) # [N, N, 2]
        if self.huber_loss_angle_fc is not None:
            feat_angle_distmat = feat_angle_distmat.view(-1, feat_angle_distmat.size(-1))
            feat_angle_distmat = self.huber_loss_angle_fc(feat_angle_distmat)
            feat_angle_distmat = feat_angle_distmat.view(embeddings.size(0), embeddings.size(0), -1)
        huber_loss_angle = self.huber_loss_angle(feat_angle_distmat, positives_mask, true_angle_distmat)
        loss = self.huber_loss_dist_weight * huber_loss_dist + self.huber_loss_angle_weight * huber_loss_angle

        self.huber_loss_dist_meter.update(huber_loss_dist.detach().cpu().numpy())
        self.huber_loss_angle_meter.update(huber_loss_angle.detach().cpu().numpy())
        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'huber_loss_dist: {huber_loss_dist.detach().cpu().numpy():.6f}  '
                f'huber_loss_angle: {huber_loss_angle.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"huber_loss_dist": self.huber_loss_dist_meter.avg,
                                "huber_loss_angle": self.huber_loss_angle_meter.avg,}, step=epoch)
            self.huber_loss_dist_meter.reset()
            self.huber_loss_angle_meter.reset()
        return loss


class G2M_triplet_huber_angle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G2M_triplet_huber_angle_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss = TripletLoss(cfgs.triplet_loss_cfgs)
        self.huber_loss_dist = HuberLoss(cfgs.huber_loss_dist_cfgs)
        if cfgs.huber_loss_angle_cfgs.angle_type is not None:
            self.huber_loss_angle = AngleLoss(cfgs.huber_loss_angle_cfgs)
        else:
            self.huber_loss_angle = AngleLossV2(cfgs.huber_loss_angle_cfgs)
        self.triplet_loss_weight = cfgs.triplet_loss_weight
        self.huber_loss_dist_weight = cfgs.huber_loss_dist_weight
        self.huber_loss_angle_weight = cfgs.huber_loss_angle_weight
        self.embedding_normalized = cfgs.embedding_normalized
        self.triplet_loss_meter = AverageMeter()
        self.huber_loss_dist_meter = AverageMeter()
        self.huber_loss_angle_meter = AverageMeter()
        if cfgs.huber_loss_angle_cfgs.angle_type == 1:
            self.huber_loss_angle_fc = torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features)
        elif cfgs.huber_loss_angle_cfgs.angle_type == 2:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 3:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 4:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 5:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 6:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 7:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
            )
        elif cfgs.huber_loss_angle_cfgs.angle_type == 8:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),)
        elif cfgs.huber_loss_angle_cfgs.angle_type == 9:
            self.huber_loss_angle_fc = torch.nn.Sequential(
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.in_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(cfgs.huber_loss_angle_cfgs.out_features, cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.BatchNorm1d(cfgs.huber_loss_angle_cfgs.out_features),
                torch.nn.ReLU(),)
        elif cfgs.huber_loss_angle_cfgs.angle_type == None:
            self.huber_loss_angle_fc = None
        else:
            raise ValueError("Invalid angle_type")
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss, _ = self.triplet_loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        if self.embedding_normalized:
            embeddings = torch.nn.functional.normalize(data_output['embeddings'], p=2.0, dim=1)
        else:
            embeddings = data_output['embeddings']
        feat_distmat = torch.cdist(embeddings, embeddings, p=2.0)
        coords = data_input['coords'] # [N, 2]
        true_distmat = torch.cdist(coords, coords, p=2.0) # [N, N]
        positives_mask = data_input['positives_mask']
        huber_loss_dist = self.huber_loss_dist(feat_distmat, positives_mask, true_distmat)

        feat_angle_distmat = embeddings.unsqueeze(1) - embeddings.unsqueeze(0) # [N, N, D]
        true_angle_distmat = coords.unsqueeze(1) - coords.unsqueeze(0) # [N, N, 2]
        if self.huber_loss_angle_fc is not None:
            feat_angle_distmat = feat_angle_distmat.view(-1, feat_angle_distmat.size(-1))
            feat_angle_distmat = self.huber_loss_angle_fc(feat_angle_distmat)
            feat_angle_distmat = feat_angle_distmat.view(embeddings.size(0), embeddings.size(0), -1)
        huber_loss_angle = self.huber_loss_angle(feat_angle_distmat, positives_mask, true_angle_distmat)
        loss = self.triplet_loss_weight * triplet_loss + self.huber_loss_dist_weight * huber_loss_dist + self.huber_loss_angle_weight * huber_loss_angle

        self.triplet_loss_meter.update(triplet_loss.detach().cpu().numpy())
        self.huber_loss_dist_meter.update(huber_loss_dist.detach().cpu().numpy())
        self.huber_loss_angle_meter.update(huber_loss_angle.detach().cpu().numpy())
        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'triplet_loss: {triplet_loss.detach().cpu().numpy():.6f}  '
                f'huber_loss_dist: {huber_loss_dist.detach().cpu().numpy():.6f}  '
                f'huber_loss_angle: {huber_loss_angle.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"triplet_loss": self.triplet_loss_meter.avg,
                                "huber_loss_dist": self.huber_loss_dist_meter.avg,
                                "huber_loss_angle": self.huber_loss_angle_meter.avg,}, step=epoch)
            self.triplet_loss_meter.reset()
            self.huber_loss_dist_meter.reset()
            self.huber_loss_angle_meter.reset()
        return loss


class G1M_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G1M_triplet_circle_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss = TripletLoss(cfgs.triplet_loss_cfgs)
        self.circle_loss = CircleLoss(cfgs.circle_loss_cfgs)
        self.triplet_loss_weight = cfgs.triplet_loss_weight
        self.circle_loss_weight = cfgs.circle_loss_weight
        self.triplet_loss_meter = AverageMeter()
        self.circle_loss_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss, _ = self.triplet_loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        normalized_embeddings = torch.nn.functional.normalize(data_output['embeddings'], p=2.0, dim=1)
        circle_loss = self.circle_loss(normalized_embeddings, normalized_embeddings, data_input['positives_mask'], data_input['negatives_mask'])
        loss = self.triplet_loss_weight * triplet_loss + self.circle_loss_weight * circle_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'triplet_loss: {triplet_loss.detach().cpu().numpy():.6f}  '
                f'circle_loss: {circle_loss.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"triplet_loss": self.triplet_loss_meter.avg,
                                "circle_loss": self.circle_loss_meter.avg}, step=epoch)
            self.triplet_loss_meter.reset()
            self.circle_loss_meter.reset()
        return loss

class MB_triplet_v2_infonce_lossor(nn.Module):

    def __init__(self, cfgs):
        super(MB_triplet_v2_infonce_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss_1 = TripletLoss_v2(cfgs.v2_triplet_loss_cfgs)
        self.triplet_loss_2 = TripletLoss_v2(cfgs.v2_triplet_loss_cfgs)
        self.mb_infonce_loss_1 = MBInfoNCELoss(cfgs.mb_infonce_loss_cfgs)
        self.mb_infonce_loss_2 = MBInfoNCELoss(cfgs.mb_infonce_loss_cfgs)
        self.v2_triplet_loss_weight = cfgs.v2_triplet_loss_weight
        self.mb_infonce_loss_weight = cfgs.mb_infonce_loss_weight
        self.v2_triplet_loss_meter1 = AverageMeter()
        self.v2_triplet_loss_meter2 = AverageMeter()
        self.mb_infornce_loss_meter1 = AverageMeter()
        self.mb_infornce_loss_meter2 = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        v2_triplet_loss_1 = self.triplet_loss_1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        v2_triplet_loss_2 = self.triplet_loss_2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        mb_infonce_loss_1 = self.mb_infonce_loss_1(data_output['embeddings1'], data_output['key_embeddings2'], data_input['database_key_embeddings2'], data_input['query_to_database_positives_mask'], data_input['query_to_database_negatives_mask'])
        mb_infonce_loss_2 = self.mb_infonce_loss_2(data_output['embeddings2'], data_output['key_embeddings1'], data_input['database_key_embeddings1'], data_input['query_to_database_positives_mask'], data_input['query_to_database_negatives_mask'])
        loss = self.v2_triplet_loss_weight * (0.5 * v2_triplet_loss_1 + 0.5 * v2_triplet_loss_2) + self.mb_infonce_loss_weight * (0.5 * mb_infonce_loss_1 + 0.5 * mb_infonce_loss_2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.v2_triplet_loss_meter1.update(v2_triplet_loss_1.detach().cpu().numpy())
        self.v2_triplet_loss_meter2.update(v2_triplet_loss_2.detach().cpu().numpy())
        self.mb_infornce_loss_meter1.update(mb_infonce_loss_1.detach().cpu().numpy())
        self.mb_infornce_loss_meter2.update(mb_infonce_loss_2.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'v2_triplet_loss: {(0.5 * v2_triplet_loss_1 + 0.5 * v2_triplet_loss_2).detach().cpu().numpy():.6f}  '
                f'mb_infonce_loss: {((0.5 * mb_infonce_loss_1 + 0.5 * mb_infonce_loss_2)).detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"v2_triplet_loss": 0.5 * (self.v2_triplet_loss_meter1.avg + self.v2_triplet_loss_meter2.avg),
                                "mb_infornce_loss": 0.5 * (self.mb_infornce_loss_meter1.avg + self.mb_infornce_loss_meter2.avg)}, step=epoch)
            self.v2_triplet_loss_meter1.reset()
            self.v2_triplet_loss_meter2.reset()
            self.mb_infornce_loss_meter1.reset()
            self.mb_infornce_loss_meter2.reset()
        return loss

class MBKL_triplet_v2_infonce_lossor(nn.Module):

    def __init__(self, cfgs):
        super(MBKL_triplet_v2_infonce_lossor, self).__init__()
        self.loss_meter = AverageMeter()
        self.triplet_loss_1 = TripletLoss_v2(cfgs.v2_triplet_loss_cfgs)
        self.triplet_loss_2 = TripletLoss_v2(cfgs.v2_triplet_loss_cfgs)
        self.mb_infonce_loss_1 = MBInfoNCELoss(cfgs.mb_infonce_loss_cfgs)
        self.mb_infonce_loss_2 = MBInfoNCELoss(cfgs.mb_infonce_loss_cfgs)
        self.kl_loss = KLLoss(cfgs.kl_loss_cfgs)
        self.v2_triplet_loss_weight = cfgs.v2_triplet_loss_weight
        self.mb_infonce_loss_weight = cfgs.mb_infonce_loss_weight
        self.kl_loss_weight = cfgs.kl_loss_weight
        self.v2_triplet_loss_meter1 = AverageMeter()
        self.v2_triplet_loss_meter2 = AverageMeter()
        self.mb_infornce_loss_meter1 = AverageMeter()
        self.mb_infornce_loss_meter2 = AverageMeter()
        self.kl_loss_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        v2_triplet_loss_1 = self.triplet_loss_1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        v2_triplet_loss_2 = self.triplet_loss_2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        mb_infonce_loss_1 = self.mb_infonce_loss_1(data_output['embeddings1'], data_output['key_embeddings2'], data_input['database_key_embeddings2'], data_input['query_to_database_positives_mask'], data_input['query_to_database_negatives_mask'])
        mb_infonce_loss_2 = self.mb_infonce_loss_2(data_output['embeddings2'], data_output['key_embeddings1'], data_input['database_key_embeddings1'], data_input['query_to_database_positives_mask'], data_input['query_to_database_negatives_mask'])
        kl_loss = self.kl_loss(data_output['embeddings1'], 
                               data_output['embeddings2'], 
                               data_output['key_embeddings2'], 
                               data_output['key_embeddings1'],
                               data_input['database_key_embeddings2'],
                               data_input['database_key_embeddings1'],
                               data_input['query_to_database_positives_mask'],
                               data_input['query_to_database_negatives_mask'])
        
        loss = self.v2_triplet_loss_weight * (0.5 * v2_triplet_loss_1 + 0.5 * v2_triplet_loss_2) + self.mb_infonce_loss_weight * (0.5 * mb_infonce_loss_1 + 0.5 * mb_infonce_loss_2) + self.kl_loss_weight * kl_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.v2_triplet_loss_meter1.update(v2_triplet_loss_1.detach().cpu().numpy())
        self.v2_triplet_loss_meter2.update(v2_triplet_loss_2.detach().cpu().numpy())
        self.mb_infornce_loss_meter1.update(mb_infonce_loss_1.detach().cpu().numpy())
        self.mb_infornce_loss_meter2.update(mb_infonce_loss_2.detach().cpu().numpy())
        self.kl_loss_meter.update(kl_loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                f'v2_triplet_loss: {(0.5 * v2_triplet_loss_1 + 0.5 * v2_triplet_loss_2).detach().cpu().numpy():.6f}  '
                f'mb_infonce_loss: {((0.5 * mb_infonce_loss_1 + 0.5 * mb_infonce_loss_2)).detach().cpu().numpy():.6f}  '
                f'kl_loss: {kl_loss.detach().cpu().numpy():.6f}  '
                )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if is_main_process():
                wandb.log(data={"v2_triplet_loss": 0.5 * (self.v2_triplet_loss_meter1.avg + self.v2_triplet_loss_meter2.avg),
                                "mb_infornce_loss": 0.5 * (self.mb_infornce_loss_meter1.avg + self.mb_infornce_loss_meter2.avg),
                                "kl_loss": self.kl_loss_meter.avg}, step=epoch)
            self.v2_triplet_loss_meter1.reset()
            self.v2_triplet_loss_meter2.reset()
            self.mb_infornce_loss_meter1.reset()
            self.mb_infornce_loss_meter2.reset()
            self.kl_loss_meter.reset()
        return loss

class triplet_v3_lossor(nn.Module):
    
    def __init__(self, cfgs):
        super(triplet_v3_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = TripletLoss_v3(cfgs)
            self.loss2 = TripletLoss_v3(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = TripletLoss_v3(cfgs)
            self.loss2 = TripletLoss_v3(cfgs)
            self.loss3 = TripletLoss_v3(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = TripletLoss_v3(cfgs)
            self.loss2 = TripletLoss_v3(cfgs)
            self.loss3 = TripletLoss_v3(cfgs)
            self.loss4 = TripletLoss_v3(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = TripletLoss_v3(cfgs)
            self.loss2 = TripletLoss_v3(cfgs)
            self.loss3 = TripletLoss_v3(cfgs)
            self.loss4 = TripletLoss_v3(cfgs)
            self.loss5 = TripletLoss_v3(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = TripletLoss_v3(cfgs)

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['overlap_ratio'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            self.loss_meter5.update(loss5.detach().cpu().numpy())
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process:
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
        return loss

class triplet_v4_lossor(nn.Module):

    def __init__(self, cfgs):
        super(triplet_v4_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = TripletLoss_v4(cfgs)
            self.loss2 = TripletLoss_v4(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = TripletLoss_v4(cfgs)
            self.loss2 = TripletLoss_v4(cfgs)
            self.loss3 = TripletLoss_v4(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = TripletLoss_v4(cfgs)
            self.loss2 = TripletLoss_v4(cfgs)
            self.loss3 = TripletLoss_v4(cfgs)
            self.loss4 = TripletLoss_v4(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = TripletLoss_v4(cfgs)
            self.loss2 = TripletLoss_v4(cfgs)
            self.loss3 = TripletLoss_v4(cfgs)
            self.loss4 = TripletLoss_v4(cfgs)
            self.loss5 = TripletLoss_v4(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = TripletLoss_v4(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['overlap_ratio'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'].T)
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            self.loss_meter5.update(loss5.detach().cpu().numpy())
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process:
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
        return loss


class cmpm_lossor(nn.Module):

    def __init__(self, cfgs):
        super(cmpm_lossor, self).__init__()
        self.loss = CmpmLoss(cfgs)
        self.loss_meter = AverageMeter()
        self.avg_sim = cfgs.avg_sim_info 
        if self.avg_sim:
            self.pos_avg_sim_meter = AverageMeter()
            self.neg_avg_sim_meter = AverageMeter()
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        loss, stat = self.loss(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'])
        self.loss_meter.update(loss.detach().cpu().numpy())
        logger.info(
                f'epoch[{all_epochs}|{epoch}]  '
                f'iter[{bn}|{iter_num}]  '
                f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                )
        if self.avg_sim:
            self.pos_avg_sim_meter.update(stat['pos_avg_sim'])
            self.neg_avg_sim_meter.update(stat['neg_avg_sim'])
            logger.info(
                f'pos_avg_sim: {self.pos_avg_sim_meter.val:.6f}  '
                f'neg_avg_sim: {self.neg_avg_sim_meter.val:.6f}  '
                )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg},step=epoch)
            self.loss_meter.reset()
            if self.avg_sim:
                stat_wandb = {
                    "pos_avg_sim": self.pos_avg_sim_meter.avg,
                    "neg_avg_sim": self.neg_avg_sim_meter.avg,
                }
                if is_main_process():
                    wandb.log(data={"avg_sim_stat": stat_wandb}, step=epoch)
                self.pos_avg_sim_meter.reset()
                self.neg_avg_sim_meter.reset()

        return loss

class circle_lossor(nn.Module):
    
    def __init__(self, cfgs):
        super(circle_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = CircleLoss(cfgs)
            self.loss2 = CircleLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = CircleLoss(cfgs)
            self.loss2 = CircleLoss(cfgs)
            self.loss3 = CircleLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = CircleLoss(cfgs)
            self.loss2 = CircleLoss(cfgs)
            self.loss3 = CircleLoss(cfgs)
            self.loss4 = CircleLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = CircleLoss(cfgs)
            self.loss2 = CircleLoss(cfgs)
            self.loss3 = CircleLoss(cfgs)
            self.loss4 = CircleLoss(cfgs)
            self.loss5 = CircleLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = CircleLoss(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()

        return loss      
        
class infonce_lossor(nn.Module):
    
    def __init__(self, cfgs):
        super(infonce_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = InfoNCELoss(cfgs)
            self.loss2 = InfoNCELoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = InfoNCELoss(cfgs)
            self.loss2 = InfoNCELoss(cfgs)
            self.loss3 = InfoNCELoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = InfoNCELoss(cfgs)
            self.loss2 = InfoNCELoss(cfgs)
            self.loss3 = InfoNCELoss(cfgs)
            self.loss4 = InfoNCELoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = InfoNCELoss(cfgs)
            self.loss2 = InfoNCELoss(cfgs)
            self.loss3 = InfoNCELoss(cfgs)
            self.loss4 = InfoNCELoss(cfgs)
            self.loss5 = InfoNCELoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = InfoNCELoss(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['positives_mask'], data_input['negatives_mask'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
                
        return loss

class infonce_v2_lossor(nn.Module):
    
    def __init__(self, cfgs):
        super(infonce_v2_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = InfoNCELoss_v2(cfgs)
            self.loss2 = InfoNCELoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = InfoNCELoss_v2(cfgs)
            self.loss2 = InfoNCELoss_v2(cfgs)
            self.loss3 = InfoNCELoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = InfoNCELoss_v2(cfgs)
            self.loss2 = InfoNCELoss_v2(cfgs)
            self.loss3 = InfoNCELoss_v2(cfgs)
            self.loss4 = InfoNCELoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = InfoNCELoss_v2(cfgs)
            self.loss2 = InfoNCELoss_v2(cfgs)
            self.loss3 = InfoNCELoss_v2(cfgs)
            self.loss4 = InfoNCELoss_v2(cfgs)
            self.loss5 = InfoNCELoss_v2(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = InfoNCELoss_v2(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['overlap_ratio'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            self.loss_meter1.update(loss1.detach().cpu().numpy())
            self.loss_meter2.update(loss2.detach().cpu().numpy())
            self.loss_meter3.update(loss3.detach().cpu().numpy())
            self.loss_meter4.update(loss4.detach().cpu().numpy())
            self.loss_meter5.update(loss5.detach().cpu().numpy())
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
        return loss

class general_contrastive_lossor(nn.Module):
    
    def __init__(self, cfgs):
        super(general_contrastive_lossor, self).__init__()
        if '_cm' in cfgs.loss_type:
            self.loss_num = 2
        elif '_fsm' in cfgs.loss_type: # for the case of using 1 fusion loss and 2 single loss
            self.loss_num = 3
        elif '_csm' in cfgs.loss_type: # for the case of 2 single loss and 2 cross loss
            self.loss_num = 4
        elif '_fcsm' in cfgs.loss_type: # for the case of using 1 fusion loss, 2 cross loss and 2 single loss
            self.loss_num = 5
        else:
            self.loss_num = 1
        self.loss_meter = AverageMeter()
        if self.loss_num == 2:
            self.loss1 = GeneralContrastiveLoss(cfgs)
            self.loss2 = GeneralContrastiveLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
        elif self.loss_num == 3:
            self.loss1 = GeneralContrastiveLoss(cfgs)
            self.loss2 = GeneralContrastiveLoss(cfgs)
            self.loss3 = GeneralContrastiveLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
        elif self.loss_num == 4:
            self.loss1 = GeneralContrastiveLoss(cfgs)
            self.loss2 = GeneralContrastiveLoss(cfgs)
            self.loss3 = GeneralContrastiveLoss(cfgs)
            self.loss4 = GeneralContrastiveLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
        elif self.loss_num == 5:
            self.loss1 = GeneralContrastiveLoss(cfgs)
            self.loss2 = GeneralContrastiveLoss(cfgs)
            self.loss3 = GeneralContrastiveLoss(cfgs)
            self.loss4 = GeneralContrastiveLoss(cfgs)
            self.loss5 = GeneralContrastiveLoss(cfgs)
            self.loss_meter1 = AverageMeter()
            self.loss_meter2 = AverageMeter()
            self.loss_meter3 = AverageMeter()
            self.loss_meter4 = AverageMeter()
            self.loss_meter5 = AverageMeter()
        else:
            self.loss = GeneralContrastiveLoss(cfgs)
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if self.loss_num == 1:
            loss = self.loss(data_output['embeddings'], data_output['embeddings'], data_input['overlap_ratio'])
        elif self.loss_num == 2:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss = 0.5 * loss1 + 0.5 * loss2
        elif self.loss_num == 3:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            loss = 0.333 * loss1 + 0.333 * loss2 + 0.333 * loss3
        elif self.loss_num == 4:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss = 0.25 * loss1 + 0.25 * loss2 + 0.25 * loss3 + 0.25 * loss4
        elif self.loss_num == 5:
            loss1 = self.loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss2 = self.loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss3 = self.loss3(data_output['embeddings1'], data_output['embeddings1'], data_input['overlap_ratio'])
            loss4 = self.loss4(data_output['embeddings2'], data_output['embeddings2'], data_input['overlap_ratio'])
            loss5 = self.loss5(data_output['embeddings3'], data_output['embeddings3'], data_input['overlap_ratio'])
            loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.2 * loss4 + 0.2 * loss5
        self.loss_meter.update(loss.detach().cpu().numpy())
        if self.loss_num == 2:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 3:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 4:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    )
        elif self.loss_num == 5:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'loss1: {loss1.detach().cpu().numpy():.6f}  '
                    f'loss2: {loss2.detach().cpu().numpy():.6f}  '
                    f'loss3: {loss3.detach().cpu().numpy():.6f}  '
                    f'loss4: {loss4.detach().cpu().numpy():.6f}  '
                    f'loss5: {loss5.detach().cpu().numpy():.6f}  '
                    )
        else:
            logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            if self.loss_num == 2:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
            elif self.loss_num == 3:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
            elif self.loss_num == 4:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
            elif self.loss_num == 5:
                if is_main_process():
                    wandb.log(data={"loss1": self.loss_meter1.avg,
                                    "loss2": self.loss_meter2.avg,
                                    "loss3": self.loss_meter3.avg,
                                    "loss4": self.loss_meter4.avg,
                                    "loss5": self.loss_meter5.avg}, step=epoch)
                self.loss_meter1.reset()
                self.loss_meter2.reset()
                self.loss_meter3.reset()
                self.loss_meter4.reset()
                self.loss_meter5.reset()
                
        return loss

class two_general_contrastive_lossor(nn.Module):

    def __init__(self, cfgs):
        super(two_general_contrastive_lossor, self).__init__()
        self.gc_loss1 = GeneralContrastiveLoss(cfgs)
        self.gc_loss2 = GeneralContrastiveLoss(cfgs)
        self.gc_loss3 = GeneralContrastiveLoss(cfgs)
        self.gc_loss4 = GeneralContrastiveLoss(cfgs)

        self.loss_meter = AverageMeter()
        self.gc_loss1_meter = AverageMeter()
        self.gc_loss2_meter = AverageMeter()
        self.gc_loss3_meter = AverageMeter()
        self.gc_loss4_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        gc_loss1 = self.gc_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        gc_loss2 = self.gc_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'])
        gc_loss3 = self.gc_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        gc_loss4 = self.gc_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'])
        loss = (0.5 * (0.5 * gc_loss1 + 0.5 * gc_loss2)
                + 0.5 * (0.5 * gc_loss3 + 0.5 * gc_loss4))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.gc_loss1_meter.update(gc_loss1.detach().cpu().numpy())
        self.gc_loss2_meter.update(gc_loss2.detach().cpu().numpy())
        self.gc_loss3_meter.update(gc_loss3.detach().cpu().numpy())
        self.gc_loss4_meter.update(gc_loss4.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'gc_loss1: {gc_loss1.detach().cpu().numpy():.6f}  '
                    f'gc_loss2: {gc_loss2.detach().cpu().numpy():.6f}  '
                    f'gc_loss3: {gc_loss3.detach().cpu().numpy():.6f}  '
                    f'gc_loss4: {gc_loss4.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"gc_loss1": self.gc_loss1_meter.avg,
                                "gc_loss2": self.gc_loss2_meter.avg,
                                "gc_loss3": self.gc_loss3_meter.avg,
                                "gc_loss4": self.gc_loss4_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.gc_loss1_meter.reset()
            self.gc_loss2_meter.reset()
            self.gc_loss3_meter.reset()
            self.gc_loss4_meter.reset()
        return loss

class GL_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.max_pos_pair_dist_meter = AverageMeter()
        self.max_neg_pair_dist_meter = AverageMeter()
        self.mean_pos_pair_dist_meter = AverageMeter()
        self.mean_neg_pair_dist_meter = AverageMeter()
        self.min_pos_pair_dist_meter = AverageMeter()
        self.min_neg_pair_dist_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'embeddings3' in data_output.keys():
            triplet_loss1_c, triplet_loss_stat1_c = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2_c, triplet_loss_stat2_c = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss1_f, triplet_loss_stat1_f = self.triplet_loss1(data_output['embeddings3'], data_output['embeddings4'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2_f, triplet_loss_stat2_f = self.triplet_loss2(data_output['embeddings4'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss1 = 0.5 * triplet_loss1_c + 0.5 * triplet_loss1_f
            triplet_loss2 = 0.5 * triplet_loss2_c + 0.5 * triplet_loss2_f
            triplet_loss_stat1 = {k: 0.5 * triplet_loss_stat1_c[k] + 0.5 * triplet_loss_stat1_f[k] for k in triplet_loss_stat1_c}
            triplet_loss_stat2 = {k: 0.5 * triplet_loss_stat2_c[k] + 0.5 * triplet_loss_stat2_f[k] for k in triplet_loss_stat2_c}
        else:
            triplet_loss1, triplet_loss_stat1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2, triplet_loss_stat2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        triplet_loss_stat = {k: 0.5 * triplet_loss_stat1[k] + 0.5 * triplet_loss_stat2[k] for k in triplet_loss_stat1}
        self.max_pos_pair_dist_meter.update(triplet_loss_stat['max_pos_pair_dist'])
        self.max_neg_pair_dist_meter.update(triplet_loss_stat['max_neg_pair_dist'])
        self.mean_pos_pair_dist_meter.update(triplet_loss_stat['mean_pos_pair_dist'])
        self.mean_neg_pair_dist_meter.update(triplet_loss_stat['mean_neg_pair_dist'])
        self.min_pos_pair_dist_meter.update(triplet_loss_stat['min_pos_pair_dist'])
        self.min_neg_pair_dist_meter.update(triplet_loss_stat['min_neg_pair_dist'])
        logger.info(
                f'max_pos_pair_dist: {self.max_pos_pair_dist_meter.val:.6f}  '
                f'max_neg_pair_dist: {self.max_neg_pair_dist_meter.val:.6f}  '
                f'mean_pos_pair_dist: {self.mean_pos_pair_dist_meter.val:.6f}  '
                f'mean_neg_pair_dist: {self.mean_neg_pair_dist_meter.val:.6f}  '
                f'min_pos_pair_dist: {self.min_pos_pair_dist_meter.val:.6f}  '
                f'min_neg_pair_dist: {self.min_neg_pair_dist_meter.val:.6f}  '
                )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
            if is_main_process():
                stat_wandb = {
                    "max_pos_pair_dist": self.max_pos_pair_dist_meter.avg,
                    "max_neg_pair_dist": self.max_neg_pair_dist_meter.avg,
                    "mean_pos_pair_dist": self.mean_pos_pair_dist_meter.avg,
                    "mean_neg_pair_dist": self.mean_neg_pair_dist_meter.avg,
                    "min_pos_pair_dist": self.min_pos_pair_dist_meter.avg,
                    "min_neg_pair_dist": self.min_neg_pair_dist_meter.avg
                }
                wandb.log(data={"pair_dist_stat": stat_wandb}, step=epoch)
            self.max_pos_pair_dist_meter.reset()
            self.max_neg_pair_dist_meter.reset()
            self.mean_pos_pair_dist_meter.reset()
            self.mean_neg_pair_dist_meter.reset()
            self.min_pos_pair_dist_meter.reset()
            self.min_neg_pair_dist_meter.reset()
        return loss

class GL_v2_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_v2_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'embeddings3' in data_output.keys():
            triplet_loss1_c = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2_c = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
            triplet_loss1_f = self.triplet_loss1(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
            triplet_loss2_f = self.triplet_loss2(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
            triplet_loss1 = 0.5 * triplet_loss1_c + 0.5 * triplet_loss1_f
            triplet_loss2 = 0.5 * triplet_loss2_c + 0.5 * triplet_loss2_f
        else:
            triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class GL_v2_triplet_v2_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_v2_triplet_v2_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss_v2(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss_v2(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)

        local_positive_mask = data_output['local_positive_mask']
        local_negative_mask = data_output['local_negative_mask']
        circle_loss1 = self.circle_loss1(data_output['img_feats_local'], data_output['pc_feats_local'], local_positive_mask, local_negative_mask)
        circle_loss2 = self.circle_loss2(data_output['pc_feats_local'], data_output['img_feats_local'], local_positive_mask.permute(0, 2, 1), local_negative_mask.permute(0, 2, 1))
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class GL_v2_triplet_double_v2_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_v2_triplet_double_v2_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss_v2(cfgs.local_loss_cfgs_1)
        self.circle_loss2 = CircleLoss_v2(cfgs.local_loss_cfgs_1)
        self.circle_loss3 = CircleLoss_v2(cfgs.local_loss_cfgs_2)
        self.circle_loss4 = CircleLoss_v2(cfgs.local_loss_cfgs_2)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.circle_loss3_meter = AverageMeter()
        self.circle_loss4_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight_1 = cfgs.local_loss_weight_1
        self.local_loss_weight_2 = cfgs.local_loss_weight_2

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)

        local_positive_mask = data_output['local_positive_mask']
        local_negative_mask = data_output['local_negative_mask']
        circle_loss1 = self.circle_loss1(data_output['img_feats_local'], data_output['pc_feats_local'], local_positive_mask, local_negative_mask)
        circle_loss2 = self.circle_loss2(data_output['pc_feats_local'], data_output['img_feats_local'], local_positive_mask.permute(0, 2, 1), local_negative_mask.permute(0, 2, 1))

        onlypc_local_positive_mask = data_output['onlypc_local_positive_mask']
        onlypc_local_negative_mask = data_output['onlypc_local_negative_mask']
        circle_loss3 = self.circle_loss3(data_output['onlypc_embeddings1'], 
                                         data_output['onlypc_embeddings2'], 
                                         onlypc_local_positive_mask, 
                                         onlypc_local_negative_mask)
        circle_loss4 = self.circle_loss4(data_output['onlypc_embeddings2'], 
                                         data_output['onlypc_embeddings1'], 
                                         onlypc_local_positive_mask.permute(0, 2, 1), 
                                         onlypc_local_negative_mask.permute(0, 2, 1))


        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight_1 * (0.5 * circle_loss1 + 0.5 * circle_loss2) + self.local_loss_weight_2 * (0.5 * circle_loss3 + 0.5 * circle_loss4)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        self.circle_loss3_meter.update(circle_loss3.detach().cpu().numpy())
        self.circle_loss4_meter.update(circle_loss4.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss_1: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss3: {circle_loss3.detach().cpu().numpy():.6f}  '
                    f'circle_loss4: {circle_loss4.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss_2: {0.5 * circle_loss3.detach().cpu().numpy() + 0.5 * circle_loss4.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg,
                                "circle_loss3": self.circle_loss3_meter.avg,
                                "circle_loss4": self.circle_loss4_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
            self.circle_loss3_meter.reset()
            self.circle_loss4_meter.reset()
        return loss

class GL_v3_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_v3_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss_v2(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss_v2(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
        local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
        circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
        circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.permute(0, 2, 1), local_negative_mask.permute(0, 2, 1))
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class GLD_v2_triplet_circle_silog_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GLD_v2_triplet_circle_silog_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.silog_Loss = SiLogLoss(cfgs.depth_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.silog_loss_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.depth_loss_weight = cfgs.depth_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'embeddings3' in data_output.keys():
            triplet_loss1_c = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2_c = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
            triplet_loss1_f = self.triplet_loss1(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
            triplet_loss2_f = self.triplet_loss2(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
            triplet_loss1 = 0.5 * triplet_loss1_c + 0.5 * triplet_loss1_f
            triplet_loss2 = 0.5 * triplet_loss2_c + 0.5 * triplet_loss2_f
        else:
            triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        
        silog_loss = self.silog_Loss(data_output['rgb_depth_preds'], data_input['rgb_depth_labels'])

        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2) + self.depth_loss_weight * silog_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        self.silog_loss_meter.update(silog_loss.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    f'silog_loss: {silog_loss.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg,
                                "silog_loss": self.silog_loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
            self.silog_loss_meter.reset()
        return loss

class GL_double_v2_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_double_v2_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss1_weight = cfgs.global_loss1_weight
        self.global_loss2_weight = cfgs.global_loss2_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        loss = (self.global_loss1_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + self.global_loss2_weight * (0.5 * triplet_loss3 + 0.5 * triplet_loss4) 
                + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class GL_five_v2_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_five_v2_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss5 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss6 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss7 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss8 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss9 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss10 = TripletLoss_v2(cfgs.global_loss_cfgs)

        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()
        self.triplet_loss5_meter = AverageMeter()
        self.triplet_loss6_meter = AverageMeter()
        self.triplet_loss7_meter = AverageMeter()
        self.triplet_loss8_meter = AverageMeter()
        self.triplet_loss9_meter = AverageMeter()
        self.triplet_loss10_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss1_weight = cfgs.global_loss1_weight
        self.global_loss2_weight = cfgs.global_loss2_weight
        self.global_loss3_weight = cfgs.global_loss3_weight
        self.global_loss4_weight = cfgs.global_loss4_weight
        self.global_loss5_weight = cfgs.global_loss5_weight
        self.local_loss_weight = cfgs.local_loss_weight
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        triplet_loss5 = self.triplet_loss5(data_output['embeddings5'], data_output['embeddings6'], data_input['overlap_ratio'])
        triplet_loss6 = self.triplet_loss6(data_output['embeddings6'], data_output['embeddings5'], data_input['overlap_ratio'].T)
        triplet_loss7 = self.triplet_loss7(data_output['embeddings7'], data_output['embeddings8'], data_input['overlap_ratio'])
        triplet_loss8 = self.triplet_loss8(data_output['embeddings8'], data_output['embeddings7'], data_input['overlap_ratio'].T)
        triplet_loss9 = self.triplet_loss9(data_output['embeddings9'], data_output['embeddings10'], data_input['overlap_ratio'])
        triplet_loss10 = self.triplet_loss10(data_output['embeddings10'], data_output['embeddings9'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        loss = (self.global_loss1_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + self.global_loss2_weight * (0.5 * triplet_loss3 + 0.5 * triplet_loss4)
                + self.global_loss3_weight * (0.5 * triplet_loss5 + 0.5 * triplet_loss6)
                + self.global_loss4_weight * (0.5 * triplet_loss7 + 0.5 * triplet_loss8)
                + self.global_loss5_weight * (0.5 * triplet_loss9 + 0.5 * triplet_loss10) 
                + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        self.triplet_loss5_meter.update(triplet_loss5.detach().cpu().numpy())
        self.triplet_loss6_meter.update(triplet_loss6.detach().cpu().numpy())
        self.triplet_loss7_meter.update(triplet_loss7.detach().cpu().numpy())
        self.triplet_loss8_meter.update(triplet_loss8.detach().cpu().numpy())
        self.triplet_loss9_meter.update(triplet_loss9.detach().cpu().numpy())
        self.triplet_loss10_meter.update(triplet_loss10.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    f'triplet_loss5: {triplet_loss5.detach().cpu().numpy():.6f}  '
                    f'triplet_loss6: {triplet_loss6.detach().cpu().numpy():.6f}  '
                    f'triplet_loss7: {triplet_loss7.detach().cpu().numpy():.6f}  '
                    f'triplet_loss8: {triplet_loss8.detach().cpu().numpy():.6f}  '
                    f'triplet_loss9: {triplet_loss9.detach().cpu().numpy():.6f}  '
                    f'triplet_loss10: {triplet_loss10.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,
                                "triplet_loss5": self.triplet_loss5_meter.avg,
                                "triplet_loss6": self.triplet_loss6_meter.avg,
                                "triplet_loss7": self.triplet_loss7_meter.avg,
                                "triplet_loss8": self.triplet_loss8_meter.avg,
                                "triplet_loss9": self.triplet_loss9_meter.avg,
                                "triplet_loss10": self.triplet_loss10_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
            self.triplet_loss5_meter.reset()
            self.triplet_loss6_meter.reset()
            self.triplet_loss7_meter.reset()
            self.triplet_loss8_meter.reset()
            self.triplet_loss9_meter.reset()
            self.triplet_loss10_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class v2_five_triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(v2_five_triplet_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs)
        self.triplet_loss5 = TripletLoss_v2(cfgs)
        self.triplet_loss6 = TripletLoss_v2(cfgs)
        self.triplet_loss7 = TripletLoss_v2(cfgs)
        self.triplet_loss8 = TripletLoss_v2(cfgs)
        self.triplet_loss9 = TripletLoss_v2(cfgs)
        self.triplet_loss10 = TripletLoss_v2(cfgs)

        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()
        self.triplet_loss5_meter = AverageMeter()
        self.triplet_loss6_meter = AverageMeter()
        self.triplet_loss7_meter = AverageMeter()
        self.triplet_loss8_meter = AverageMeter()
        self.triplet_loss9_meter = AverageMeter()
        self.triplet_loss10_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        triplet_loss5 = self.triplet_loss5(data_output['embeddings5'], data_output['embeddings6'], data_input['overlap_ratio'])
        triplet_loss6 = self.triplet_loss6(data_output['embeddings6'], data_output['embeddings5'], data_input['overlap_ratio'].T)
        triplet_loss7 = self.triplet_loss7(data_output['embeddings7'], data_output['embeddings8'], data_input['overlap_ratio'])
        triplet_loss8 = self.triplet_loss8(data_output['embeddings8'], data_output['embeddings7'], data_input['overlap_ratio'].T)
        triplet_loss9 = self.triplet_loss9(data_output['embeddings9'], data_output['embeddings10'], data_input['overlap_ratio'])
        triplet_loss10 = self.triplet_loss10(data_output['embeddings10'], data_output['embeddings9'], data_input['overlap_ratio'].T)
        loss = (0.2 * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + 0.2 * (0.5 * triplet_loss3 + 0.5 * triplet_loss4)
                + 0.2 * (0.5 * triplet_loss5 + 0.5 * triplet_loss6)
                + 0.2 * (0.5 * triplet_loss7 + 0.5 * triplet_loss8)
                + 0.2 * (0.5 * triplet_loss9 + 0.5 * triplet_loss10))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        self.triplet_loss5_meter.update(triplet_loss5.detach().cpu().numpy())
        self.triplet_loss6_meter.update(triplet_loss6.detach().cpu().numpy())
        self.triplet_loss7_meter.update(triplet_loss7.detach().cpu().numpy())
        self.triplet_loss8_meter.update(triplet_loss8.detach().cpu().numpy())
        self.triplet_loss9_meter.update(triplet_loss9.detach().cpu().numpy())
        self.triplet_loss10_meter.update(triplet_loss10.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    f'triplet_loss5: {triplet_loss5.detach().cpu().numpy():.6f}  '
                    f'triplet_loss6: {triplet_loss6.detach().cpu().numpy():.6f}  '
                    f'triplet_loss7: {triplet_loss7.detach().cpu().numpy():.6f}  '
                    f'triplet_loss8: {triplet_loss8.detach().cpu().numpy():.6f}  '
                    f'triplet_loss9: {triplet_loss9.detach().cpu().numpy():.6f}  '
                    f'triplet_loss10: {triplet_loss10.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,
                                "triplet_loss5": self.triplet_loss5_meter.avg,
                                "triplet_loss6": self.triplet_loss6_meter.avg,
                                "triplet_loss7": self.triplet_loss7_meter.avg,
                                "triplet_loss8": self.triplet_loss8_meter.avg,
                                "triplet_loss9": self.triplet_loss9_meter.avg,
                                "triplet_loss10": self.triplet_loss10_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
            self.triplet_loss5_meter.reset()
            self.triplet_loss6_meter.reset()
            self.triplet_loss7_meter.reset()
            self.triplet_loss8_meter.reset()
            self.triplet_loss9_meter.reset()
            self.triplet_loss10_meter.reset()
        return loss

class v2_four_triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(v2_four_triplet_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs)
        self.triplet_loss5 = TripletLoss_v2(cfgs)
        self.triplet_loss6 = TripletLoss_v2(cfgs)
        self.triplet_loss7 = TripletLoss_v2(cfgs)
        self.triplet_loss8 = TripletLoss_v2(cfgs)

        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()
        self.triplet_loss5_meter = AverageMeter()
        self.triplet_loss6_meter = AverageMeter()
        self.triplet_loss7_meter = AverageMeter()
        self.triplet_loss8_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        triplet_loss5 = self.triplet_loss5(data_output['embeddings5'], data_output['embeddings6'], data_input['overlap_ratio'])
        triplet_loss6 = self.triplet_loss6(data_output['embeddings6'], data_output['embeddings5'], data_input['overlap_ratio'].T)
        triplet_loss7 = self.triplet_loss7(data_output['embeddings7'], data_output['embeddings8'], data_input['overlap_ratio'])
        triplet_loss8 = self.triplet_loss8(data_output['embeddings8'], data_output['embeddings7'], data_input['overlap_ratio'].T)
        loss = (0.25 * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + 0.25 * (0.5 * triplet_loss3 + 0.5 * triplet_loss4)
                + 0.25 * (0.5 * triplet_loss5 + 0.5 * triplet_loss6)
                + 0.25 * (0.5 * triplet_loss7 + 0.5 * triplet_loss8))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        self.triplet_loss5_meter.update(triplet_loss5.detach().cpu().numpy())
        self.triplet_loss6_meter.update(triplet_loss6.detach().cpu().numpy())
        self.triplet_loss7_meter.update(triplet_loss7.detach().cpu().numpy())
        self.triplet_loss8_meter.update(triplet_loss8.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    f'triplet_loss5: {triplet_loss5.detach().cpu().numpy():.6f}  '
                    f'triplet_loss6: {triplet_loss6.detach().cpu().numpy():.6f}  '
                    f'triplet_loss7: {triplet_loss7.detach().cpu().numpy():.6f}  '
                    f'triplet_loss8: {triplet_loss8.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,
                                "triplet_loss5": self.triplet_loss5_meter.avg,
                                "triplet_loss6": self.triplet_loss6_meter.avg,
                                "triplet_loss7": self.triplet_loss7_meter.avg,
                                "triplet_loss8": self.triplet_loss8_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
            self.triplet_loss5_meter.reset()
            self.triplet_loss6_meter.reset()
            self.triplet_loss7_meter.reset()
            self.triplet_loss8_meter.reset()
        return loss

class v2_two_triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(v2_two_triplet_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs)

        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        loss = (0.5 * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + 0.5 * (0.5 * triplet_loss3 + 0.5 * triplet_loss4))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
        return loss

class two_triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(two_triplet_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss(cfgs)
        self.triplet_loss2 = TripletLoss(cfgs)
        self.triplet_loss3 = TripletLoss(cfgs)
        self.triplet_loss4 = TripletLoss(cfgs)

        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1, _ = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
        triplet_loss2, _ = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
        triplet_loss3, _ = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['positives_mask'], data_input['negatives_mask'])
        triplet_loss4, _ = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
        loss = (0.5 * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + 0.5 * (0.5 * triplet_loss3 + 0.5 * triplet_loss4))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
        return loss

class v2_three_triplet_lossor(nn.Module):

    def __init__(self, cfgs):
        super(v2_three_triplet_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs)
        self.triplet_loss3 = TripletLoss_v2(cfgs)
        self.triplet_loss4 = TripletLoss_v2(cfgs)
        self.triplet_loss5 = TripletLoss_v2(cfgs)
        self.triplet_loss6 = TripletLoss_v2(cfgs)

        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.triplet_loss3_meter = AverageMeter()
        self.triplet_loss4_meter = AverageMeter()
        self.triplet_loss5_meter = AverageMeter()
        self.triplet_loss6_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        triplet_loss3 = self.triplet_loss3(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
        triplet_loss4 = self.triplet_loss4(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
        triplet_loss5 = self.triplet_loss5(data_output['embeddings5'], data_output['embeddings6'], data_input['overlap_ratio'])
        triplet_loss6 = self.triplet_loss6(data_output['embeddings6'], data_output['embeddings5'], data_input['overlap_ratio'].T)
        loss = (0.5 * (0.5 * triplet_loss1 + 0.5 * triplet_loss2)
                + 0.5 * (0.5 * triplet_loss3 + 0.5 * triplet_loss4)
                + 0.5 * (0.5 * triplet_loss5 + 0.5 * triplet_loss6))

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.triplet_loss3_meter.update(triplet_loss3.detach().cpu().numpy())
        self.triplet_loss4_meter.update(triplet_loss4.detach().cpu().numpy())
        self.triplet_loss5_meter.update(triplet_loss5.detach().cpu().numpy())
        self.triplet_loss6_meter.update(triplet_loss6.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss3: {triplet_loss3.detach().cpu().numpy():.6f}  '
                    f'triplet_loss4: {triplet_loss4.detach().cpu().numpy():.6f}  '
                    f'triplet_loss5: {triplet_loss5.detach().cpu().numpy():.6f}  '
                    f'triplet_loss6: {triplet_loss6.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "triplet_loss3": self.triplet_loss3_meter.avg,
                                "triplet_loss4": self.triplet_loss4_meter.avg,
                                "triplet_loss5": self.triplet_loss5_meter.avg,
                                "triplet_loss6": self.triplet_loss6_meter.avg,}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.triplet_loss3_meter.reset()
            self.triplet_loss4_meter.reset()
            self.triplet_loss5_meter.reset()
            self.triplet_loss6_meter.reset()
        return loss

class GL_seperate_v2_triplet_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_seperate_v2_triplet_circle_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.circle_loss1 = CircleLoss(cfgs.local_loss_cfgs)
        self.circle_loss2 = CircleLoss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.global_loss_weight = 0.5
        self.local_loss_weight = 0.5
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        show_loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * circle_loss1 + 0.5 * circle_loss2)

        self.loss_meter.update(show_loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {show_loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * circle_loss1.detach().cpu().numpy() + 0.5 * circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        
        loss = torch.stack((0.5 * triplet_loss1 + 0.5 * triplet_loss2, 0.5 * circle_loss1 + 0.5 * circle_loss2), dim=0) # produce loss (2, )
        return loss

class L_circle_lossor(nn.Module):

    def __init__(self, cfgs):
        super(L_circle_lossor, self).__init__()
        self.circle_loss1 = CircleLoss(cfgs)
        self.circle_loss2 = CircleLoss(cfgs)
        self.loss_meter = AverageMeter()
        self.circle_loss1_meter = AverageMeter()
        self.circle_loss2_meter = AverageMeter()
        self.local_positive_mask_margin = cfgs.local_positive_mask_margin
        self.local_negative_mask_margin = cfgs.local_negative_mask_margin

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'local_overlap_ratio1' in data_output.keys():
            local_positive_mask = torch.gt(data_output['local_overlap_ratio1'], self.local_positive_mask_margin)
            local_negative_mask = torch.lt(data_output['local_overlap_ratio1'], self.local_negative_mask_margin)
            circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], local_positive_mask, local_negative_mask)
            circle_loss2 = self.circle_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], local_positive_mask.T, local_negative_mask.T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                circle_loss1 = 0.0
                circle_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    local_positive_mask = torch.gt(data_output['local_overlap_ratio'][i], self.local_positive_mask_margin[i])
                    local_negative_mask = torch.lt(data_output['local_overlap_ratio'][i], self.local_negative_mask_margin[i])
                    circle_loss1 += per_loss_scale * self.circle_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], local_positive_mask, local_negative_mask)
                    circle_loss2 += per_loss_scale * self.circle_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], local_positive_mask.T, local_negative_mask.T)
            else:
                local_positive_mask = torch.gt(data_output['local_overlap_ratio'], self.local_positive_mask_margin)
                local_negative_mask = torch.lt(data_output['local_overlap_ratio'], self.local_negative_mask_margin)
                circle_loss1 = self.circle_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], local_positive_mask, local_negative_mask)
                circle_loss2 = self.circle_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], local_positive_mask.T, local_negative_mask.T)
        loss = 0.5 * circle_loss1 + 0.5 * circle_loss2

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.circle_loss1_meter.update(circle_loss1.detach().cpu().numpy())
        self.circle_loss2_meter.update(circle_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'circle_loss1: {circle_loss1.detach().cpu().numpy():.6f}  '
                    f'circle_loss2: {circle_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"circle_loss1": self.circle_loss1_meter.avg,
                                "circle_loss2": self.circle_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.circle_loss1_meter.reset()
            self.circle_loss2_meter.reset()
        return loss

class G1M_triplet_cmpm_lossor(nn.Module):

    def __init__(self, cfgs):
        super(G1M_triplet_cmpm_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss(cfgs.triplet_loss_cfgs)
        self.triplet_loss2 = TripletLoss(cfgs.triplet_loss_cfgs)
        self.cmpm_loss = CmpmLoss(cfgs.cmpm_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.cmpm_loss_meter = AverageMeter()
        self.max_pos_pair_dist_meter = AverageMeter()
        self.max_neg_pair_dist_meter = AverageMeter()
        self.mean_pos_pair_dist_meter = AverageMeter()
        self.mean_neg_pair_dist_meter = AverageMeter()
        self.min_pos_pair_dist_meter = AverageMeter()
        self.min_neg_pair_dist_meter = AverageMeter()
        self.pos_avg_sim_meter = AverageMeter()
        self.neg_avg_sim_meter = AverageMeter() 
        self.cmpm_loss_weight = cfgs.cmpm_loss_weight
        self.triplet_loss_weight = cfgs.triplet_loss_weight

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        triplet_loss1, triplet_loss_stat1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
        triplet_loss2, triplet_loss_stat2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
        cmpm_loss, cmpm_stat = self.cmpm_loss(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'])
        loss = self.triplet_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.cmpm_loss_weight * cmpm_loss
        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.cmpm_loss_meter.update(cmpm_loss.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'triplet_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'cmpm_loss: {cmpm_loss.detach().cpu().numpy():.6f}  '
                    )
        
        triplet_loss_stat = {k: 0.5 * triplet_loss_stat1[k] + 0.5 * triplet_loss_stat2[k] for k in triplet_loss_stat1}
        self.max_pos_pair_dist_meter.update(triplet_loss_stat['max_pos_pair_dist'])
        self.max_neg_pair_dist_meter.update(triplet_loss_stat['max_neg_pair_dist'])
        self.mean_pos_pair_dist_meter.update(triplet_loss_stat['mean_pos_pair_dist'])
        self.mean_neg_pair_dist_meter.update(triplet_loss_stat['mean_neg_pair_dist'])
        self.min_pos_pair_dist_meter.update(triplet_loss_stat['min_pos_pair_dist'])
        self.min_neg_pair_dist_meter.update(triplet_loss_stat['min_neg_pair_dist'])
        self.pos_avg_sim_meter.update(cmpm_stat['pos_avg_sim'])
        self.neg_avg_sim_meter.update(cmpm_stat['neg_avg_sim'])
        logger.info(
                f'max_pos_pair_dist: {self.max_pos_pair_dist_meter.val:.6f}  '
                f'max_neg_pair_dist: {self.max_neg_pair_dist_meter.val:.6f}  '
                f'mean_pos_pair_dist: {self.mean_pos_pair_dist_meter.val:.6f}  '
                f'mean_neg_pair_dist: {self.mean_neg_pair_dist_meter.val:.6f}  '
                f'min_pos_pair_dist: {self.min_pos_pair_dist_meter.val:.6f}  '
                f'min_neg_pair_dist: {self.min_neg_pair_dist_meter.val:.6f}  '
                f'pos_avg_sim: {self.pos_avg_sim_meter.val:.6f}  '
                f'neg_avg_sim: {self.neg_avg_sim_meter.val:.6f}  '
                )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "cmpm_loss": self.cmpm_loss_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.cmpm_loss_meter.reset()
            if is_main_process():
                stat_wandb = {
                    "max_pos_pair_dist": self.max_pos_pair_dist_meter.avg,
                    "max_neg_pair_dist": self.max_neg_pair_dist_meter.avg,
                    "mean_pos_pair_dist": self.mean_pos_pair_dist_meter.avg,
                    "mean_neg_pair_dist": self.mean_neg_pair_dist_meter.avg,
                    "min_pos_pair_dist": self.min_pos_pair_dist_meter.avg,
                    "min_neg_pair_dist": self.min_neg_pair_dist_meter.avg,
                    "pos_avg_sim": self.pos_avg_sim_meter.avg,
                    "neg_avg_sim": self.neg_avg_sim_meter.avg
                }
                wandb.log(data={"pair_dist_stat": stat_wandb}, step=epoch)
            self.max_pos_pair_dist_meter.reset()
            self.max_neg_pair_dist_meter.reset()
            self.mean_pos_pair_dist_meter.reset()
            self.mean_neg_pair_dist_meter.reset()
            self.min_pos_pair_dist_meter.reset()
            self.min_neg_pair_dist_meter.reset()
            self.pos_avg_sim_meter.reset()
            self.neg_avg_sim_meter.reset()
        return loss

class GL_triplet_cfi2p_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_triplet_cfi2p_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss(cfgs.global_loss_cfgs)
        self.cfi2p_loss1 = CFI2P_loss(cfgs.local_loss_cfgs)
        self.cfi2p_loss2 = CFI2P_loss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.cfi2p_loss1_meter = AverageMeter()
        self.cfi2p_loss2_meter = AverageMeter()
        self.max_pos_pair_dist_meter = AverageMeter()
        self.max_neg_pair_dist_meter = AverageMeter()
        self.mean_pos_pair_dist_meter = AverageMeter()
        self.mean_neg_pair_dist_meter = AverageMeter()
        self.min_pos_pair_dist_meter = AverageMeter()
        self.min_neg_pair_dist_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'embeddings3' in data_output.keys():
            triplet_loss1_c, triplet_loss_stat1_c = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2_c, triplet_loss_stat2_c = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss1_f, triplet_loss_stat1_f = self.triplet_loss1(data_output['embeddings3'], data_output['embeddings4'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2_f, triplet_loss_stat2_f = self.triplet_loss2(data_output['embeddings4'], data_output['embeddings3'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss1 = 0.5 * triplet_loss1_c + 0.5 * triplet_loss1_f
            triplet_loss2 = 0.5 * triplet_loss2_c + 0.5 * triplet_loss2_f
            triplet_loss_stat1 = {k: 0.5 * triplet_loss_stat1_c[k] + 0.5 * triplet_loss_stat1_f[k] for k in triplet_loss_stat1_c}
            triplet_loss_stat2 = {k: 0.5 * triplet_loss_stat2_c[k] + 0.5 * triplet_loss_stat2_f[k] for k in triplet_loss_stat2_c}
        else:
            triplet_loss1, triplet_loss_stat1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['positives_mask'], data_input['negatives_mask'])
            triplet_loss2, triplet_loss_stat2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['positives_mask'], data_input['negatives_mask'])
        if 'local_overlap_ratio1' in data_output.keys():
            cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], data_input['local_overlap_ratio1'])
            cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], data_input['local_overlap_ratio1'].T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                cfi2p_loss1 = 0.0
                cfi2p_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    cfi2p_loss1 += per_loss_scale * self.cfi2p_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], data_output['local_overlap_ratio'][i])
                    cfi2p_loss2 += per_loss_scale * self.cfi2p_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], data_output['local_overlap_ratio'][i].T)
            else:
                cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], data_output['local_overlap_ratio'])
                cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], data_output['local_overlap_ratio'].T)
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * cfi2p_loss1 + 0.5 * cfi2p_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.cfi2p_loss1_meter.update(cfi2p_loss1.detach().cpu().numpy())
        self.cfi2p_loss2_meter.update(cfi2p_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss1: {cfi2p_loss1.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss2: {cfi2p_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * cfi2p_loss1.detach().cpu().numpy() + 0.5 * cfi2p_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        triplet_loss_stat = {k: 0.5 * triplet_loss_stat1[k] + 0.5 * triplet_loss_stat2[k] for k in triplet_loss_stat1}
        self.max_pos_pair_dist_meter.update(triplet_loss_stat['max_pos_pair_dist'])
        self.max_neg_pair_dist_meter.update(triplet_loss_stat['max_neg_pair_dist'])
        self.mean_pos_pair_dist_meter.update(triplet_loss_stat['mean_pos_pair_dist'])
        self.mean_neg_pair_dist_meter.update(triplet_loss_stat['mean_neg_pair_dist'])
        self.min_pos_pair_dist_meter.update(triplet_loss_stat['min_pos_pair_dist'])
        self.min_neg_pair_dist_meter.update(triplet_loss_stat['min_neg_pair_dist'])
        logger.info(
                f'max_pos_pair_dist: {self.max_pos_pair_dist_meter.val:.6f}  '
                f'max_neg_pair_dist: {self.max_neg_pair_dist_meter.val:.6f}  '
                f'mean_pos_pair_dist: {self.mean_pos_pair_dist_meter.val:.6f}  '
                f'mean_neg_pair_dist: {self.mean_neg_pair_dist_meter.val:.6f}  '
                f'min_pos_pair_dist: {self.min_pos_pair_dist_meter.val:.6f}  '
                f'min_neg_pair_dist: {self.min_neg_pair_dist_meter.val:.6f}  '
                )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "cfi2p_loss1": self.cfi2p_loss1_meter.avg,
                                "cfi2p_loss2": self.cfi2p_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.cfi2p_loss1_meter.reset()
            self.cfi2p_loss2_meter.reset()
            if is_main_process():
                stat_wandb = {
                    "max_pos_pair_dist": self.max_pos_pair_dist_meter.avg,
                    "max_neg_pair_dist": self.max_neg_pair_dist_meter.avg,
                    "mean_pos_pair_dist": self.mean_pos_pair_dist_meter.avg,
                    "mean_neg_pair_dist": self.mean_neg_pair_dist_meter.avg,
                    "min_pos_pair_dist": self.min_pos_pair_dist_meter.avg,
                    "min_neg_pair_dist": self.min_neg_pair_dist_meter.avg
                }
                wandb.log(data={"pair_dist_stat": stat_wandb}, step=epoch)
            self.max_pos_pair_dist_meter.reset()
            self.max_neg_pair_dist_meter.reset()
            self.mean_pos_pair_dist_meter.reset()
            self.mean_neg_pair_dist_meter.reset()
            self.min_pos_pair_dist_meter.reset()
            self.min_neg_pair_dist_meter.reset()
        return loss

class GL_v2_triplet_cfi2p_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_v2_triplet_cfi2p_lossor, self).__init__()
        self.triplet_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.triplet_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.cfi2p_loss1 = CFI2P_loss(cfgs.local_loss_cfgs)
        self.cfi2p_loss2 = CFI2P_loss(cfgs.local_loss_cfgs)
        self.loss_meter = AverageMeter()
        self.triplet_loss1_meter = AverageMeter()
        self.triplet_loss2_meter = AverageMeter()
        self.cfi2p_loss1_meter = AverageMeter()
        self.cfi2p_loss2_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.local_loss_weight = cfgs.local_loss_weight

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'embeddings3' in data_output.keys():
            triplet_loss1_c = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2_c = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
            triplet_loss1_f = self.triplet_loss1(data_output['embeddings3'], data_output['embeddings4'], data_input['overlap_ratio'])
            triplet_loss2_f = self.triplet_loss2(data_output['embeddings4'], data_output['embeddings3'], data_input['overlap_ratio'].T)
            triplet_loss1 = 0.5 * triplet_loss1_c + 0.5 * triplet_loss1_f
            triplet_loss2 = 0.5 * triplet_loss2_c + 0.5 * triplet_loss2_f
        else:
            triplet_loss1 = self.triplet_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
            triplet_loss2 = self.triplet_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        if 'local_overlap_ratio1' in data_output.keys():
            cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], data_output['local_overlap_ratio1'])
            cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], data_output['local_overlap_ratio1'].T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                cfi2p_loss1 = 0.0
                cfi2p_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    cfi2p_loss1 += per_loss_scale * self.cfi2p_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], data_output['local_overlap_ratio'][i])
                    cfi2p_loss2 += per_loss_scale * self.cfi2p_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], data_output['local_overlap_ratio'][i].T)
            else:
                cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], data_output['local_overlap_ratio'])
                cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], data_output['local_overlap_ratio'].T)
        loss = self.global_loss_weight * (0.5 * triplet_loss1 + 0.5 * triplet_loss2) + self.local_loss_weight * (0.5 * cfi2p_loss1 + 0.5 * cfi2p_loss2)

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.triplet_loss1_meter.update(triplet_loss1.detach().cpu().numpy())
        self.triplet_loss2_meter.update(triplet_loss2.detach().cpu().numpy())
        self.cfi2p_loss1_meter.update(cfi2p_loss1.detach().cpu().numpy())
        self.cfi2p_loss2_meter.update(cfi2p_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'triplet_loss1: {triplet_loss1.detach().cpu().numpy():.6f}  '
                    f'triplet_loss2: {triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_global_loss: {0.5 * triplet_loss1.detach().cpu().numpy() + 0.5 * triplet_loss2.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss1: {cfi2p_loss1.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss2: {cfi2p_loss2.detach().cpu().numpy():.6f}  '
                    f'weighted_local_loss: {0.5 * cfi2p_loss1.detach().cpu().numpy() + 0.5 * cfi2p_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"triplet_loss1": self.triplet_loss1_meter.avg,
                                "triplet_loss2": self.triplet_loss2_meter.avg,
                                "cfi2p_loss1": self.cfi2p_loss1_meter.avg,
                                "cfi2p_loss2": self.cfi2p_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.triplet_loss1_meter.reset()
            self.triplet_loss2_meter.reset()
            self.cfi2p_loss1_meter.reset()
            self.cfi2p_loss2_meter.reset()
        return loss

class L_cfi2p_lossor(nn.Module):

    def __init__(self, cfgs):
        super(L_cfi2p_lossor, self).__init__()
        self.cfi2p_loss1 = CFI2P_loss(cfgs)
        self.cfi2p_loss2 = CFI2P_loss(cfgs)
        self.loss_meter = AverageMeter()
        self.cfi2p_loss1_meter = AverageMeter()
        self.cfi2p_loss2_meter = AverageMeter()

    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        if 'local_overlap_ratio1' in data_output.keys():
            cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings1'], data_output['img_local_embeddings1'], data_output['local_overlap_ratio1'])
            cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings1'], data_output['pc_local_embeddings1'], data_output['local_overlap_ratio1'].T)
        elif 'local_overlap_ratio' in data_output.keys():
            if isinstance(data_output['local_overlap_ratio'], list):
                cfi2p_loss1 = 0.0
                cfi2p_loss2 = 0.0
                overlap_ratio_matrix_len = len(data_output['local_overlap_ratio'])
                per_loss_scale = 1.0 / overlap_ratio_matrix_len 
                for i in range(len(data_output['local_overlap_ratio'])):
                    cfi2p_loss1 += per_loss_scale * self.cfi2p_loss1(data_output['pc_local_embeddings'][i], data_output['img_local_embeddings'][i], data_output['local_overlap_ratio'][i])
                    cfi2p_loss2 += per_loss_scale * self.cfi2p_loss2(data_output['img_local_embeddings'][i], data_output['pc_local_embeddings'][i], data_output['local_overlap_ratio'][i].T)
            else:
                cfi2p_loss1 = self.cfi2p_loss1(data_output['pc_local_embeddings'], data_output['img_local_embeddings'], data_output['local_overlap_ratio'])
                cfi2p_loss2 = self.cfi2p_loss2(data_output['img_local_embeddings'], data_output['pc_local_embeddings'], data_output['local_overlap_ratio'].T)
        loss = 0.5 * cfi2p_loss1 + 0.5 * cfi2p_loss2

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.cfi2p_loss1_meter.update(cfi2p_loss1.detach().cpu().numpy())
        self.cfi2p_loss2_meter.update(cfi2p_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss1: {cfi2p_loss1.detach().cpu().numpy():.6f}  '
                    f'cfi2p_loss2: {cfi2p_loss2.detach().cpu().numpy():.6f}  '
                    )
        
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"cfi2p_loss1": self.cfi2p_loss1_meter.avg,
                                "cfi2p_loss2": self.cfi2p_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.cfi2p_loss1_meter.reset()
            self.cfi2p_loss2_meter.reset()
        return loss

class GL_semantic_cluster_lossor(nn.Module):

    def __init__(self, cfgs):
        super(GL_semantic_cluster_lossor, self).__init__()
        self.global_loss1 = TripletLoss_v2(cfgs.global_loss_cfgs)
        self.global_loss2 = TripletLoss_v2(cfgs.global_loss_cfgs)
        if cfgs.cluster_loss_cfgs.loss_type == 'infonce':
            self.cluster_loss1 = InfoNCELoss(cfgs.cluster_loss_cfgs)
            self.cluster_loss2 = InfoNCELoss(cfgs.cluster_loss_cfgs)
            self.cluster_loss_type = 'infonce'
        elif cfgs.cluster_loss_cfgs.loss_type == 'loftr_focal':
            self.cluster_loss1 = LoftrFocalLoss(cfgs.cluster_loss_cfgs)
            self.cluster_loss2 = LoftrFocalLoss(cfgs.cluster_loss_cfgs)
            self.cluster_loss_type = 'loftr_focal'
        else:
            raise ValueError(f"Unknown loss: {cfgs.cluster_loss_cfgs.loss_type}")
        if cfgs.semantic_in_batch_loss_cfgs.loss_type == 'infonce':
            self.semantic_in_batch_loss1 = InfoNCELoss(cfgs.semantic_in_batch_loss_cfgs)
            self.semantic_in_batch_loss2 = InfoNCELoss(cfgs.semantic_in_batch_loss_cfgs)
            self.semantic_in_batch_loss_type = 'infonce'
        elif cfgs.semantic_in_batch_loss_cfgs.loss_type == 'loftr_focal':
            self.semantic_in_batch_loss1 = LoftrFocalLoss(cfgs.semantic_in_batch_loss_cfgs)
            self.semantic_in_batch_loss2 = LoftrFocalLoss(cfgs.semantic_in_batch_loss_cfgs)
            self.semantic_in_batch_loss_type = 'loftr_focal'
        else:
            raise ValueError(f"Unknown loss: {cfgs.semantic_in_batch_loss_cfgs.loss_type}")
        if cfgs.semantic_all_loss_cfgs.loss_type == 'infonce':
            self.semantic_all_loss1 = InfoNCELoss(cfgs.semantic_all_loss_cfgs)
            self.semantic_all_loss2 = InfoNCELoss(cfgs.semantic_all_loss_cfgs)
            self.semantic_all_loss_type = 'infonce'
        elif cfgs.semantic_all_loss_cfgs.loss_type == 'loftr_focal':
            self.semantic_all_loss1 = LoftrFocalLoss(cfgs.semantic_all_loss_cfgs)
            self.semantic_all_loss2 = LoftrFocalLoss(cfgs.semantic_all_loss_cfgs)
            self.semantic_all_loss_type = 'loftr_focal'
        else:
            raise ValueError(f"Unknown loss: {cfgs.semantic_all_loss_cfgs.loss_type}")
        
        self.loss_meter = AverageMeter()
        self.global_loss1_meter = AverageMeter()
        self.global_loss2_meter = AverageMeter()
        self.cluster_loss1_meter = AverageMeter()
        self.cluster_loss2_meter = AverageMeter()
        self.semantic_in_batch_loss1_meter = AverageMeter()
        self.semantic_in_batch_loss2_meter = AverageMeter()
        self.semantic_all_loss1_meter = AverageMeter()
        self.semantic_all_loss2_meter = AverageMeter()
        self.global_loss_weight = cfgs.global_loss_weight
        self.cluster_loss_weight = cfgs.cluster_loss_weight
        self.semantic_in_batch_loss_weight = cfgs.semantic_in_batch_loss_weight
        self.semantic_all_loss_weight = cfgs.semantic_all_loss_weight
    
    def forward(self, data_output, data_input, logger, epoch, bn, iter_num, all_epochs):
        global_loss1 = self.global_loss1(data_output['embeddings1'], data_output['embeddings2'], data_input['overlap_ratio'])
        global_loss2 = self.global_loss2(data_output['embeddings2'], data_output['embeddings1'], data_input['overlap_ratio'].T)
        global_loss = 0.5 * global_loss1 + 0.5 * global_loss2

        if self.cluster_loss_type == 'infonce':
            positives_mask = torch.eye(data_output['img_cluster_embeddings'].shape[0], device=data_output['img_cluster_embeddings'].device, dtype=torch.bool)
            negatives_mask = ~positives_mask
            cluster_loss1 = self.cluster_loss1(data_output['img_cluster_embeddings'], data_output['pc_cluster_embeddings'], positives_mask, negatives_mask)
            cluster_loss2 = self.cluster_loss2(data_output['pc_cluster_embeddings'], data_output['img_cluster_embeddings'], positives_mask.T, negatives_mask.T)
        elif self.cluster_loss_type == 'loftr_focal':
            cluster_loss1 = self.cluster_loss1(data_output['img_cluster_embeddings'], data_output['pc_cluster_embeddings'])
            cluster_loss2 = self.cluster_loss2(data_output['pc_cluster_embeddings'], data_output['img_cluster_embeddings'])
        cluster_loss = 0.5 * cluster_loss1 + 0.5 * cluster_loss2

        if self.semantic_in_batch_loss_type == 'infonce':
            positives_mask = torch.eye(data_output['img_in_batch_semantic_embeddings'].shape[0], device=data_output['img_in_batch_semantic_embeddings'].device, dtype=torch.bool)
            negatives_mask = ~positives_mask
            semantic_in_batch_loss1 = self.semantic_in_batch_loss1(data_output['img_in_batch_semantic_embeddings'], data_output['pc_in_batch_semantic_embeddings'], positives_mask, negatives_mask)
            semantic_in_batch_loss2 = self.semantic_in_batch_loss2(data_output['pc_in_batch_semantic_embeddings'], data_output['img_in_batch_semantic_embeddings'], positives_mask.T, negatives_mask.T)
        elif self.semantic_in_batch_loss_type == 'loftr_focal':
            semantic_in_batch_loss1 = self.semantic_in_batch_loss1(data_output['img_in_batch_semantic_embeddings'], data_output['pc_in_batch_semantic_embeddings'])
            semantic_in_batch_loss2 = self.semantic_in_batch_loss2(data_output['pc_in_batch_semantic_embeddings'], data_output['img_in_batch_semantic_embeddings'])
        semantic_in_batch_loss = 0.5 * semantic_in_batch_loss1 + 0.5 * semantic_in_batch_loss2

        if self.semantic_all_loss_type == 'infonce':
            positives_mask = torch.eye(data_output['img_all_semantic_embeddings'].shape[0], device=data_output['img_all_semantic_embeddings'].device, dtype=torch.bool)
            negatives_mask = ~positives_mask
            semantic_all_loss1 = self.semantic_all_loss1(data_output['img_all_semantic_embeddings'], data_output['pc_all_semantic_embeddings'], positives_mask, negatives_mask)
            semantic_all_loss2 = self.semantic_all_loss2(data_output['pc_all_semantic_embeddings'], data_output['img_all_semantic_embeddings'], positives_mask.T, negatives_mask.T)
        elif self.semantic_all_loss_type == 'loftr_focal':
            semantic_all_loss1 = self.semantic_all_loss1(data_output['img_all_semantic_embeddings'], data_output['pc_all_semantic_embeddings'])
            semantic_all_loss2 = self.semantic_all_loss2(data_output['pc_all_semantic_embeddings'], data_output['img_all_semantic_embeddings'])
        semantic_all_loss = 0.5 * semantic_all_loss1 + 0.5 * semantic_all_loss2

        loss = self.global_loss_weight * global_loss + self.cluster_loss_weight * cluster_loss + self.semantic_in_batch_loss_weight * semantic_in_batch_loss + self.semantic_all_loss_weight * semantic_all_loss

        self.loss_meter.update(loss.detach().cpu().numpy())
        self.global_loss1_meter.update(global_loss1.detach().cpu().numpy())
        self.global_loss2_meter.update(global_loss2.detach().cpu().numpy())
        self.cluster_loss1_meter.update(cluster_loss1.detach().cpu().numpy())
        self.cluster_loss2_meter.update(cluster_loss2.detach().cpu().numpy())
        self.semantic_in_batch_loss1_meter.update(semantic_in_batch_loss1.detach().cpu().numpy())
        self.semantic_in_batch_loss2_meter.update(semantic_in_batch_loss2.detach().cpu().numpy())
        self.semantic_all_loss1_meter.update(semantic_all_loss1.detach().cpu().numpy())
        self.semantic_all_loss2_meter.update(semantic_all_loss2.detach().cpu().numpy())
        logger.info(
                    f'epoch[{all_epochs}|{epoch}]  '
                    f'iter[{bn}|{iter_num}]  '
                    f'all_loss: {loss.detach().cpu().numpy():.6f}  '
                    f'global_loss: {global_loss.detach().cpu().numpy():.6f}  '
                    f'cluster_loss: {cluster_loss.detach().cpu().numpy():.6f}  '
                    f'semantic_in_batch_loss: {semantic_in_batch_loss.detach().cpu().numpy():.6f}  '
                    f'semantic_all_loss: {semantic_all_loss.detach().cpu().numpy():.6f}  '
                    )
        if (bn + 1) == iter_num:
            if is_main_process():
                wandb.log(data={"all_loss": self.loss_meter.avg}, step=epoch)
                wandb.log(data={"global_loss1": self.global_loss1_meter.avg,
                                "global_loss2": self.global_loss2_meter.avg,
                                "cluster_loss1": self.cluster_loss1_meter.avg,
                                "cluster_loss2": self.cluster_loss2_meter.avg,
                                "semantic_in_batch_loss1": self.semantic_in_batch_loss1_meter.avg,
                                "semantic_in_batch_loss2": self.semantic_in_batch_loss2_meter.avg,
                                "semantic_all_loss1": self.semantic_all_loss1_meter.avg,
                                "semantic_all_loss2": self.semantic_all_loss2_meter.avg}, step=epoch)
            self.loss_meter.reset()
            self.global_loss1_meter.reset()
            self.global_loss2_meter.reset()
            self.cluster_loss1_meter.reset()
            self.cluster_loss2_meter.reset()
            self.semantic_in_batch_loss1_meter.reset()
            self.semantic_in_batch_loss2_meter.reset()
            self.semantic_all_loss1_meter.reset()
            self.semantic_all_loss2_meter.reset()
        return loss

def make_loss(cfgs, device):
    if cfgs.loss_type.startswith('triplet'):
        lossor = triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('two_triplet'):
        lossor = two_triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('cmpm'):
        lossor = cmpm_lossor(cfgs)
    elif cfgs.loss_type.startswith('circle'):
        lossor = circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('infonce'):
        lossor = infonce_lossor(cfgs)
    elif cfgs.loss_type.startswith('general_contrastive'):
        lossor = general_contrastive_lossor(cfgs)
    elif cfgs.loss_type.startswith('two_general_contrastive'):
        lossor = two_general_contrastive_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_triplet'):
        lossor = triplet_v2_lossor(cfgs)
    elif cfgs.loss_type.startswith('v3_triplet'):
        lossor = triplet_v3_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_infonce'):
        lossor = infonce_v2_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_triplet_circle'):
        lossor = GL_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('G1M_triplet_cmpm'):
        lossor = G1M_triplet_cmpm_lossor(cfgs)
    elif cfgs.loss_type.startswith('G1M_triplet_ap'):
        lossor = G1M_triplet_ap_lossor(cfgs)
    elif cfgs.loss_type.startswith('G1M_triplet_circle'):
        lossor = G1M_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('G1M_triplet_roadmap'):
        lossor = G1M_triplet_roadmap_lossor(cfgs)
    elif cfgs.loss_type.startswith('G1M_huber_angle'):
        lossor = G1M_huber_angle_lossor(cfgs)
    elif cfgs.loss_type.startswith('G2M_triplet_huber_angle'):
        lossor = G2M_triplet_huber_angle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_triplet_cfi2p_lossor'):
        lossor = GL_triplet_cfi2p_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_v2_triplet_circle'):
        lossor = GL_v2_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_v3_triplet_circle'):
        lossor = GL_v3_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_double_v2_triplet_circle'):
        lossor = GL_double_v2_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_five_v2_triplet_circle'):
        lossor = GL_five_v2_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_seperate_v2_triplet_circle'):
        lossor = GL_seperate_v2_triplet_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_v2_triplet_cfi2p'):
        lossor = GL_v2_triplet_cfi2p_lossor(cfgs)
    elif cfgs.loss_type.startswith('L_circle'):
        lossor = L_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('L_cfi2p'):
        lossor = L_cfi2p_lossor(cfgs)
    elif cfgs.loss_type.startswith('GLD_v2_triplet_circle_silog'):
        lossor = GLD_v2_triplet_circle_silog_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_five_triplet'):
        lossor = v2_five_triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_four_triplet'):
        lossor = v2_four_triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_three_triplet'):
        lossor = v2_three_triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('v2_two_triplet'):
        lossor = v2_two_triplet_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_semantic_cluster'):
        lossor = GL_semantic_cluster_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_v2_triplet_v2_circle'):
        lossor = GL_v2_triplet_v2_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('GL_v2_triplet_double_v2_circle'):
        lossor = GL_v2_triplet_double_v2_circle_lossor(cfgs)
    elif cfgs.loss_type.startswith('v4_triplet'):
        lossor = triplet_v4_lossor(cfgs)
    elif cfgs.loss_type.startswith('MB_triplet_v2_infonce'):
        lossor = MB_triplet_v2_infonce_lossor(cfgs)
    elif cfgs.loss_type.startswith('MBKL_triplet_v2_infonce'):
        lossor = MBKL_triplet_v2_infonce_lossor(cfgs)
    else:
        raise ValueError(f"Unknown loss: {cfgs.loss_type}")
    
    lossor.to(device)
    return lossor