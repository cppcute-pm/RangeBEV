import torch
import os
import torch.nn as nn
from .pretrained_dp import pe_check, fc_check, state_dict_filter_and_generator, state_dict_prefix_adder, state_dict_deleteor, state_dict_preffix_updator
import copy

def load_cct(part, state_dict, model: nn.Module):
    state_dict = pe_check(model, state_dict)
    state_dict = fc_check(model, state_dict)
    # check patch_embed, if not match， then delete 'patch_embed'
    patch_embed_pretrained = state_dict['tokenizer.conv_layers.0.0.weight']
    Nc1 = patch_embed_pretrained.shape[1]
    Nc2 = model.tokenizer.conv_layers[0][0].weight.shape[1]
    if (Nc1 != Nc2):
        del state_dict['tokenizer.conv_layers.0.0.weight']
        del state_dict['tokenizer.conv_layers.0.0.bias']
        
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_res18(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_resfpnmmseg(part, state_dict, model: nn.Module):
    if part == 0:
        state_dict = state_dict_deleteor(state_dict, 'decode_head.conv_seg')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    elif part == 2:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        state_dict = state_dict_preffix_updator(state_dict, 'c_fc', 'layer1')
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_unetmmseg(part, state_dict, model: nn.Module):
    if part == 0:
        state_dict = state_dict_deleteor(state_dict, 'decode_head')
        state_dict = state_dict_deleteor(state_dict, 'auxiliary_head')
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_resunetmmseg(part, state_dict, model: nn.Module):
    if part == 0:
        state_dict = state_dict_deleteor(state_dict, 'neck')
        state_dict = state_dict_deleteor(state_dict, 'decode_head')
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        state_dict = state_dict_preffix_updator(state_dict, 'c_fc', 'layer1')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 2:
        state_dict = state_dict_deleteor(state_dict, 'neck')
        state_dict = state_dict_deleteor(state_dict, 'decode_head')
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        state_dict = state_dict_preffix_updator(state_dict, 'c_fc', 'layer1')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 3:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 4:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 5:
        state_dict = state_dict_deleteor(state_dict, 'neck')
        state_dict = state_dict_deleteor(state_dict, 'decode_head')
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_dgcnn(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 2:
        state_dict = state_dict_deleteor(state_dict, 'c_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 3:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        state_dict = state_dict_deleteor(state_dict, 'c_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_fpt(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        state_dict
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_pointmlp(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_minkunet(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_pointnext(part, state_dict, model: nn.Module):
    if part == 0:
        state_dict = state_dict_deleteor(state_dict, 'head')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 1:
        msg = model.load_state_dict(state_dict, strict=False)
    elif part == 2:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    elif part is None:
        msg = None
    elif part == 3:
        state_dict = state_dict_deleteor(state_dict, 'f_fc')
        state_dict = state_dict_preffix_updator(state_dict, 'c_fc', 'layer1_fc')
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_img_encoder_pretrained_weights(state_dict, cfgs, backbone_type, model):
    backbone_msg = {}
    state_dict = state_dict_filter_and_generator(state_dict, 'module', True)
    if backbone_type == 'Res18':
        backbone_msg[backbone_type] = load_res18(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'CCT':
        backbone_msg[backbone_type] = load_cct(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'ResFPNmmseg' or backbone_type == 'ResFPNmmsegv2':
        backbone_msg[backbone_type] = load_resfpnmmseg(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'UNetmmseg':
        backbone_msg[backbone_type] = load_unetmmseg(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'ResUNetmmseg' or backbone_type == 'ResUNetmmsegv2' or backbone_type == 'ResUNetmmsegv3' or backbone_type == 'ResUNetmmsegv4':
        backbone_msg[backbone_type] = load_resunetmmseg(cfgs.backbone_part, state_dict, model.module)
    else:
        raise ValueError("Invalid backbone type")
    return backbone_msg

def load_gem(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=True)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_netvlad(part, state_dict, model: nn.Module):
    if part == 0:
        msg = model.load_state_dict(state_dict, strict=True)
    elif part is None:
        msg = None
    else:
        raise ValueError("Invalid part")
    
    return msg

def load_imgnet_pretrained_weights(cfgs, imgnet_cfgs, weight_root, model):
    imgnet_msg = {}
    if cfgs.path is None:
        return
    elif cfgs.path.startswith('/'):
        backbone_state_dict = torch.load(cfgs.path, map_location='cpu')
        backbone_state_dict = state_dict_prefix_adder(backbone_state_dict, 'module')
        imgnet_msg['img_backbone'] = load_img_encoder_pretrained_weights(backbone_state_dict, cfgs, imgnet_cfgs.backbone_type, model.backbone)
    else:
        state_dict = torch.load(os.path.join(weight_root, cfgs.path), map_location='cpu')
        state_dict = state_dict['model']
        backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
        aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
        imgnet_msg['img_backbone'] = load_img_encoder_pretrained_weights(backbone_state_dict, cfgs, imgnet_cfgs.backbone_type, model.backbone)
        imgnet_msg['img_aggregator'] = load_aggregator_pretrained_weights(aggregator_state_dict, cfgs, imgnet_cfgs.aggregate_type, model.aggregator)
    return imgnet_msg

def load_pc_encoder_pretrained_weights(state_dict, cfgs, backbone_type, model):
    backbone_msg = {}
    state_dict = state_dict_filter_and_generator(state_dict, 'module', True)
    if backbone_type == 'DGCNN':
        backbone_msg[backbone_type] = load_dgcnn(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'FPT':
        backbone_msg[backbone_type] = load_fpt(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'PointMLP':
        backbone_msg[backbone_type] = load_pointmlp(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'MinkUNet':
        backbone_msg[backbone_type] = load_minkunet(cfgs.backbone_part, state_dict, model.module)
    elif backbone_type == 'PointNext' or backbone_type == 'PointNextv2' or backbone_type == 'PointNextv3' or backbone_type == 'PointNextv4':
        backbone_msg[backbone_type] = load_pointnext(cfgs.backbone_part, state_dict, model.module)
    else:
        raise ValueError("Invalid backbone type")
    return backbone_msg

def load_rendernet_pretrained_weights(cfgs, rendernet_cfgs, weight_root, model):
    rendernet_msg = {}
    if cfgs.path is None:
        return
    elif cfgs.path.startswith('/'):
        backbone_state_dict = torch.load(cfgs.path, map_location='cpu')
        backbone_state_dict = state_dict_prefix_adder(backbone_state_dict, 'module')
        rendernet_msg['render_backbone'] = load_img_encoder_pretrained_weights(backbone_state_dict, cfgs, rendernet_cfgs.backbone_type, model.backbone)
    else:
        state_dict = torch.load(os.path.join(weight_root, cfgs.path), map_location='cpu')
        state_dict = state_dict['model']
        backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
        aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
        rendernet_msg['render_backbone'] = load_img_encoder_pretrained_weights(backbone_state_dict, cfgs, rendernet_cfgs.backbone_type, model.backbone)
        rendernet_msg['render_aggregator'] = load_aggregator_pretrained_weights(aggregator_state_dict, cfgs, rendernet_cfgs.aggregate_type, model.aggregator)
    return rendernet_msg

def load_aggregator_pretrained_weights(state_dict, cfgs, aggregate_type, model):
    aggregator_msg = {}
    state_dict = state_dict_filter_and_generator(state_dict, 'module', True)
    if aggregate_type == 'GeM' or aggregate_type == 'PoS_GeM':
        aggregator_msg[aggregate_type] = load_gem(cfgs.aggregate_part, state_dict, model.module)
    elif aggregate_type == 'NetVLAD':
        aggregator_msg[aggregate_type] = load_netvlad(cfgs.aggregate_part, state_dict, model.module)
    else:
        raise ValueError("Invalid aggregate type")
    return aggregator_msg

def load_pcnet_pretrained_weights(cfgs, pcnet_cfgs, weight_root, model):
    pcnet_msg = {}
    if cfgs.path is None:
        return
    elif cfgs.path.startswith('/'):
        backbone_state_dict = torch.load(cfgs.path, map_location='cpu')
        backbone_state_dict = state_dict_prefix_adder(backbone_state_dict, 'module')
        pcnet_msg['pc_backbone'] = load_pc_encoder_pretrained_weights(backbone_state_dict, cfgs, pcnet_cfgs.backbone_type, model.backbone)
    else:
        state_dict = torch.load(os.path.join(weight_root, cfgs.path), map_location='cpu')
        state_dict = state_dict['model']
        backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
        aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
        pcnet_msg['pc_backbone'] = load_pc_encoder_pretrained_weights(backbone_state_dict, cfgs, pcnet_cfgs.backbone_type, model.backbone)
        pcnet_msg['pc_aggregator'] = load_aggregator_pretrained_weights(aggregator_state_dict, cfgs, pcnet_cfgs.aggregate_type, model.aggregator)

    return pcnet_msg

def load_cmvpr_pretrained_weights(cfgs, cmvpr_cfgs, weight_root, model):
    cmvprnet_msg = {}
    if cfgs.image_path is None and cfgs.pc_path is None and ('cmvpr_path' not in cfgs.keys() or cfgs.cmvpr_path is None):
        return
    else:
        if cfgs.image_path is not None:
            if cfgs.image_path.startswith('/'):
                img_backbone_state_dict = torch.load(cfgs.image_path, map_location='cpu')
                img_backbone_state_dict = state_dict_prefix_adder(img_backbone_state_dict, 'module')
                cmvprnet_msg['img_backbone'] = load_img_encoder_pretrained_weights(img_backbone_state_dict, cfgs.image_cfgs, cmvpr_cfgs.image_encoder_type, model.image_encoder)
            else:
                state_dict = torch.load(os.path.join(weight_root, cfgs.image_path), map_location='cpu')
                state_dict = state_dict['model']
                img_backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
                img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
                if not img_aggregator_state_dict:
                    img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregate')
                cmvprnet_msg['img_backbone'] = load_img_encoder_pretrained_weights(img_backbone_state_dict, cfgs.image_cfgs, cmvpr_cfgs.image_encoder_type, model.image_encoder)
                cmvprnet_msg['img_aggregator'] = load_aggregator_pretrained_weights(img_aggregator_state_dict, cfgs.image_aggregator_cfgs, cmvpr_cfgs.image_aggregator_type, model.image_aggregator)
            
        if cfgs.pc_path is not None:
            if cfgs.pc_path.startswith('/'):
                pc_backbone_state_dict = torch.load(cfgs.pc_path, map_location='cpu')
                pc_backbone_state_dict = state_dict_prefix_adder(pc_backbone_state_dict, 'module')
                cmvprnet_msg['pc_backbone'] = load_pc_encoder_pretrained_weights(pc_backbone_state_dict, cfgs.pc_cfgs, cmvpr_cfgs.pc_encoder_type, model.pc_encoder)
            else:
                state_dict = torch.load(os.path.join(weight_root, cfgs.pc_path), map_location='cpu')
                state_dict = state_dict['model']
                pc_backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
                pc_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
                if not pc_aggregator_state_dict:
                    pc_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregate')
                cmvprnet_msg['pc_backbone'] = load_pc_encoder_pretrained_weights(pc_backbone_state_dict, cfgs.pc_cfgs, cmvpr_cfgs.pc_encoder_type, model.pc_encoder)
                cmvprnet_msg['pc_aggregator'] = load_aggregator_pretrained_weights(pc_aggregator_state_dict, cfgs.pc_aggregator_cfgs, cmvpr_cfgs.pc_aggregator_type, model.pc_aggregator)
        
        if 'cmvpr_path' in cfgs.keys() and cfgs.cmvpr_path is not None:
            state_dict = torch.load(os.path.join(weight_root, cfgs.cmvpr_path), map_location='cpu')
            state_dict = state_dict['model']
            img_encoder_state_dict = state_dict_filter_and_generator(state_dict, 'image_encoder')
            img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'image_aggregator')
            cmvprnet_msg['img_encoder'] = load_img_encoder_pretrained_weights(img_encoder_state_dict, cfgs.image_cfgs, cmvpr_cfgs.image_encoder_type, model.image_encoder)
            cmvprnet_msg['img_aggregator'] = load_aggregator_pretrained_weights(img_aggregator_state_dict, cfgs.image_aggregator_cfgs, cmvpr_cfgs.image_aggregator_type, model.image_aggregator)
            pc_encoder_state_dict = state_dict_filter_and_generator(state_dict, 'pc_encoder')
            pc_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'pc_aggregator')
            cmvprnet_msg['pc_encoder'] = load_pc_encoder_pretrained_weights(pc_encoder_state_dict, cfgs.pc_cfgs, cmvpr_cfgs.pc_encoder_type, model.pc_encoder)
            cmvprnet_msg['pc_aggregator'] = load_aggregator_pretrained_weights(pc_aggregator_state_dict, cfgs.pc_aggregator_cfgs, cmvpr_cfgs.pc_aggregator_type, model.pc_aggregator)

            if 'cmvpr_pretrained_aggregator' in cfgs.keys() and cfgs.cmvpr_pretrained_aggregator:
                curr_phase_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'phase7_aggregator')
                cmvprnet_msg['phase_aggregator'] = load_aggregator_pretrained_weights(curr_phase_aggregator_state_dict, cfgs.phase_aggregator_cfgs, 'GeM', model.phase18_aggregator)
        
        return cmvprnet_msg

def load_cmvpr2_pretrained_weights(cfgs, cmvpr2_cfgs, weight_root, model):
    cmvpr2net_msg = {}
    if cfgs.image_path is None and cfgs.render_path is None and ('cmvpr2_path' not in cfgs.keys() or cfgs.cmvpr2_path is None):
        return
    else:
        if cfgs.image_path is not None:
            if cfgs.image_path.startswith('/'):
                img_backbone_state_dict = torch.load(cfgs.image_path, map_location='cpu')
                img_backbone_state_dict = state_dict_prefix_adder(img_backbone_state_dict, 'module')
                cmvpr2net_msg['img_backbone'] = load_img_encoder_pretrained_weights(img_backbone_state_dict, cfgs.image_cfgs, cmvpr2_cfgs.image_encoder_type, model.image_encoder)
            else:
                state_dict = torch.load(os.path.join(weight_root, cfgs.image_path), map_location='cpu')
                state_dict = state_dict['model']
                img_backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
                img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
                if not img_aggregator_state_dict:
                    img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregate')
                cmvpr2net_msg['img_backbone'] = load_img_encoder_pretrained_weights(img_backbone_state_dict, cfgs.image_cfgs, cmvpr2_cfgs.image_encoder_type, model.image_encoder)
                cmvpr2net_msg['img_aggregator'] = load_aggregator_pretrained_weights(img_aggregator_state_dict, cfgs.image_aggregator_cfgs, cmvpr2_cfgs.image_aggregator_type, model.image_aggregator)
            
        if cfgs.render_path is not None:
            if cfgs.render_path.startswith('/'):
                render_backbone_state_dict = torch.load(cfgs.render_path, map_location='cpu')
                render_backbone_state_dict = state_dict_prefix_adder(render_backbone_state_dict, 'module')
                cmvpr2net_msg['render_backbone'] = load_img_encoder_pretrained_weights(render_backbone_state_dict, cfgs.render_cfgs, cmvpr2_cfgs.render_encoder_type, model.render_encoder)
            else:
                state_dict = torch.load(os.path.join(weight_root, cfgs.render_path), map_location='cpu')
                state_dict = state_dict['model']
                render_backbone_state_dict = state_dict_filter_and_generator(state_dict, 'backbone')
                render_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregator')
                if not render_aggregator_state_dict:
                    render_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'aggregate')
                cmvpr2net_msg['render_backbone'] = load_img_encoder_pretrained_weights(render_backbone_state_dict, cfgs.render_cfgs, cmvpr2_cfgs.render_encoder_type, model.render_encoder)
                cmvpr2net_msg['render_aggregator'] = load_aggregator_pretrained_weights(render_aggregator_state_dict, cfgs.render_aggregator_cfgs, cmvpr2_cfgs.render_aggregator_type, model.render_aggregator)
        
        if 'cmvpr2_path' in cfgs.keys() and cfgs.cmvpr2_path is not None:
            state_dict = torch.load(os.path.join(weight_root, cfgs.cmvpr2_path), map_location='cpu')
            state_dict = state_dict['model']
            img_encoder_state_dict = state_dict_filter_and_generator(state_dict, 'image_encoder')
            img_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'image_aggregator')
            cmvpr2net_msg['img_encoder'] = load_img_encoder_pretrained_weights(img_encoder_state_dict, cfgs.image_cfgs, cmvpr2_cfgs.image_encoder_type, model.image_encoder)
            cmvpr2net_msg['img_aggregator'] = load_aggregator_pretrained_weights(img_aggregator_state_dict, cfgs.image_aggregator_cfgs, cmvpr2_cfgs.image_aggregator_type, model.image_aggregator)
            render_encoder_state_dict = state_dict_filter_and_generator(state_dict, 'render_encoder')
            render_aggregator_state_dict = state_dict_filter_and_generator(state_dict, 'render_aggregator')
            cmvpr2net_msg['render_encoder'] = load_img_encoder_pretrained_weights(render_encoder_state_dict, cfgs.render_cfgs, cmvpr2_cfgs.render_encoder_type, model.render_encoder)
            cmvpr2net_msg['render_aggregator'] = load_aggregator_pretrained_weights(render_aggregator_state_dict, cfgs.render_aggregator_cfgs, cmvpr2_cfgs.render_aggregator_type, model.render_aggregator)

        
        return cmvpr2net_msg

def load_pretrained_weights(cfgs, model_cfgs, weight_root, model, loss_fn):
    pretrained_msg = {}
    if model_cfgs.modal_type == 1:
        pretrained_msg['imgnet'] = load_imgnet_pretrained_weights(cfgs.imgnet, model_cfgs.imgnet_cfgs, weight_root, model)
    elif model_cfgs.modal_type == 2:
        pretrained_msg['pcnet'] = load_pcnet_pretrained_weights(cfgs.pcnet, model_cfgs.pcnet_cfgs, weight_root, model)
    elif model_cfgs.modal_type == 3:
        pretrained_msg['cmvpr'] = load_cmvpr_pretrained_weights(cfgs.cmvprnet, model_cfgs.cmvpr_cfgs, weight_root, model)
    elif model_cfgs.modal_type == 4:
        pretrained_msg['rendernet'] = load_rendernet_pretrained_weights(cfgs.rendernet, model_cfgs.rendernet_cfgs, weight_root, model)
    elif model_cfgs.modal_type == 5:
        pretrained_msg['cmvpr'] = load_cmvpr2_pretrained_weights(cfgs.cmvpr2net, model_cfgs.cmvpr2_cfgs, weight_root, model)
    else:
        raise ValueError("Invalid modal type")
    
    return pretrained_msg