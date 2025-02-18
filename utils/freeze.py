def resfpn82_freeze(model, backbone):
    if backbone == 0:
        pass
    else:
        raise ValueError('Unknown ResFPN82 backbone: {}'.format(backbone))

def resfpn164_freeze(model, backbone):
    if backbone == 0:
        pass
    else:
        raise ValueError('Unknown ResFPN164 backbone: {}'.format(backbone))

def res18_freeze(model, backbone):
    if backbone == 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 1:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 2:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer2.parameters():
            parameter.requires_grad = True
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 3:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer2.parameters():
            parameter.requires_grad = True
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError('Unknown Res18 backbone: {}'.format(backbone)) 

def resfpnmmseg_freeze(model, backbone):
    if backbone == 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.neck.parameters():
            parameter.requires_grad = True
        for parameter in model.decode_head.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 1:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.neck.parameters():
            parameter.requires_grad = True
        for parameter in model.decode_head.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 2:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.decode_head.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 3:
        pass
    elif backbone == 4:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.neck.parameters():
            parameter.requires_grad = True
        for parameter in model.decode_head.parameters():
            parameter.requires_grad = True
        for parameter in model.layer1.parameters():
            parameter.requires_grad = True
        for parameter in model.layer2.parameters():
            parameter.requires_grad = True
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif backbone == 5:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 6:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 7:
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        raise ValueError('Unknown Res18 backbone: {}'.format(backbone)) 

def cct_freeze(model, backbone):
    if backbone == 0:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 11: # freeze before the 12th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 1:
        pass
    elif backbone == 2:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 7: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 3:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 9: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 4:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 13: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 5:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 11: # freeze before the 12th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 6:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 5: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 7:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 4:
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 8:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 3:
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 9:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 2:
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 10:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 1:
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 11:
        for params in model.f_fc.parameters():
            params.requires_grad = False
        for params in model.classifier.norm.parameters():
            params.requires_grad = False
        for params in model.classifier.attention_pool.parameters():
            params.requires_grad = False
    elif backbone == 12:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 6: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    elif backbone == 13:
        for p in model.parameters():
            p.requires_grad = False
        for name, child in model.classifier.blocks.named_children():
            if int(name) > 5: # freeze before the 8th block
                for params in child.parameters():
                    params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        if model.tokenizer.conv_layers[0][0].weight.shape[1] != 3:
            for params in model.tokenizer.parameters():
                params.requires_grad = True
    else:
        raise ValueError('Unknown CCT backbone: {}'.format(backbone))

def unetmmseg_freeze(model, backbone):
    if backbone == 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.encoder[3].parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.encoder[4].parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.decoder.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 1:
        pass
    elif backbone == 2:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.encoder.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 3:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.encoder[3].parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.encoder[4].parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError('Unknown Res18 backbone: {}'.format(backbone)) 

def resunetmmseg_freeze(model, backbone):
    if backbone == 0:
        pass
    elif backbone == 1:
        for parameter in model.decoder_head_1.last_layer.parameters():
            parameter.requires_grad = False
        for parameter in model.decoder_head_1.layer4.parameters():
            parameter.requires_grad = False
    elif backbone == 2:
        for parameter in model.decoder_head_1.last_layer.parameters():
            parameter.requires_grad = False
        for parameter in model.decoder_head_1.layer4.parameters():
            parameter.requires_grad = False
        for parameter in model.decoder_head_1.layer3.parameters():
            parameter.requires_grad = False
    elif backbone == 3:
        for parameter in model.decoder_head_1.parameters():
            parameter.requires_grad = False
    elif backbone == 4:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.layer3.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.c_fc.parameters():
            parameter.requires_grad = True
        for parameter in model.f_fc.parameters():
            parameter.requires_grad = True
    elif backbone == 5:
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        raise ValueError('Unknown Res18 backbone: {}'.format(backbone)) 

def fpt_freeze(model, backbone):
    if backbone == 0:
        pass
    elif backbone == 1:
        for names, child in model.named_children():
            if names == 'attn5p8':
                break
            for paramsss in child.parameters():
                paramsss.requires_grad = False
    elif backbone == 2:
        for names, child in model.named_children():
            if names == 'attn4p8':
                break
            for paramsss in child.parameters():
                paramsss.requires_grad = False
    else:
        raise ValueError('Unknown FPT backbone: {}'.format(backbone))

def dgcnn_freeze(model, backbone):
    if backbone == 0:
        pass
    elif backbone == 1:
        for names, child in model.named_children():
            if names == 'conv5':
                break
            for paramsss in child.parameters():
                paramsss.requires_grad = False
    elif backbone == 2:
        for names, child in model.named_children():
            if names == 'conv4':
                break
            for paramsss in child.parameters():
                paramsss.requires_grad = False
    elif backbone == 3:
        for params in model.f_fc.parameters():
            params.requires_grad = False
    elif backbone == 4:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
    elif backbone == 5:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.conv4.parameters():
            params.requires_grad = True
        for params in model.conv5.parameters():
            params.requires_grad = True
    elif backbone == 6:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.c_fc.parameters():
            params.requires_grad = False
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.conv3.parameters():
            params.requires_grad = True
        for params in model.conv2.parameters():
            params.requires_grad = True
    else:
        raise ValueError('Unknown DGCNN backbone: {}'.format(backbone))

def pointmlp_freeze(model, backbone):
    if backbone == 0:
        pass
    elif backbone == 1:
        for params in model.embedding.parameters():
            params.requires_grad = False
        for params in model.local_grouper_list[:-1].parameters():
            params.requires_grad = False
        for params in model.pre_blocks_list[:-1].parameters():
            params.requires_grad = False
        for params in model.pos_blocks_list[:-1].parameters():
            params.requires_grad = False
    elif backbone == 2:
        for params in model.embedding.parameters():
            params.requires_grad = False
        for params in model.local_grouper_list[:-2].parameters():
            params.requires_grad = False
        for params in model.pre_blocks_list[:-2].parameters():
            params.requires_grad = False
        for params in model.pos_blocks_list[:-2].parameters():
            params.requires_grad = False
    else:
        raise ValueError('Unknown DGCNN backbone: {}'.format(backbone))

def minkunet_freeze(model, backbone):
    if backbone == 0:
        pass
    elif backbone == 1:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
    elif backbone == 2:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.final.parameters():
            params.requires_grad = True
    elif backbone == 3:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.f_fc.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.final.parameters():
            params.requires_grad = True
        for params in model.block8.parameters():
            params.requires_grad = True
        for params in model.bntr7.parameters():
            params.requires_grad = True
        for params in model.convtr7p2s2.parameters():
            params.requires_grad = True

def pointnext_freeze(model, backbone):
    if backbone == 0:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.encoder.encoder[3].parameters():
            params.requires_grad = True
        for params in model.encoder.encoder[4].parameters():
            params.requires_grad = True
        for params in model.decoder.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
    elif backbone == 1:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.decoder.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
    elif backbone == 2:
        pass
    elif backbone == 3:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.encoder.encoder[3].parameters():
            params.requires_grad = True
        for params in model.encoder.encoder[4].parameters():
            params.requires_grad = True
        for params in model.decoder.parameters():
            params.requires_grad = True
        for params in model.layer1_fc.parameters():
            params.requires_grad = True
        for params in model.layer2_fc.parameters():
            params.requires_grad = True
        for params in model.layer3_fc.parameters():
            params.requires_grad = True
        for params in model.layer4_fc.parameters():
            params.requires_grad = True
    elif backbone == 4:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.encoder.parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
    elif backbone == 5:
        for params in model.decoder.decoder[0].parameters():
            params.requires_grad = False
    elif backbone == 6:
        for params in model.decoder.decoder[0].parameters():
            params.requires_grad = False
        for params in model.decoder.decoder[1].parameters():
            params.requires_grad = False
    elif backbone == 7:
        for params in model.decoder.parameters():
            params.requires_grad = False
    elif backbone == 8:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.encoder.encoder[3].parameters():
            params.requires_grad = True
        for params in model.encoder.encoder[4].parameters():
            params.requires_grad = True
        for params in model.c_fc.parameters():
            params.requires_grad = True
        for params in model.f_fc.parameters():
            params.requires_grad = True
    else:
        raise ValueError('Unknown Res18 backbone: {}'.format(backbone)) 

def gem_freeze(model, aggregate):
    if aggregate == 0:
        pass
    elif aggregate == 1:
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Unknown GeM aggregate: {}'.format(aggregate))

def pos_gem_freeze(model, aggregate):
    if aggregate == 0:
        pass
    elif aggregate == 1:
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Unknown PoS_GeM aggregate: {}'.format(aggregate))

def boq_freeze(model, aggregate):
    if aggregate == 0:
        pass
    elif aggregate == 1:
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Unknown BoQ aggregate: {}'.format(aggregate))

def netvlad_freeze(model, aggregate):
    if aggregate == 0:
        pass
    elif aggregate == 1:
        for param in model.cluster_weights.parameters():
            param.requires_grad = False
    elif aggregate == 2:
        for param in model.cluster_weights2.parameters():
            param.requires_grad = False
    elif aggregate == 3:
        for param in model.hidden1_weights.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Unknown NetVLAD aggregate: {}'.format(aggregate))

def make_freeze_pc_encoder(freeze_cfgs, model_cfgs, model, type='pcnet'):
    if type == 'pcnet':
        if model_cfgs.backbone_type == 'FPT':
            fpt_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'DGCNN':
            dgcnn_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'PointMLP':
            pointmlp_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'MinkUNet':
            minkunet_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'PointNext' or model_cfgs.backbone_type == 'PointNextv2' or model_cfgs.backbone_type == 'PointNextv3' or model_cfgs.backbone_type == 'PointNextv4':
            pointnext_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown backbone type: {}'.format(model_cfgs.backbone_type))
    elif type == 'cmvpr':
        if model_cfgs.pc_encoder_type == 'FPT':
            fpt_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.pc_encoder_type == 'DGCNN':
            dgcnn_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.pc_encoder_type == 'PointMLP':
            pointmlp_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.pc_encoder_type == 'MinkUNet':
            minkunet_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.pc_encoder_type == 'PointNext' or model_cfgs.pc_encoder_type == 'PointNextv2' or model_cfgs.pc_encoder_type == 'PointNextv3' or model_cfgs.pc_encoder_type == 'PointNextv4':
            pointnext_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown encoder type: {}'.format(model_cfgs.pc_encoder_type))
    else:
        raise ValueError('Unknown type: {}'.format(type))

def make_freeze_img_encoder(freeze_cfgs, model_cfgs, model, type='imgnet'):
    if type == 'imgnet':
        if model_cfgs.backbone_type == 'ResFPN82':
            resfpn82_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResFPN164':
            resfpn164_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'Res18':
            res18_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'CCT':
            cct_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResFPNmmseg' or model_cfgs.backbone_type == 'ResFPNmmsegv2':
            resfpnmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'UNetmmseg':
            unetmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResUNetmmseg' or model_cfgs.backbone_type == 'ResUNetmmsegv2' or model_cfgs.backbone_type == 'ResUNetmmsegv3' or model_cfgs.backbone_type == 'ResUNetmmsegv4':
            resunetmmseg_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown backbone type: {}'.format(model_cfgs.backbone_type))
    elif type == 'cmvpr':
        if model_cfgs.image_encoder_type == 'ResFPN82':
            resfpn82_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'ResFPN164':
            resfpn164_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'Res18':
            res18_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'CCT':
            cct_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'ResFPNmmseg' or model_cfgs.image_encoder_type == 'ResFPNmmsegv2':
            resfpnmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'UNetmmseg':
            unetmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.image_encoder_type == 'ResUNetmmseg' or model_cfgs.image_encoder_type == 'ResUNetmmsegv2' or model_cfgs.image_encoder_type == 'ResUNetmmsegv3' or model_cfgs.image_encoder_type == 'ResUNetmmsegv4':
            resunetmmseg_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown encoder type: {}'.format(model_cfgs.image_encoder_type))
    else:
        raise ValueError('Unknown type: {}'.format(type))

def make_freeze_render_encoder(freeze_cfgs, model_cfgs, model, type='rendernet'):
    if type == 'rendernet':
        if model_cfgs.backbone_type == 'ResFPN82':
            resfpn82_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResFPN164':
            resfpn164_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'Res18':
            res18_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'CCT':
            cct_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResFPNmmseg' or model_cfgs.backbone_type == 'ResFPNmmsegv2':
            resfpnmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.backbone_type == 'ResUNetmmseg' or model_cfgs.backbone_type == 'ResUNetmmsegv2' or model_cfgs.backbone_type == 'ResUNetmmsegv3' or model_cfgs.backbone_type == 'ResUNetmmsegv4':
            resunetmmseg_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown backbone type: {}'.format(model_cfgs.backbone_type))
    elif type == 'cmvpr2':
        if model_cfgs.render_encoder_type == 'ResFPN82':
            resfpn82_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.render_encoder_type == 'ResFPN164':
            resfpn164_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.render_encoder_type == 'Res18':
            res18_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.render_encoder_type == 'CCT':
            cct_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.render_encoder_type == 'ResFPNmmseg' or model_cfgs.render_encoder_type == 'ResFPNmmsegv2':
            resfpnmmseg_freeze(model.module, freeze_cfgs.backbone)
        elif model_cfgs.render_encoder_type == 'ResUNetmmseg' or model_cfgs.render_encoder_type == 'ResUNetmmsegv2' or model_cfgs.render_encoder_type == 'ResUNetmmsegv3' or model_cfgs.render_encoder_type == 'ResUNetmmsegv4':
            resunetmmseg_freeze(model.module, freeze_cfgs.backbone)
        else:
            raise ValueError('Unknown encoder type: {}'.format(model_cfgs.render_encoder_type))
    else:
        raise ValueError('Unknown type: {}'.format(type))

def make_freeze_aggregate(freeze_cfgs, model_cfgs, model, type=None):
    if type is None:
        if model_cfgs.aggregate_type.startswith('GeM'):
            gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.aggregate_type.startswith('NetVLAD'):
            netvlad_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.aggregate_type.startswith('PoS_GeM'):
            pos_gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.aggregate_type.startswith('BoQ'):
            boq_freeze(model.module, freeze_cfgs.aggregate)
        else:
            raise ValueError('Unknown aggregate type: {}'.format(model_cfgs.aggregate_type))
    elif type == 'img':
        if model_cfgs.image_aggregator_type.startswith('GeM'):
            gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.image_aggregator_type.startswith('NetVLAD'):
            netvlad_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.image_aggregator_type.startswith('PoS_GeM'):
            pos_gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.image_aggregator_type.startswith('BoQ'):
            boq_freeze(model.module, freeze_cfgs.aggregate)
        else:
            raise ValueError('Unknown aggregate type: {}'.format(model_cfgs.img_aggregator_type))
    elif type == 'pc':
        if model_cfgs.pc_aggregator_type.startswith('GeM'):
            gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.pc_aggregator_type.startswith('NetVLAD'):
            netvlad_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.pc_aggregator_type.startswith('PoS_GeM'):
            pos_gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.pc_aggregator_type.startswith('BoQ'):
            boq_freeze(model.module, freeze_cfgs.aggregate)
        else:
            raise ValueError('Unknown aggregate type: {}'.format(model_cfgs.pc_aggregator_type))
    elif type == 'render':
        if model_cfgs.render_aggregator_type.startswith('GeM'):
            gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.render_aggregator_type.startswith('NetVLAD'):
            netvlad_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.render_aggregator_type.startswith('PoS_GeM'):
            pos_gem_freeze(model.module, freeze_cfgs.aggregate)
        elif model_cfgs.render_aggregator_type.startswith('BoQ'):
            boq_freeze(model.module, freeze_cfgs.aggregate)
        else:
            raise ValueError('Unknown aggregate type: {}'.format(model_cfgs.render_aggregator_type))
    else:
        raise ValueError('Unknown type: {}'.format(type))

def make_freeze_cmvpr(freeze_cfgs, model_cfgs, model):
    make_freeze_img_encoder(freeze_cfgs.freeze_img_encoder, model_cfgs, model.image_encoder, 'cmvpr')
    make_freeze_pc_encoder(freeze_cfgs.freeze_pc_encoder, model_cfgs, model.pc_encoder, 'cmvpr')
    make_freeze_aggregate(freeze_cfgs.freeze_img_aggregator, model_cfgs, model.image_aggregator, 'img')
    make_freeze_aggregate(freeze_cfgs.freeze_pc_aggregator, model_cfgs, model.pc_aggregator, 'pc')

def make_freeze_cmvpr2(freeze_cfgs, model_cfgs, model):
    make_freeze_img_encoder(freeze_cfgs.freeze_img_encoder, model_cfgs, model.image_encoder, 'cmvpr')
    make_freeze_render_encoder(freeze_cfgs.freeze_render_encoder, model_cfgs, model.render_encoder, 'cmvpr2')
    make_freeze_aggregate(freeze_cfgs.freeze_img_aggregator, model_cfgs, model.image_aggregator, 'img')
    make_freeze_aggregate(freeze_cfgs.freeze_render_aggregator, model_cfgs, model.render_aggregator, 'render')

def make_freeze(freeze_cfgs, model_cfgs, model):

    if model_cfgs.modal_type == 1:
        make_freeze_img_encoder(freeze_cfgs.freeze_imgnet, model_cfgs.imgnet_cfgs, model.backbone, type='imgnet')
        make_freeze_aggregate(freeze_cfgs.freeze_imgnet, model_cfgs.imgnet_cfgs, model.aggregator, type=None)
    elif model_cfgs.modal_type == 2:
        make_freeze_pc_encoder(freeze_cfgs.freeze_pcnet, model_cfgs.pcnet_cfgs, model.backbone, type='pcnet')
        make_freeze_aggregate(freeze_cfgs.freeze_pcnet, model_cfgs.pcnet_cfgs, model.aggregator, type=None)
    elif model_cfgs.modal_type == 3:
        make_freeze_cmvpr(freeze_cfgs.freeze_cmvpr, model_cfgs.cmvpr_cfgs, model)
    elif model_cfgs.modal_type == 4:
        make_freeze_render_encoder(freeze_cfgs.freeze_rendernet, model_cfgs.rendernet_cfgs, model.backbone, type='rendernet')
    elif model_cfgs.modal_type == 5:
        make_freeze_cmvpr2(freeze_cfgs.freeze_cmvpr2, model_cfgs.cmvpr2_cfgs, model)
    else:
        raise ValueError('Unknown modal type: {}'.format(model_cfgs.modal_type))
