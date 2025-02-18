import math
from datasets.utils import make_collate_fn
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import MinkowskiEngine as ME


def initialize_netvlad_layer(cfgs, netvlad_cfgs, device, dataset, backbone_img, backbone_pc, netvlader):
    if not netvlad_cfgs.initialize:
        return
    num_workers = 16
    batch_size = 16
    type = netvlad_cfgs.type
    features_dim = netvlader.feature_size
    descriptors_num = netvlad_cfgs.descriptors_num # 50000
    descs_num_per_sample = netvlad_cfgs.descs_num_per_sample # 100
    normalize_flag = netvlad_cfgs.normalize_flag
    modal_num = math.ceil(descriptors_num / descs_num_per_sample)
    random_sampler = SubsetRandomSampler(np.random.choice(len(dataset), modal_num, replace=False))
    collate_fn = make_collate_fn(dataset)
    random_dl = DataLoader(dataset=dataset, num_workers=num_workers,
                            batch_size=batch_size, sampler=random_sampler, collate_fn = collate_fn,)
    with torch.no_grad():
        if type == 'image' or type == 'both':
            backbone_img.eval()
            descriptors_img = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
        if type == 'pc'or type == 'both':
            backbone_pc.eval()
            descriptors_pc = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
        print("Extracting features to initialize NetVLAD layer")
        
        for iteration, inputs in enumerate(random_dl):
            if type == 'image' or type == 'both':
                img_inputs = inputs['images'].to(device)
                img_outputs = backbone_img(img_inputs)
                if normalize_flag:
                    img_norm_outputs = F.normalize(img_outputs, p=2, dim=1)
                else:
                    img_norm_outputs = img_outputs
                image_descriptors = img_norm_outputs.view(img_norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * 1 * descs_num_per_sample
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_sample, replace=False)
                    startix = batchix + ix * descs_num_per_sample
                    descriptors_img[startix:startix + descs_num_per_sample, :] = image_descriptors[ix, sample, :]
            if type == 'pc' or type == 'both':
                another_input_list = [pc.type(torch.float32) for pc in inputs['clouds']]
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
                    pc_inputs = ME.TensorField(features=feats,
                                        coordinates=coords,
                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                        device=device)
                else:
                    pc_inputs = torch.stack(another_input_list, dim=0).to(device)
                pc_outputs, _ = backbone_pc(pc_inputs)
                if normalize_flag:
                    pc_norm_outputs = F.normalize(pc_outputs, p=2, dim=1)
                else:
                    pc_norm_outputs = pc_outputs
                pc_descriptors = pc_norm_outputs.permute(0, 2, 1).cpu().numpy()
                batchix = iteration * 1 * descs_num_per_sample
                for ix in range(pc_descriptors.shape[0]):
                    sample = np.random.choice(pc_descriptors.shape[1], descs_num_per_sample, replace=False)
                    startix = batchix + ix * descs_num_per_sample
                    descriptors_pc[startix:startix + descs_num_per_sample, :] = pc_descriptors[ix, sample, :]
    if type == 'image':
        descriptors = descriptors_img
    elif type == 'pc':
        descriptors = descriptors_pc
    elif type == 'both':
        descriptors = np.concatenate((descriptors_img, descriptors_pc), axis=0)


    kmeans = faiss.Kmeans(features_dim, netvlader.cluster_size, niter=100, verbose=False)
    kmeans.train(descriptors)
    print(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
    netvlader.init_params(kmeans.centroids, descriptors)