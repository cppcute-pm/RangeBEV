_base_ = ['../base/datasets/boreas_v2.py',
            '../base/losses/v2_triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=5,
    out_dim=256,
    cmvpr2_cfgs=dict(
        image_out_layer=None,
        render_out_layer=None,
        image_encoder_type='ResFPNmmseg',
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=4,
            c_layer=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            backbone=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            decode_head=dict(
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0,
                ),
            ),
            new_definition=True,),
        render_encoder_type='ResFPNmmseg',
        render_encoder_out_layer=3,
        render_encoder_cfgs=dict(
            f_layer=4,
            c_layer=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            backbone=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            decode_head=dict(
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0,
                ),
            ),
            new_definition=True,),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(
            p=3,
            eps=1e-6),
        render_aggregator_type='GeM',
        render_aggregator_cfgs=dict(
            p=3,
            eps=1e-6),
        phase=5,
        pc_bev_encoder_type='ResFPNmmseg',
        pc_bev_encoder_out_layer=3,
        pc_bev_encoder_cfgs=dict(
            f_layer=4,
            c_layer=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            backbone=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            decode_head=dict(
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0,
                ),
            ),
            new_definition=True,),
        image_bev_encoder_type='ResFPNmmseg',
        image_bev_encoder_out_layer=3,
        image_bev_encoder_cfgs=dict(
            f_layer=4,
            c_layer=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            backbone=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            decode_head=dict(
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0,
                ),
            ),
            new_definition=True,),
        
        two_aggr=False,
        phase5_aggregator1_type='GeM',
        phase5_aggregator1_cfgs=dict(
            p=3,
            eps=1e-6),
        phase5_aggregator2_type='GeM',
        phase5_aggregator2_cfgs=dict(
            p=3,
            eps=1e-6),
        phase5_aggregator_feature_output_type='split',
        phase5_attention=False,
        phase5_pc_bev_aggr_layer=1,
        phase5_image_bev_aggr_layer=1,
        phase5_render_aggr_layer=1,
        phase5_image_aggr_layer=1,
    ),
)

dataset_cfgs=dict(
    data_type='kitti',
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    train_pc_aug_mode=0,
    eval_pc_aug_mode=0,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    train_range_aug_mode=17,
    eval_range_aug_mode=17,
    raw_dir_name='dataset',
    pc_dir_name=None,
    image_dir_name='768x128_image',
    range_dir_name='16384_to_4096_cliped_fov_range_image',
    image_size=[128, 768],
    range_img_size=[64, 224],
    train_coords_filename='train_UTM_coords_v1_10m.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    use_cloud=False,
    use_image=True,
    use_range=True,
    use_pc_bev=True,
    use_image_bev=True,
    pc_bev_dir_name='16384_to_4096_cliped_fov_bev_nonground',
    image_bev_dir_name='768x128_image_bev_v2_sober_17',
    train_pc_bev_aug_mode=17,
    eval_pc_bev_aug_mode=17,
    train_image_bev_aug_mode=17,
    eval_image_bev_aug_mode=17,
    pc_bev_img_size=[128, 128],
    image_bev_img_size=[128, 128],
    num_points=4096,
    use_overlap_ratio=True,
    crop_location=None,
    dist_caculation_type='all_coords_L2_mean',
    overlap_ratio_type='pose_dist_sim',
    pose_dist_threshold=7.5,
    train_sequence_list=[
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
    ],
    test_sequence_list=[
        '00', '02', '05', '06', '07', '08'
    ],
    true_neighbour_dist=10.0,
    positive_distance=10.0,
    non_negative_distance=10.0,
    use_original_pc_correspondence=False,
    generate_overlap_mask=True,
    negative_mask_overlap_threshld=0.01,
    positive_mask_overlap_threshld=0.5,
)

batch_sampler_cfgs = dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v19',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=5,
    useout_times=1)

sampler_cfgs=dict(
    sampler_type='none',
)

collect_fn_cfgs=dict(
    collect_fn_type='5',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

# resnet1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
resnet1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
resnet2_path=None

loss_cfgs = dict(
    loss_type='two_triplet',
    margin=0.2,
    normalize_embeddings=False,
    hard_mining=True,
    pair_dist_info=False,
)

optimizer_cfgs = dict(optimizer_type='adam', lr=1e-05, weight_decay=0.0001)
scheduler_cfgs = dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-07,
    T_mult=2)

pretrained_cfgs=dict(
    cmvpr2net=dict(
        image_path=resnet1_path,
        image_cfgs=dict(
            backbone_part=0
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0
        ),
        render_path=resnet2_path,
        render_cfgs=dict(
            backbone_part=0
        ),
        render_aggregator_cfgs=dict(
            aggregate_part=0
        ),
    )
)

freeze_cfgs=dict(
    freeze_cmvpr2=dict(
        freeze_img_encoder=dict(
            backbone=5,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_render_encoder=dict(
            backbone=5,
        ),
        freeze_render_aggregator=dict(
            aggregate=0,
        ),
    )
)

use_mp=True
evaluator_type = 8
find_unused_parameters=False
all_gather_cfgs=dict(
    all_gather_flag=False,
    types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
)
accumulate_iter=1

evaluate_8_cfgs=dict(
    type=2,
    rank_weight1=0.5,
    rank_weight2=0.5,
)
