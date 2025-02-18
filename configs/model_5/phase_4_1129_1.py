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
        image_encoder_type='ResUNetmmseg',
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=5,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        render_encoder_type='ResUNetmmseg',
        render_encoder_out_layer=3,
        render_encoder_cfgs=dict(
            f_layer=5,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(
            p=3,
            eps=1e-6),
        render_aggregator_type='GeM',
        render_aggregator_cfgs=dict(
            p=3,
            eps=1e-6),
        phase=4,
        pc_bev_encoder_type='ResUNetmmseg',
        pc_bev_encoder_out_layer=3,
        pc_bev_encoder_cfgs=dict(
            f_layer=5,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        
        image_bev_encoder_type='ResUNetmmseg',
        image_bev_encoder_out_layer=3,
        image_bev_encoder_cfgs=dict(
            f_layer=5,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        
        two_aggr=False,
        phase4_aggregator_type='GeM',
        phase4_aggregator_cfgs=dict(
            p=3,
            eps=1e-6),
        phase4_pc_bev_aggr_layer=1,
        phase4_image_bev_aggr_layer=1,
    ),
)

dataset_cfgs=dict(
    data_type='kitti',
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    train_pc_aug_mode=0,
    eval_pc_aug_mode=0,
    train_image_aug_mode=17,
    eval_image_aug_mode=17,
    raw_dir_name='dataset',
    pc_dir_name=None,
    image_dir_name=None,
    image_size=[128, 768],
    train_coords_filename='train_UTM_coords_v1_10m.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    use_cloud=False,
    use_image=False,
    use_range=False,
    use_pc_bev=True,
    use_image_bev=True,
    pc_bev_dir_name='16384_to_4096_cliped_fov_bev',
    image_bev_dir_name='768x128_image_bev',
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
resnet1_path=None
resnet2_path=None

loss_cfgs = dict(
    loss_type='v2_triplet_cm',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True,
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
            backbone_part=5
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0
        ),
        render_path=resnet2_path,
        render_cfgs=dict(
            backbone_part=5
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
            aggregate=1,
        ),
        freeze_render_encoder=dict(
            backbone=5,
        ),
        freeze_render_aggregator=dict(
            aggregate=1,
        ),
    )
)

use_mp=True
evaluator_type = 5
find_unused_parameters=False
all_gather_cfgs=dict(
    all_gather_flag=False,
    types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
)
accumulate_iter=1
