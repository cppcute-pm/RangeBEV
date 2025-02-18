_base_ = ['../base/datasets/boreas_v2.py',
            '../base/losses/v2_triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=5,
    out_dim=256,
    cmvpr2_cfgs=dict(
        image_out_layer=1,
        render_out_layer=1,
        image_encoder_type='ResUNetmmseg',
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=3,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        render_encoder_type='ResUNetmmseg',
        render_encoder_out_layer=3,
        render_encoder_cfgs=dict(
            f_layer=3,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        image_aggregator_type='NetVLAD',
        image_aggregator_cfgs=dict(
            gating=True,
            add_batch_norm=False,
            cluster_size=64,),
        render_aggregator_type='NetVLAD',
        render_aggregator_cfgs=dict(
            gating=True,
            add_batch_norm=False,
            cluster_size=64,),
        phase=1,
    ),
)

dataset_cfgs=dict(
    data_type='kitti',
    train_range_aug_mode=17,
    eval_range_aug_mode=17,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    train_pc_aug_mode=0,
    eval_pc_aug_mode=0,
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    raw_dir_name='dataset',
    pc_dir_name='16384_to_4096_cliped_fov',
    image_dir_name='768x128_image',
    range_dir_name='16384_to_4096_cliped_fov_range_image',
    train_coords_filename='train_UTM_coords.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    image_size=[128, 768],
    range_img_size=[64, 224],
    use_cloud=False,
    use_image=True,
    use_range=True,
    num_points=4096,
    use_overlap_ratio=False,
    crop_location=None,
    dist_caculation_type='all_coords_L2',
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

# cct1_path="ntxoqrsy/epoch_73.pth"
# cct2_path="0ekfl6l8/epoch_63.pth"
resnet1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
resnet2_path=None

loss_cfgs = dict(
    loss_type='triplet_cm',
    margin=0.2,
    normalize_embeddings=False,
    hard_mining=True,
    pair_dist_info=True,
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
            backbone=3,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_render_encoder=dict(
            backbone=3,
        ),
        freeze_render_aggregator=dict(
            aggregate=0,
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
