dataset_cfgs = dict(
    data_type='boreas_v2',
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    raw_dir_name='Boreas_minuse',
    pc_dir_name='Boreas_minuse_40960_to_4096_cliped_fov',
    rendered_dir_name=None,
    image_dir_name='Boreas_224x224_image',
    mask_dir_name=None,
    tool_name='my_tool',
    train_coords_filename='train_point_cloud_mnn.npy',
    val_query_filename='class_split_test_queries.pickle',
    eval_query_filename='class_split_test_queries.pickle',
    minuse_lidar_filename='train_minuse_lidar_idx.pickle',
    positive_distance=10.0,
    non_negative_distance=50.0,
    positive_distance_list=[
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ],
    lidar2image_filename='lidar2image.pickle',
    class_num_path='class_split_num.pickle',
    image_size=[224, 224],
    render_size=[224, 224],
    mask_size=[224, 224],
    pc_preprocess=dict(mode=0),
    use_cloud=True,
    use_render=False,
    use_image=True,
    use_mask=False,
    num_points=4096,
    ratio_strategy='mean',
    relative_strategy=True,
    use_overlap_ratio=True,
    rgb_depth_label_dir_name=None,
    use_rgb_depth_label=False,
    crop_location=dict(x_min=0, x_max=2448, y_min=683, y_max=1366),
    img_neighbor_num=1,
    overlap_ratio_type='area_overlap_ratio',
    use_original_pc_correspondence=False,
    generate_overlap_mask=True,
    negative_mask_overlap_threshld=0.1,
    positive_mask_overlap_threshld=0.3)
sampler_cfgs = dict(sampler_type='none')
batch_sampler_cfgs = dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v16',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    useout_times=1)
collect_fn_cfgs = dict(collect_fn_type='4')
dataloader_cfgs = dict(
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs)
optimizer_cfgs = dict(optimizer_type='adam', lr=1e-05, weight_decay=0.0001)
scheduler_cfgs = dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-07,
    T_mult=2)
start_epoch = 0
epoch = 80
save_interval = 1
val_interval = 3
train_val = False
need_eval = True
eval_interval = 2
evaluator_type = 1
use_mp = True
accumulate_iter = 1
find_unused_parameters = False
all_gather_cfgs = dict(all_gather_flag=False)
model_cfgs = dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='ResUNetmmseg',
        image_out_layer=None,
        pc_out_layer=None,
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=3,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        pc_encoder_type='PointNext',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            in_channels=4,
            width=32,
            radius=3.5,
            nsample=32,
            dropout=0.5,
            pointnextv3_op=1,
            f_layer=6,
            c_layer=5),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(p=3, eps=1e-06),
        pc_aggregator_type='GeM',
        pc_aggregator_cfgs=dict(p=3, eps=1e-06),
        phase=17,
        phase17_use_four_aggregator=False,
        phase17_use_SA_block=False,
        phase17_aggregator_type='GeM',
        phase17_aggregator_cfgs=dict(p=3, eps=1e-06)))
loss_cfgs = dict(
    loss_type='v2_triplet_cm',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True)
dgcnn_path = None
resnet_path = "/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
cmvpr_path = None
pretrained_cfgs = dict(
    cmvprnet=dict(
        image_path=resnet_path,
        image_cfgs=dict(backbone_part=5),
        image_aggregator_cfgs=dict(aggregate_part=0),
        pc_path=dgcnn_path,
        pc_cfgs=dict(backbone_part=2),
        pc_aggregator_cfgs=dict(aggregate_part=0),
        cmvpr_path=None))
freeze_cfgs = dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(backbone=3),
        freeze_img_aggregator=dict(aggregate=0),
        freeze_pc_encoder=dict(backbone=7),
        freeze_pc_aggregator=dict(aggregate=0)))
